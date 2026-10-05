#!/usr/bin/env python3
"""
nav_error_monitor
------------------
Theo dõi sai số giữa vị trí thực của robot (qua TF map -> base_frame) và
toạ độ đích được HMI yêu cầu (topic 'hmi/requested_goal', PoseStamped), cùng
với vận tốc dài/góc thực đo được (topic 'odom', nav_msgs/Odometry).

- Trong lúc chạy: publish sai số ra các topic Float64 để xem trực tiếp bằng
  rqt_plot / PlotJuggler.
- Khi tới đích (sai số vị trí + góc nằm trong tolerance, giữ ổn định trong
  settle_time giây): tự lưu 1 file CSV (toàn bộ chuỗi sai số theo thời gian)
  vào log_dir, để nạp vào MATLAB/Excel/... vẽ biểu đồ riêng.

Chạy live plot, ví dụ:
    ros2 run rqt_plot rqt_plot /nav_error/xy_error /nav_error/yaw_error_deg
"""
import math
import csv
import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float64
import tf2_ros
from tf2_ros import TransformException
import tf_transformations


def yaw_from_quaternion(q):
    _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
    return yaw


def angle_diff(a, b):
    """a - b, chuẩn hoá về [-pi, pi]."""
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


def point_segment_distance(px, py, ax, ay, bx, by):
    """Khoảng cách từ điểm (px,py) tới đoạn thẳng (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def path_cross_track_error(path_points, px, py):
    """Khoảng cách nhỏ nhất từ (px,py) tới polyline path_points [(x,y), ...]."""
    if not path_points:
        return None
    if len(path_points) == 1:
        ax, ay = path_points[0]
        return math.hypot(px - ax, py - ay)
    return min(
        point_segment_distance(px, py, ax, ay, bx, by)
        for (ax, ay), (bx, by) in zip(path_points, path_points[1:])
    )


class NavErrorMonitor(Node):
    def __init__(self):
        super().__init__('nav_error_monitor')

        self.declare_parameter('goal_topic', 'hmi/requested_goal')
        self.declare_parameter('path_topic', '/plan')
        self.declare_parameter('odom_topic', 'diff_drive_controller/odom')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('sample_rate_hz', 10.0)
        self.declare_parameter('xy_goal_tolerance', 0.1)      # m
        self.declare_parameter('yaw_goal_tolerance_deg', 3.0)  # deg
        self.declare_parameter('settle_time', 0.5)            # s trong tolerance mới tính là "đã tới"
        self.declare_parameter('log_dir', os.path.expanduser('~/prestobot_nav_logs'))

        self.goal_topic_ = self.get_parameter('goal_topic').value
        self.path_topic_ = self.get_parameter('path_topic').value
        self.odom_topic_ = self.get_parameter('odom_topic').value
        self.map_frame_ = self.get_parameter('map_frame').value
        self.base_frame_ = self.get_parameter('base_frame').value
        self.xy_tol_ = self.get_parameter('xy_goal_tolerance').value
        self.yaw_tol_deg_ = self.get_parameter('yaw_goal_tolerance_deg').value
        self.settle_time_ = self.get_parameter('settle_time').value
        self.log_dir_ = self.get_parameter('log_dir').value
        os.makedirs(self.log_dir_, exist_ok=True)

        self.tf_buffer_ = tf2_ros.Buffer()
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_, self)

        self.goal_sub_ = self.create_subscription(
            PoseStamped, self.goal_topic_, self.goal_callback, 10)
        self.path_sub_ = self.create_subscription(
            Path, self.path_topic_, self.path_callback, 10)
        self.odom_sub_ = self.create_subscription(
            Odometry, self.odom_topic_, self.odom_callback, 10)

        self.xy_error_pub_ = self.create_publisher(Float64, 'nav_error/xy_error', 10)
        self.x_error_pub_ = self.create_publisher(Float64, 'nav_error/x_error', 10)
        self.y_error_pub_ = self.create_publisher(Float64, 'nav_error/y_error', 10)
        self.yaw_error_pub_ = self.create_publisher(Float64, 'nav_error/yaw_error_deg', 10)
        self.path_error_pub_ = self.create_publisher(Float64, 'nav_error/path_error', 10)
        self.linear_vel_pub_ = self.create_publisher(Float64, 'nav_error/linear_vel', 10)
        self.angular_vel_pub_ = self.create_publisher(Float64, 'nav_error/angular_vel', 10)

        self.current_goal_ = None
        self.goal_label_ = None
        self.run_start_time_ = None
        self.run_data_ = []       # list of dict, 1 dòng / sample
        self.reached_ = False
        self.reached_since_ = None
        self.run_index_ = 0

        self.current_path_points_ = None   # [(x,y), ...] từ /plan mới nhất
        self.linear_vel_ = 0.0   # m/s, cập nhật mỗi khi có /odom mới
        self.angular_vel_ = 0.0  # rad/s

        period = 1.0 / self.get_parameter('sample_rate_hz').value
        self.timer_ = self.create_timer(period, self.sample_callback)

        self.get_logger().info(
            f"nav_error_monitor: theo dõi '{self.goal_topic_}', "
            f"TF {self.map_frame_} -> {self.base_frame_}, "
            f"log tại {self.log_dir_}")

    # ------------------------------------------------------------------
    def goal_callback(self, msg: PoseStamped):
        if self.current_goal_ is not None and not self.reached_ and self.run_data_:
            self.get_logger().warn(
                "Nhận goal mới trước khi goal cũ đạt tolerance -> lưu lại run cũ (chưa tới đích).")
            self.finalize_run(reason='goal_changed')

        self.current_goal_ = msg
        self.run_start_time_ = time.monotonic()
        self.run_data_ = []
        self.reached_ = False
        self.reached_since_ = None
        self.current_path_points_ = None  # chờ planner_server phát /plan mới cho goal này
        self.run_index_ += 1
        self.get_logger().info(
            f"Goal mới #{self.run_index_}: x={msg.pose.position.x:.3f}, "
            f"y={msg.pose.position.y:.3f}, "
            f"yaw={math.degrees(yaw_from_quaternion(msg.pose.orientation)):.1f} deg")

    # ------------------------------------------------------------------
    def path_callback(self, msg: Path):
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if len(pts) < 2:
            return
        self.current_path_points_ = pts

    # ------------------------------------------------------------------
    def odom_callback(self, msg: Odometry):
        """Chỉ ghi lại giá trị mới nhất; sample_callback sẽ đọc khi lấy mẫu.
        Vận tốc thật đo được (không phải lệnh /cmd_vel), lấy từ topic thật
        diff_drive_controller publish ra (diff_drive_controller/odom) -
        KHÔNG phải 'odom' trần trụi, vì controller tên diff_drive_controller
        nên ros2_control publish theo quy ước <ten_controller>/odom."""
        self.linear_vel_ = msg.twist.twist.linear.x
        self.angular_vel_ = msg.twist.twist.angular.z

    # ------------------------------------------------------------------
    def sample_callback(self):
        if self.current_goal_ is None:
            return

        try:
            tf = self.tf_buffer_.lookup_transform(
                self.map_frame_, self.base_frame_, rclpy.time.Time(),
                timeout=Duration(seconds=0.1))
        except TransformException as ex:
            self.get_logger().warn(f"Chưa lấy được TF {self.map_frame_}->{self.base_frame_}: {ex}",
                                    throttle_duration_sec=2.0)
            return

        rx = tf.transform.translation.x
        ry = tf.transform.translation.y
        r_yaw = yaw_from_quaternion(tf.transform.rotation)

        gx = self.current_goal_.pose.position.x
        gy = self.current_goal_.pose.position.y
        g_yaw = yaw_from_quaternion(self.current_goal_.pose.orientation)

        x_err = gx - rx
        y_err = gy - ry
        xy_err = math.hypot(x_err, y_err)
        yaw_err_deg = math.degrees(angle_diff(g_yaw, r_yaw))

        path_err = None
        if self.current_path_points_ is not None:
            path_err = path_cross_track_error(self.current_path_points_, rx, ry)

        # publish live để xem bằng rqt_plot / PlotJuggler
        self.xy_error_pub_.publish(Float64(data=xy_err))
        self.x_error_pub_.publish(Float64(data=x_err))
        self.y_error_pub_.publish(Float64(data=y_err))
        self.yaw_error_pub_.publish(Float64(data=yaw_err_deg))
        if path_err is not None:
            self.path_error_pub_.publish(Float64(data=path_err))
        self.linear_vel_pub_.publish(Float64(data=self.linear_vel_))
        self.angular_vel_pub_.publish(Float64(data=self.angular_vel_))

        t = time.monotonic() - self.run_start_time_
        self.run_data_.append({
            't': t, 'rx': rx, 'ry': ry, 'r_yaw_deg': math.degrees(r_yaw),
            'gx': gx, 'gy': gy, 'g_yaw_deg': math.degrees(g_yaw),
            'x_err': x_err, 'y_err': y_err, 'xy_err': xy_err, 'yaw_err_deg': yaw_err_deg,
            'path_err': path_err if path_err is not None else '',
            'linear_vel': self.linear_vel_, 'angular_vel': self.angular_vel_,
        })

        in_tol = xy_err <= self.xy_tol_ and abs(yaw_err_deg) <= self.yaw_tol_deg_
        if in_tol:
            if self.reached_since_ is None:
                self.reached_since_ = time.monotonic()
            elif not self.reached_ and (time.monotonic() - self.reached_since_) >= self.settle_time_:
                self.reached_ = True
                self.finalize_run(reason='reached')
        else:
            self.reached_since_ = None

    # ------------------------------------------------------------------
    def finalize_run(self, reason: str):
        if not self.run_data_:
            return
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"run{self.run_index_:03d}_{stamp}_{reason}"
        csv_path = os.path.join(self.log_dir_, base_name + '.csv')

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.run_data_[0].keys()))
            writer.writeheader()
            writer.writerows(self.run_data_)

        last = self.run_data_[-1]
        self.get_logger().info(
            f"[Run #{self.run_index_}] {reason} sau {last['t']:.1f}s | "
            f"sai số cuối: xy={last['xy_err']:.3f} m, yaw={last['yaw_err_deg']:.1f} deg | "
            f"CSV: {csv_path}")


def main(args=None):
    rclpy.init(args=args)
    node = NavErrorMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
