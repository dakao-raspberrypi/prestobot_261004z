#!/usr/bin/env python3
"""
Prestobot HMI (Pygame) - revised state machine.

State machine
-------------
    STARTING ──ready──► IDLE ──pick room──► PENDING_CONFIRM ◄──pick another room──┐
                         ▲                     │      │                          │
                         │            Cancel ◄─┘      │ Confirm (close door)     │
                         │                            ▼                          │
                       Open (open door)          NAVIGATING ──Stop──► CANCELING  │
                         │                       │        │               │      │
                         │             succeeded │        │ failed        │      │
                         │          + verified   ▼        ▼               ▼      │
                         └──────────────────  ARRIVED   FAILED ◄──────────┘      │
                                                         │  │                    │
                                              Retry ◄────┘  └── pick room ───────┘
                                       (back to NAVIGATING)

Key rules
- The compartment door can ONLY be opened in ARRIVED, i.e. after Nav2 reported
  success AND the robot's real pose (TF) is within ARRIVAL_CHECK_TOLERANCE of
  the goal. A failed or cancelled run lands in FAILED, where the door stays
  closed; the operator can Retry, or pick another destination (e.g. Home).
- Room buttons are clickable in IDLE, PENDING_CONFIRM (to change the choice)
  and FAILED. A Cancel button clears a pending choice.
- All Nav2 calls (waitUntilNav2Active, goToPose, followWaypoints,
  isTaskComplete, cancelTask) run on a dedicated worker thread that owns the
  BasicNavigator. The Pygame loop never blocks on Nav2, so the window stays
  responsive even while waiting for the initial pose at startup.
- The current hallway is derived from the robot's real pose when available,
  not from the last requested goal.
- Hallway-junction waypoints are oriented toward the next point of the route,
  so the robot doesn't spin in place at each corner.
- Closing the window cancels any running navigation task.
"""
import math
import queue
import sys
import threading
import time
import traceback
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped
import tf_transformations
import tf2_ros
from tf2_ros import TransformException
import numpy as np
import pygame


# ---------------------------------------------------------------------------
# State machine definition
# ---------------------------------------------------------------------------
class State(Enum):
    STARTING = 'starting'                # waiting for Nav2 (+ AMCL initial pose)
    IDLE = 'idle'                        # nothing selected
    PENDING_CONFIRM = 'pending_confirm'  # room selected, robot not moving yet
    NAVIGATING = 'navigating'            # goal sent, robot moving, door closed
    CANCELING = 'canceling'              # Stop pressed, waiting for Nav2 to cancel
    ARRIVED = 'arrived'                  # verified arrival, door may be opened
    FAILED = 'failed'                    # failed / cancelled, door stays closed


ALLOWED_TRANSITIONS = {
    State.STARTING: {State.IDLE},
    State.IDLE: {State.PENDING_CONFIRM},
    State.PENDING_CONFIRM: {State.PENDING_CONFIRM, State.IDLE, State.NAVIGATING},
    State.NAVIGATING: {State.CANCELING, State.ARRIVED, State.FAILED},
    # A task may finish successfully right before the cancel takes effect.
    State.CANCELING: {State.ARRIVED, State.FAILED},
    State.ARRIVED: {State.IDLE},
    State.FAILED: {State.PENDING_CONFIRM, State.NAVIGATING},
}

# States in which the room buttons (incl. Home) can be clicked.
ROOMS_ENABLED_STATES = {State.IDLE, State.PENDING_CONFIRM, State.FAILED}


# ---------------------------------------------------------------------------
# Map geometry (frame 'map', meters)
# ---------------------------------------------------------------------------
# Each hallway as a line segment along its centerline. A point belongs to the
# hallway whose segment is nearest. This replaces the hand-written
# room-number -> hallway table, so there is a single source of truth: the
# coordinates themselves.
CORRIDORS = {
    1: ((0.0, 0.0), (66.0, 0.0)),    # bottom hallway (Home, rooms 1-14, 34-41)
    2: ((8.0, 0.0), (8.0, 40.0)),    # vertical hallway (rooms 15-22, 42-47)
    3: ((8.0, 40.0), (66.0, 40.0)),  # top hallway (rooms 23-33)
}
# Junction point between two adjacent hallways.
JUNCTIONS = {
    frozenset((1, 2)): (8.0, 0.0),
    frozenset((2, 3)): (8.0, 40.0),
}

# After Nav2 reports success, the robot must be at most this far (m) from the
# goal according to TF; otherwise the run is treated as FAILED. Nav2's own
# xy_goal_tolerance is 0.1 m; this is a looser sanity check against AMCL jitter.
ARRIVAL_CHECK_TOLERANCE = 0.5
# A TF pose older than this (s) is considered stale for decisions.
POSE_MAX_AGE = 1.0


def _point_segment_distance(p, a, b):
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    if seg_len2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def hall_of_point(x, y):
    """Hallway (1, 2 or 3) whose centerline is nearest to (x, y)."""
    return min(CORRIDORS, key=lambda h: _point_segment_distance((x, y), *CORRIDORS[h]))


def plan_junction_points(start_hall, dest_xy):
    """
    Junction waypoints needed to go from start_hall to the hallway of dest_xy,
    as a list of (x, y, yaw). Each yaw points toward the next point of the
    route (next junction or the destination), so the robot arrives at a corner
    already facing where it's going next.
    """
    dest_hall = hall_of_point(*dest_xy)
    if start_hall is None or start_hall == dest_hall:
        return []
    step = 1 if dest_hall > start_hall else -1
    halls = list(range(start_hall, dest_hall + step, step))
    points = [JUNCTIONS[frozenset((a, b))] for a, b in zip(halls, halls[1:])]
    targets = points + [tuple(dest_xy)]
    result = []
    for i, (px, py) in enumerate(points):
        nx, ny = targets[i + 1]
        result.append((px, py, math.atan2(ny - py, nx - px)))
    return result


# ---------------------------------------------------------------------------
# Navigation worker: the ONLY thread that touches BasicNavigator
# ---------------------------------------------------------------------------
class NavWorker(threading.Thread):
    """
    BasicNavigator spins itself inside its own methods (isTaskComplete() blocks
    for up to 0.1 s, cancelTask() blocks until the server answers), so it is
    kept off the UI thread. The UI talks to this thread only through
    submit() / request_cancel() / request_shutdown() and reads results
    with poll_event().

    Events put on the queue:
      ('ready', None)       Nav2 is active, AMCL has an initial pose
      ('error', str)        Nav2 startup failed
      ('done', dict)        task finished: {'result', 'canceled', 'reason'}
    """

    def __init__(self, navigator, logger):
        super().__init__(daemon=True, name='nav_worker')
        self.navigator_ = navigator
        self.logger_ = logger
        self.commands_ = queue.Queue()
        self.events_ = queue.Queue()
        self.cancel_requested_ = threading.Event()
        self.shutdown_requested_ = threading.Event()
        self.busy_ = threading.Event()

    # ---- API used by the UI thread ----
    def submit(self, waypoints):
        """waypoints: list of PoseStamped, last one is the destination."""
        self.cancel_requested_.clear()
        self.commands_.put(list(waypoints))

    def request_cancel(self):
        self.cancel_requested_.set()

    def request_shutdown(self):
        self.shutdown_requested_.set()
        self.cancel_requested_.set()

    def is_busy(self):
        return self.busy_.is_set()

    def poll_event(self):
        try:
            return self.events_.get_nowait()
        except queue.Empty:
            return None

    # ---- worker thread ----
    def run(self):
        try:
            # Blocks until Nav2 lifecycle nodes are active AND AMCL has an
            # initial pose (set_initial_pose: false -> set it in RViz).
            self.navigator_.waitUntilNav2Active()
        except Exception:
            if not self.shutdown_requested_.is_set():
                self.events_.put(('error', traceback.format_exc()))
            return
        if self.shutdown_requested_.is_set():
            return
        self.events_.put(('ready', None))

        while not self.shutdown_requested_.is_set():
            try:
                waypoints = self.commands_.get(timeout=0.1)
            except queue.Empty:
                continue
            self.busy_.set()
            try:
                outcome = self._execute(waypoints)
            except Exception:
                self.logger_.error('Navigation task crashed:\n' + traceback.format_exc())
                outcome = {'result': TaskResult.FAILED, 'canceled': False,
                           'reason': 'internal error (see log)'}
            finally:
                self.busy_.clear()
            self.events_.put(('done', outcome))

    def _execute(self, waypoints):
        if len(waypoints) == 1:
            accepted = self.navigator_.goToPose(waypoints[0])
        else:
            accepted = self.navigator_.followWaypoints(waypoints)
        # Most versions return True/False; some older ones return None.
        if accepted is False:
            return {'result': TaskResult.FAILED, 'canceled': False,
                    'reason': 'goal rejected by Nav2'}

        cancel_sent = False
        while not self.navigator_.isTaskComplete():   # blocks <= 0.1 s per call
            if self.cancel_requested_.is_set() and not cancel_sent:
                self.navigator_.cancelTask()
                cancel_sent = True

        result = self.navigator_.getResult()
        reason = None
        # With stop_on_failure: false, waypoint_follower skips failed waypoints
        # and may still report SUCCEEDED. Treat a missed FINAL waypoint as failure.
        if (result == TaskResult.SUCCEEDED and len(waypoints) > 1
                and self._final_waypoint_missed(len(waypoints) - 1)):
            result = TaskResult.FAILED
            reason = 'destination waypoint was skipped'
        return {'result': result, 'canceled': cancel_sent, 'reason': reason}

    def _final_waypoint_missed(self, final_index):
        # Reads the raw action result. The field is int32[] in older Nav2 and a
        # list of MissedWaypoint (with .index) in newer ones; handle both. If
        # the field isn't available, fall back to the TF distance check in the UI.
        try:
            missed = self.navigator_.result_future.result().result.missed_waypoints
        except Exception:
            return False
        return any(getattr(m, 'index', m) == final_index for m in missed)


# ---------------------------------------------------------------------------
# HMI node
# ---------------------------------------------------------------------------
class HmiNode(Node):
    def __init__(self):
        super().__init__("hmi_node")

        # Pygame Initialization (driven by the main thread loop)
        pygame.init()
        self.screen_ = pygame.display.set_mode((1200, 700))
        pygame.display.set_caption('Prestobot HMI')

        self.WHITE_ = (255, 255, 255)
        self.BLACK_ = (0, 0, 0)
        self.BLUE_ = (173, 216, 230)
        self.GREEN_ = (144, 238, 144)
        self.GRAY_ = (200, 200, 200)
        self.DARK_GRAY_ = (150, 150, 150)
        self.ORANGE_ = (255, 178, 102)
        self.RED_ = (255, 130, 130)
        self.font1_ = pygame.font.SysFont('sans', 25)
        self.font_small_ = pygame.font.SysFont('sans', 20)

        # Publishes the exact goal pose sent to Nav2, so external nodes
        # (e.g. nav_error_monitor) can know what the robot was asked to reach.
        self.goal_request_pub_ = self.create_publisher(PoseStamped, 'hmi/requested_goal', 10)

        # TF: robot's real position (red dot + hallway detection + arrival check)
        self.tf_buffer_ = tf2_ros.Buffer()
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_, self)
        self.pose_lock_ = threading.Lock()
        self.robot_world_pos_ = None   # (x, y) in 'map'; None until first TF
        self.robot_pose_time_ = None   # time.monotonic() of the last update
        self.create_timer(0.1, self.update_robot_pose)  # 10 Hz

        # State
        self.state_ = State.STARTING
        self.pending_label_ = None   # selected but not confirmed yet
        self.pending_pose_ = None
        self.target_label_ = None    # destination of the current/last run (for Retry)
        self.target_pose_ = None
        self.active_goal_button_ = None  # label of the button with the red marker
        # Fallback hallway when no fresh TF is available. Only updated after a
        # VERIFIED arrival; set to None after a failure (position unknown).
        self.current_hall_ = 1
        self.info_text_ = ("Waiting for Nav2 to become active... If AMCL has no initial "
                           "pose yet, set it with '2D Pose Estimate' in RViz.")

        # Layout: info box + two action buttons in the empty center area
        self.info_box_rect_ = pygame.Rect(450, 270, 650, 90)
        self.primary_button_rect_ = pygame.Rect(450, 390, 300, 70)
        self.secondary_button_rect_ = pygame.Rect(780, 390, 300, 70)

        self.room_coordinates = [(65.3, 0.0, -1.57), (58.4, 0.0, -1.57), (55.3, 0.0, -1.57), (48.4, 0.0, -1.57), (45.3, 0.0, -1.57), (38.4, 0.0, -1.57), (35.3, 0.0, -1.57), (28.4, 0.0, -1.57), (25.3, 0.0, -1.57), (18.4, 0.0, -1.57), (15.3, 0.0, -1.57), (8.4, 0.0, -1.57), (5.6, 0.0, -1.57), (8.0, 2.4, 3.14), (8.0, 9.6, 3.14), (8.0, 12.4, 3.14), (8.0, 19.6, 3.14), (8.0, 22.4, 3.14), (8.0, 29.6, 3.14), (8.0, 32.4, 3.14), (8.0, 39.6, 3.14), (65.3, 40.0, -1.57), (58.4, 40.0, -1.57), (55.3, 40.0, -1.57), (48.4, 40.0, -1.57), (45.3, 40.0, -1.57), (38.4, 40.0, -1.57), (35.3, 40.0, -1.57), (28.4, 40.0, -1.57), (25.3, 40.0, -1.57), (18.4, 40.0, -1.57), (15.3, 40.0, -1.57), (30.3, 0.0, 1.57), (33.4, 0.0, 1.57), (40.3, 0.0, 1.57), (43.4, 0.0, 1.57), (50.3, 0.0, 1.57), (53.4, 0.0, 1.57), (60.3, 0.0, 1.57), (63.4, 0.0, 1.57), (8.0, 4.6, 0.0), (8.0, 7.4, 0.0), (8.0, 14.6, 0.0), (8.0, 17.4, 0.0), (8.0, 24.6, 0.0), (8.0, 27.4, 0.0)]
        self.room_button_positions = [(1, (1080, 620)), (2, (1000, 620)), (3, (920, 620)), (4, (840, 620)), (5, (760, 620)), (6, (680, 620)), (7, (600, 620)), (8, (520, 620)), (9, (440, 620)), (10, (360, 620)), (11, (280, 620)), (12, (200, 620)), (14, (120, 620)), (15, (120, 500)), (16, (120, 440)), (17, (120, 380)), (18, (120, 320)), (19, (120, 260)), (20, (120, 200)), (21, (120, 140)), (22, (120, 80)), (23, (1080, 140)), (24, (1000, 140)), (25, (920, 140)), (26, (840, 140)), (27, (760, 140)), (28, (680, 140)), (29, (600, 140)), (30, (520, 140)), (31, (440, 140)), (32, (360, 140)), (33, (280, 140)), (34, (520, 500)), (35, (600, 500)), (36, (680, 500)), (37, (760, 500)), (38, (840, 500)), (39, (920, 500)), (40, (1000, 500)), (41, (1080, 500)), (42, (280, 500)), (43, (280, 440)), (44, (280, 380)), (45, (280, 320)), (46, (280, 260)), (47, (280, 200))]
        # The two lists are paired by index; fail loudly if someone edits one
        # and forgets the other.
        if len(self.room_coordinates) != len(self.room_button_positions):
            raise ValueError(
                f"room_coordinates has {len(self.room_coordinates)} entries but "
                f"room_button_positions has {len(self.room_button_positions)}")

        self.buttons_ = []
        self.generate_buttons()
        self.compute_world_to_screen_transform()

        # Navigation: BasicNavigator is owned by the worker thread from here on.
        self.navigator_ = BasicNavigator()
        self.nav_worker_ = NavWorker(self.navigator_, self.get_logger())
        self.nav_worker_.start()
        self.get_logger().info("HMI Node structural initialization finished.")

    # ------------------------------------------------------------------ setup
    def generate_buttons(self):
        home_pose = self.create_pose_stamped(0.0, 0.0, 0.0)
        home_text = self.font1_.render('Home', True, self.BLACK_)
        home_rect = pygame.Rect(50, 555, 120, 60)
        self.buttons_.append({'text': home_text, 'rect': home_rect, 'pose': home_pose, 'label': 'Home', 'color': self.GREEN_})
        button_width, button_height = 80, 50
        for (room_label, (x_pos, y_pos)), coords in zip(self.room_button_positions, self.room_coordinates):
            text_surface = self.font1_.render(str(room_label), True, self.BLACK_)
            button_rect = pygame.Rect(x_pos, y_pos, button_width, button_height)
            goal_pose = self.create_pose_stamped(*coords)
            self.buttons_.append({'text': text_surface, 'rect': button_rect, 'pose': goal_pose, 'label': f'Room {room_label}', 'color': self.BLUE_})

    def compute_world_to_screen_transform(self):
        """
        Estimate an affine transform (meters in the 'map' frame -> screen pixels)
        via least-squares, using the real coordinate + screen position of every
        button (Home + all rooms) as correspondence points.
        """
        world_pts = np.array([
            [b['pose'].pose.position.x, b['pose'].pose.position.y, 1.0]
            for b in self.buttons_
        ])
        screen_x = np.array([b['rect'].center[0] for b in self.buttons_])
        screen_y = np.array([b['rect'].center[1] for b in self.buttons_])
        self.coeff_x_, *_ = np.linalg.lstsq(world_pts, screen_x, rcond=None)
        self.coeff_y_, *_ = np.linalg.lstsq(world_pts, screen_y, rcond=None)

    def world_to_screen(self, wx, wy):
        ax, bx, cx = self.coeff_x_
        ay, by, cy = self.coeff_y_
        return int(ax * wx + bx * wy + cx), int(ay * wx + by * wy + cy)

    def create_pose_stamped(self, position_x, position_y, rotation_z):
        q_x, q_y, q_z, q_w = tf_transformations.quaternion_from_euler(0.0, 0.0, rotation_z)
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = position_x
        goal_pose.pose.position.y = position_y
        goal_pose.pose.orientation.x, goal_pose.pose.orientation.y, goal_pose.pose.orientation.z, goal_pose.pose.orientation.w = q_x, q_y, q_z, q_w
        return goal_pose

    # ------------------------------------------------------------- robot pose
    def update_robot_pose(self):
        """Runs on the ROS thread (timer callback); just records the latest pose."""
        try:
            tf = self.tf_buffer_.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time(),
                timeout=Duration(seconds=0.05))
        except TransformException:
            return  # no TF yet -> keep the last known point
        with self.pose_lock_:
            self.robot_world_pos_ = (tf.transform.translation.x, tf.transform.translation.y)
            self.robot_pose_time_ = time.monotonic()

    def get_robot_xy(self, max_age=POSE_MAX_AGE):
        """Latest (x, y), or None if unknown / older than max_age (None = any age)."""
        with self.pose_lock_:
            pos, stamp = self.robot_world_pos_, self.robot_pose_time_
        if pos is None:
            return None
        if max_age is not None and time.monotonic() - stamp > max_age:
            return None
        return pos

    def robot_hall(self):
        """Hallway the robot is in: from fresh TF if possible, else the fallback."""
        pos = self.get_robot_xy()
        if pos is not None:
            return hall_of_point(*pos)
        return self.current_hall_

    def distance_to(self, pose):
        pos = self.get_robot_xy()
        if pos is None:
            return None
        return math.hypot(pos[0] - pose.pose.position.x, pos[1] - pose.pose.position.y)

    # ---------------------------------------------------------- state machine
    def set_state(self, new_state):
        if new_state not in ALLOWED_TRANSITIONS[self.state_]:
            self.get_logger().error(
                f"Illegal state transition {self.state_.value} -> {new_state.value} (ignored)")
            return False
        if new_state != self.state_:
            self.get_logger().info(f"State: {self.state_.value} -> {new_state.value}")
        self.state_ = new_state
        return True

    def select_destination(self, label, pose):
        """Clicking a room only STORES the destination; the robot doesn't move.
        Allowed in IDLE, PENDING_CONFIRM (change choice) and FAILED."""
        if self.state_ not in ROOMS_ENABLED_STATES:
            return
        if not self.set_state(State.PENDING_CONFIRM):
            return
        self.pending_label_ = label
        self.pending_pose_ = pose
        self.active_goal_button_ = label
        self.info_text_ = f"Selected: {label}. Press 'Confirm' to close the door and start moving."

    def cancel_selection(self):
        if self.state_ != State.PENDING_CONFIRM:
            return
        if not self.set_state(State.IDLE):
            return
        self.pending_label_ = None
        self.pending_pose_ = None
        self.active_goal_button_ = None
        self.info_text_ = "Selection cleared. Please select a room."

    def confirm_and_go(self):
        if self.state_ != State.PENDING_CONFIRM or self.pending_label_ is None:
            return
        self.close_door()
        self.target_label_, self.target_pose_ = self.pending_label_, self.pending_pose_
        self.pending_label_ = None
        self.pending_pose_ = None
        self.start_navigation()

    def retry(self):
        if self.state_ != State.FAILED or self.target_pose_ is None:
            return
        self.start_navigation()

    def start_navigation(self):
        """Builds the route from the robot's CURRENT hallway and hands it to the worker."""
        if self.nav_worker_.is_busy():
            self.get_logger().warn("Navigation worker still busy, request ignored.")
            return
        if not self.set_state(State.NAVIGATING):
            return
        dest = self.target_pose_
        dest_xy = (dest.pose.position.x, dest.pose.position.y)
        start_hall = self.robot_hall()
        waypoints = [self.create_pose_stamped(x, y, yaw)
                     for x, y, yaw in plan_junction_points(start_hall, dest_xy)]

        # Re-stamp and broadcast so error-monitoring nodes know exactly which
        # pose this run is being judged against.
        dest.header.stamp = self.get_clock().now().to_msg()
        self.goal_request_pub_.publish(dest)
        waypoints.append(dest)

        self.get_logger().info(
            f"Nav request: {self.target_label_} (hall {start_hall} -> "
            f"{hall_of_point(*dest_xy)}, {len(waypoints)} waypoint(s))")
        self.active_goal_button_ = self.target_label_
        self.info_text_ = f"Door closed. Heading to {self.target_label_}..."
        self.nav_worker_.submit(waypoints)

    def request_stop(self):
        if self.state_ != State.NAVIGATING:
            return
        if not self.set_state(State.CANCELING):
            return
        self.nav_worker_.request_cancel()
        self.info_text_ = "Stopping..."

    def on_navigation_done(self, outcome):
        if self.state_ not in (State.NAVIGATING, State.CANCELING):
            self.get_logger().warn(f"Unexpected navigation result in state {self.state_.value}")
            return
        result = outcome['result']
        reason = outcome.get('reason')

        # Don't trust SUCCEEDED blindly: check where the robot really is.
        if result == TaskResult.SUCCEEDED:
            dist = self.distance_to(self.target_pose_)
            if dist is not None and dist > ARRIVAL_CHECK_TOLERANCE:
                result = TaskResult.FAILED
                reason = f"robot is {dist:.2f} m from the goal"

        if result == TaskResult.SUCCEEDED:
            self.set_state(State.ARRIVED)
            self.current_hall_ = hall_of_point(self.target_pose_.pose.position.x,
                                               self.target_pose_.pose.position.y)
            self.info_text_ = f"Arrived at {self.target_label_}. Press 'Open' to open the door."
            return

        if reason is None:
            if outcome.get('canceled') or result == TaskResult.CANCELED:
                reason = "stopped by operator"
            else:
                reason = "Nav2 reported failure"
        self.current_hall_ = None  # position unknown until TF says otherwise
        self.set_state(State.FAILED)
        self.info_text_ = (f"Did not reach {self.target_label_} ({reason}). Door stays closed. "
                           f"Press 'Retry' or select another destination.")

    def open_compartment(self):
        """Only possible after a verified arrival; then back to IDLE."""
        if self.state_ != State.ARRIVED:
            return
        self.open_door()
        self.set_state(State.IDLE)
        self.target_label_ = None
        self.target_pose_ = None
        self.active_goal_button_ = None
        self.info_text_ = "Door opened. Please select a room."

    # Door actuator hooks (status only for now, no real actuator yet).
    def close_door(self):
        self.get_logger().info("Door: CLOSE")

    def open_door(self):
        self.get_logger().info("Door: OPEN")

    def process_worker_events(self):
        while True:
            event = self.nav_worker_.poll_event()
            if event is None:
                return
            kind, payload = event
            if kind == 'ready':
                if self.set_state(State.IDLE):
                    self.info_text_ = "Please select a room."
            elif kind == 'error':
                self.get_logger().error("Nav2 startup failed:\n" + payload)
                self.info_text_ = "Nav2 startup failed. Check the log and restart the HMI."
            elif kind == 'done':
                self.on_navigation_done(payload)

    def action_buttons(self):
        """(label, color, rect, callback) for each action button visible in the
        current state. Used for both drawing and click handling."""
        p, s = self.primary_button_rect_, self.secondary_button_rect_
        if self.state_ == State.PENDING_CONFIRM:
            return [('Confirm', self.GREEN_, p, self.confirm_and_go),
                    ('Cancel', self.GRAY_, s, self.cancel_selection)]
        if self.state_ == State.NAVIGATING:
            return [('Stop', self.RED_, p, self.request_stop)]
        if self.state_ == State.ARRIVED:
            return [('Open', self.ORANGE_, p, self.open_compartment)]
        if self.state_ == State.FAILED:
            return [('Retry', self.GREEN_, p, self.retry)]
        return []

    # --------------------------------------------------------------------- UI
    def draw_wrapped_text(self, text, rect, font, color):
        """Simple word-wrap: fits text inside rect.width, drawing top-aligned lines."""
        words = text.split(' ')
        lines = []
        current = ''
        for w in words:
            trial = (current + ' ' + w).strip()
            if font.size(trial)[0] <= rect.width - 16:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = w
        if current:
            lines.append(current)
        line_height = font.get_linesize()
        y = rect.y + 8
        for line in lines:
            surf = font.render(line, True, color)
            self.screen_.blit(surf, (rect.x + 8, y))
            y += line_height

    def update_ui(self):
        """Runs on the main thread. Never blocks on Nav2."""
        self.process_worker_events()

        self.screen_.fill(self.WHITE_)

        # Hallways
        pygame.draw.line(self.screen_, self.GRAY_, (180, 585), (1160, 585), 50)
        pygame.draw.line(self.screen_, self.GRAY_, (240, 80), (240, 560), 50)
        pygame.draw.line(self.screen_, self.GRAY_, (215, 105), (1160, 105), 50)

        # Room buttons
        rooms_enabled = self.state_ in ROOMS_ENABLED_STATES
        for button in self.buttons_:
            color = button['color'] if rooms_enabled else self.GRAY_
            pygame.draw.rect(self.screen_, color, button['rect'])
            text_rect = button['text'].get_rect(center=button['rect'].center)
            self.screen_.blit(button['text'], text_rect)
            if button['label'] == self.active_goal_button_:
                marker_pos = (button['rect'].right - 8, button['rect'].top + 8)
                pygame.draw.circle(self.screen_, (255, 0, 0), marker_pos, 6)

        # Robot position: red when fresh, gray when the last TF is stale
        last_pos = self.get_robot_xy(max_age=None)
        if last_pos is not None:
            fresh = self.get_robot_xy() is not None
            robot_screen_pos = self.world_to_screen(*last_pos)
            pygame.draw.circle(self.screen_, (255, 0, 0) if fresh else self.DARK_GRAY_, robot_screen_pos, 10)
            pygame.draw.circle(self.screen_, self.BLACK_, robot_screen_pos, 10, width=1)

        # Info box
        pygame.draw.rect(self.screen_, self.WHITE_, self.info_box_rect_)
        pygame.draw.rect(self.screen_, self.BLACK_, self.info_box_rect_, width=2)
        self.draw_wrapped_text(self.info_text_, self.info_box_rect_, self.font_small_, self.BLACK_)

        # Action buttons
        actions = self.action_buttons()
        for label, color, rect, _ in actions:
            pygame.draw.rect(self.screen_, color, rect)
            pygame.draw.rect(self.screen_, self.BLACK_, rect, width=2)
            surf = self.font1_.render(label, True, self.BLACK_)
            self.screen_.blit(surf, surf.get_rect(center=rect.center))

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                handled = False
                for _, _, rect, callback in actions:
                    if rect.collidepoint(event.pos):
                        callback()
                        handled = True
                        break
                if not handled and rooms_enabled:
                    for button in self.buttons_:
                        if button['rect'].collidepoint(event.pos):
                            self.select_destination(button['label'], button['pose'])
                            break

        pygame.display.flip()
        return True

    def shutdown(self):
        """Cancel any running task and stop the worker before the node goes away."""
        self.nav_worker_.request_shutdown()
        self.nav_worker_.join(timeout=3.0)
        if self.nav_worker_.is_alive():
            self.get_logger().warn("Navigation worker did not stop in time.")
        else:
            self.navigator_.destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = HmiNode()

        # ROS executor (TF listener + pose timer) on a background thread
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        threading.Thread(target=executor.spin, daemon=True).start()

        # Pygame on the main thread
        clock = pygame.time.Clock()
        running = True
        while running and rclpy.ok():
            running = node.update_ui()
            clock.tick(30)
    except KeyboardInterrupt:
        pass
    except Exception:
        msg = "HMI crashed:\n" + traceback.format_exc()
        if node is not None:
            node.get_logger().error(msg)
        else:
            print(msg, file=sys.stderr)
    finally:
        if node is not None:
            node.shutdown()   # cancels navigation if the robot is still moving
        pygame.quit()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit()


if __name__ == "__main__":
    main()
