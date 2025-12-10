#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hybrid Overtaking Node for Head-to-Head Autonomous Racing

This node combines two control strategies:
1. Pure Pursuit: Follows a pre-computed racing line for optimal lap times
2. Follow-the-Gap: Reactive obstacle avoidance for overtaking opponents

Mode Switching Logic:
- Default: Pure Pursuit mode (following racing line)
- Switch to Follow-the-Gap when opponent detected ahead within OVERTAKE_DISTANCE
- Return to Pure Pursuit when path is clear or opponent is passed
"""

import rospy
import os
import sys
import csv
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import tf

# =========================
# Configuration parameters
# =========================

# Car identification
CAR_NAME = str(sys.argv[1]) if len(sys.argv) > 1 else 'car_2'
TRAJECTORY_NAME = str(sys.argv[2]) if len(sys.argv) > 2 else 'raceline'

# Control mode parameters
OVERTAKE_DISTANCE = 1.0         # m - distance ahead to check for opponents
MIN_OVERTAKE_TIME = 2.0         # seconds - minimum time to stay in overtake mode
OPPONENT_WIDTH_THRESHOLD = 0.5  # m - lateral width to consider as blocking the path
CLEAR_PATH_DISTANCE = 4.0       # m - distance needed to consider path clear after overtake
MIN_OPPONENT_DISTANCE = 0.4     # m - ignore obstacles closer than this (car body, mounting)

# Pure Pursuit parameters
WHEELBASE_LEN = 0.325           # m
L_MIN = 0.8                     # m - minimum lookahead
L_MAX = 2.0                     # m - maximum lookahead
V_MIN_LD = 8.0                  # speed for min lookahead
V_MAX_LD = 15.0                 # speed for max lookahead
MAX_SPEED = 65.0                # maximum speed
MIN_SPEED = 5.0                 # minimum speed
VELOCITY_SCALE_FACTOR = 20.0    # converts m/s to 0-100 range
MAX_STEERING_ANGLE_RAD = 0.4    # radians (~23 degrees)

# Follow-the-Gap parameters
LIDAR_ZERO_DEG_IN_CAR_FRAME = 90  # degrees
STEERING_SIGN = +1.0
DISPARITY_THRESHOLD = 0.20      # m
CAR_WIDTH = 0.31                # m
CAR_LENGTH = 0.50               # m
SAFETY_MARGIN = 0.1             # m
MIN_RANGE_OBSTACLE = 0.18       # m
FTG_MAX_VELOCITY = 30.0         # max speed in gap mode
FTG_MIN_VELOCITY = 10.0         # min speed in gap mode
MIN_FREE_DISTANCE = 2.0         # m
STEERING_GAIN = 2.2
ANGLE_SMOOTH_ALPHA = 0.5
VELOCITY_SMOOTH_ALPHA = 0.5
LIDAR_SMOOTH_WINDOW = 7
SIDE_SAFETY_DISTANCE = 0.2      # m

# Steering range
STEERING_RANGE = 100.0
MAX_STEERING = 100.0
MIN_STEERING = -100.0

# Frame
FRAME_ID = 'map'

# =========================
# Utility functions
# =========================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def bubble_half_angle(distance):
    """Numerically stable half-angle (radians) that covers (car half-width + margin)."""
    radius = (CAR_WIDTH * 0.5) + SAFETY_MARGIN
    if distance <= 1e-3:
        return math.radians(45.0)
    return math.atan(radius / max(distance, 1e-3))

def car_to_lidar_angle(a_car_rad):
    """Convert an angle in the car frame (0 = straight ahead, left positive) to LiDAR frame."""
    return a_car_rad + math.radians(LIDAR_ZERO_DEG_IN_CAR_FRAME)

def lidar_to_car_angle(a_lidar_rad):
    """Convert an angle in the LiDAR frame to the car frame (0 = straight ahead, left positive)."""
    return a_lidar_rad - math.radians(LIDAR_ZERO_DEG_IN_CAR_FRAME)


class HybridOvertakeNode(object):
    def __init__(self):
        # Publishers
        self.command_pub = rospy.Publisher('/{}/offboard/command'.format(CAR_NAME), 
                                          AckermannDrive, queue_size=1)
        self.mode_marker_pub = rospy.Publisher('/{}/hybrid/mode_marker'.format(CAR_NAME), 
                                               Marker, queue_size=1)
        self.opponent_marker_pub = rospy.Publisher('/{}/hybrid/opponent_marker'.format(CAR_NAME), 
                                                   MarkerArray, queue_size=1)
        self.target_marker_pub = rospy.Publisher('/{}/hybrid/target_marker'.format(CAR_NAME), 
                                                 Marker, queue_size=1)
        self.path_marker_pub = rospy.Publisher('/{}/hybrid/path_marker'.format(CAR_NAME), 
                                               Marker, queue_size=1)
        
        # State variables
        self.mode = 'pure_pursuit'  # 'pure_pursuit' or 'follow_gap'
        self.mode_switch_time = rospy.Time.now()
        self.prev_steering = 0.0
        self.prev_velocity = 0.0
        self.current_speed_cmd = 0.0
        
        # Path data
        self.plan = []
        self.path_resolution = []
        self.construct_path()
        
        # Pose data
        self.current_pose = None
        self.current_heading = 0.0
        
        # LIDAR data
        self.latest_scan = None
        
        # Subscribers
        rospy.Subscriber('/{}/particle_filter/viz/inferred_pose'.format(CAR_NAME), 
                        PoseStamped, self.pose_callback)
        rospy.Subscriber('/{}/scan'.format(CAR_NAME), 
                        LaserScan, self.lidar_callback)
        
        rospy.loginfo("Hybrid Overtake Node initialized for {}".format(CAR_NAME))
        
        # Publish path marker once
        rospy.sleep(0.5)
        self.publish_path_marker()
    
    # =========================
    # Path construction
    # =========================
    
    def construct_path(self):
        """Load racing line from CSV file."""
        file_path = os.path.expanduser('/home/nvidia/{}.csv'.format(TRAJECTORY_NAME))
        
        try:
            with open(file_path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for waypoint in csv_reader:
                    if len(waypoint) >= 2:
                        self.plan.append(waypoint)
            
            # Convert to floats
            for index in range(len(self.plan)):
                for point in range(len(self.plan[index])):
                    self.plan[index][point] = float(self.plan[index][point])
                # Add default velocity if not provided
                if len(self.plan[index]) < 3:
                    self.plan[index].append(0.0)
            
            # Calculate path resolution
            for index in range(1, len(self.plan)):
                dx = self.plan[index][0] - self.plan[index-1][0]
                dy = self.plan[index][1] - self.plan[index-1][1]
                self.path_resolution.append(math.sqrt(dx*dx + dy*dy))
            
            rospy.loginfo("Loaded racing line with {} waypoints".format(len(self.plan)))
            
        except Exception as e:
            rospy.logerr("Failed to load racing line: {}".format(e))
            self.plan = []
    
    # =========================
    # Callbacks
    # =========================
    
    def pose_callback(self, data):
        """Store current pose for use in control loop."""
        self.current_pose = data
        self.current_heading = tf.transformations.euler_from_quaternion((
            data.pose.orientation.x,
            data.pose.orientation.y,
            data.pose.orientation.z,
            data.pose.orientation.w))[2]
    
    def lidar_callback(self, scan):
        """Store LIDAR data and run main control loop."""
        self.latest_scan = scan
        
        # Only run control if we have both pose and scan
        if self.current_pose is not None:
            self.control_loop()
    
    # =========================
    # Main control loop
    # =========================
    
    def control_loop(self):
        """Main control loop: decide mode and compute control."""
        
        # Detect opponent ahead
        opponent_detected, opponent_distance = self.detect_opponent()
        
        # Decide control mode
        self.update_mode(opponent_detected, opponent_distance)
        
        # Execute appropriate control
        if self.mode == 'pure_pursuit':
            steering, velocity = self.pure_pursuit_control()
        else:  # follow_gap
            steering, velocity = self.follow_gap_control()
        
        # Publish command
        command = AckermannDrive()
        command.steering_angle = steering
        command.speed = velocity
        self.command_pub.publish(command)
        
        # Update state
        self.current_speed_cmd = velocity
        
        # Publish visualizations
        self.publish_mode_marker()
        if opponent_detected:
            self.publish_opponent_marker(opponent_distance)
    
    # =========================
    # Opponent detection
    # =========================
    
    def detect_opponent(self):
        """
        Detect if an opponent is blocking the path ahead.
        
        Returns:
            (detected, distance): Boolean and distance to nearest opponent
        """
        if self.latest_scan is None or self.current_pose is None:
            return False, float('inf')
        
        scan = self.latest_scan
        ranges = np.array(scan.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)
        
        # Check forward-facing region for obstacles
        # Car frame: -30° to +30° (about 60° cone ahead)
        car_min = math.radians(-30.0)
        car_max = math.radians(+30.0)
        
        # Convert to LIDAR frame
        lid_min = car_to_lidar_angle(car_min)
        lid_max = car_to_lidar_angle(car_max)
        
        # Get indices
        n = len(ranges)
        i0 = int((lid_min - scan.angle_min) / scan.angle_increment)
        i1 = int((lid_max - scan.angle_min) / scan.angle_increment)
        i0 = clamp(i0, 0, n - 1)
        i1 = clamp(i1, 0, n - 1)
        
        if i0 > i1:
            i0, i1 = i1, i0
        
        # Find minimum distance in forward cone (excluding too-close readings)
        min_distance = float('inf')
        for i in range(i0, i1 + 1):
            # Ignore readings that are too close (likely the car itself)
            if ranges[i] > MIN_OPPONENT_DISTANCE and ranges[i] < min_distance:
                min_distance = ranges[i]
        
        # Opponent detected if something is within overtake distance
        detected = min_distance < OVERTAKE_DISTANCE
        
        return detected, min_distance
    
    # =========================
    # Mode switching
    # =========================
    
    def update_mode(self, opponent_detected, opponent_distance):
        """
        Update control mode based on opponent detection.
        
        Logic:
        - Switch to follow_gap when opponent detected within OVERTAKE_DISTANCE
        - Stay in follow_gap for at least MIN_OVERTAKE_TIME
        - Return to pure_pursuit when path is clear beyond CLEAR_PATH_DISTANCE
        """
        current_time = rospy.Time.now()
        time_in_mode = (current_time - self.mode_switch_time).to_sec()
        
        if self.mode == 'pure_pursuit':
            # Switch to follow_gap if opponent detected
            if opponent_detected:
                self.mode = 'follow_gap'
                self.mode_switch_time = current_time
                rospy.loginfo("SWITCHING TO FOLLOW-THE-GAP MODE (opponent at {:.2f}m)".format(opponent_distance))
        
        elif self.mode == 'follow_gap':
            # Only consider switching back after minimum time
            if time_in_mode > MIN_OVERTAKE_TIME:
                # Switch back if path is clear
                if not opponent_detected or opponent_distance > CLEAR_PATH_DISTANCE:
                    self.mode = 'pure_pursuit'
                    self.mode_switch_time = current_time
                    rospy.loginfo("SWITCHING TO PURE PURSUIT MODE (path clear)")
    
    # =========================
    # Pure Pursuit Controller
    # =========================
    
    def pure_pursuit_control(self):
        """Execute pure pursuit control."""
        
        if not self.plan or self.current_pose is None:
            return 0.0, 0.0
        
        odom_x = self.current_pose.pose.position.x
        odom_y = self.current_pose.pose.position.y
        
        # Find base projection (closest point on path)
        min_distance = float('inf')
        base_index = 0
        
        for i in range(len(self.plan)):
            dx = self.plan[i][0] - odom_x
            dy = self.plan[i][1] - odom_y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < min_distance:
                min_distance = distance
                base_index = i
        
        # Compute adaptive lookahead distance
        lookahead_distance = self.compute_lookahead_distance()
        
        # Find target point (lookahead distance ahead on path)
        cumulative_distance = 0.0
        target_index = base_index
        num_points = len(self.plan)
        
        for j in range(num_points):
            i = (base_index + j) % num_points
            i_next = (i + 1) % num_points
            
            dx = self.plan[i_next][0] - self.plan[i][0]
            dy = self.plan[i_next][1] - self.plan[i][1]
            segment_distance = math.sqrt(dx*dx + dy*dy)
            
            if cumulative_distance + segment_distance >= lookahead_distance:
                remaining_distance = lookahead_distance - cumulative_distance
                ratio = remaining_distance / segment_distance if segment_distance > 0 else 0
                target_x = self.plan[i][0] + ratio * dx
                target_y = self.plan[i][1] + ratio * dy
                target_index = i
                break
            
            cumulative_distance += segment_distance
            target_index = i_next
        else:
            target_x = self.plan[target_index][0]
            target_y = self.plan[target_index][1]
        
        # Pure pursuit steering calculation
        dx = target_x - odom_x
        dy = target_y - odom_y
        
        # Transform to vehicle frame
        target_x_vehicle = math.cos(-self.current_heading) * dx - math.sin(-self.current_heading) * dy
        target_y_vehicle = math.sin(-self.current_heading) * dx + math.cos(-self.current_heading) * dy
        
        # Calculate curvature
        curvature = 2.0 * target_y_vehicle / (lookahead_distance ** 2)
        steering_angle = math.atan(WHEELBASE_LEN * curvature)
        
        # Convert to command range
        normalized_steering = steering_angle / MAX_STEERING_ANGLE_RAD
        steering_cmd = clamp(normalized_steering * STEERING_RANGE, MIN_STEERING, MAX_STEERING)
        
        # Velocity from waypoint or dynamic scaling
        waypoint_velocity = self.plan[base_index][2] if len(self.plan[base_index]) > 2 else 0.0
        
        if waypoint_velocity > 0.0:
            velocity_cmd = min(100.0, waypoint_velocity * VELOCITY_SCALE_FACTOR)
        else:
            abs_steering = abs(steering_cmd)
            speed_scale = 1.0 - (abs_steering / STEERING_RANGE)
            velocity_cmd = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * speed_scale
        
        # Publish visualization
        self.publish_target_marker(target_x, target_y, "pure_pursuit")
        
        return steering_cmd, velocity_cmd
    
    def compute_lookahead_distance(self):
        """Adaptive lookahead based on current speed."""
        v = self.current_speed_cmd
        
        if v <= V_MIN_LD:
            return L_MIN
        if v >= V_MAX_LD:
            return L_MAX
        
        ratio = (v - V_MIN_LD) / float(V_MAX_LD - V_MIN_LD)
        return L_MIN + ratio * (L_MAX - L_MIN)
    
    # =========================
    # Follow-the-Gap Controller
    # =========================
    
    def follow_gap_control(self):
        """Execute follow-the-gap control."""
        
        if self.latest_scan is None:
            return 0.0, 0.0
        
        scan = self.latest_scan
        
        # Preprocess LIDAR
        ranges = self.preprocess_lidar(scan)
        
        # Find and extend disparities
        disparities = self.find_disparities(ranges)
        processed = self.extend_disparities(ranges, scan.angle_increment, disparities)
        
        # Get front window indices
        i0, i1 = self.front_window_indices(scan.angle_min, scan.angle_increment, len(processed))
        
        # Find target direction (farthest point in widest gap)
        target_idx, target_angle_lidar, max_dist = self.find_target_direction_farthest_distance_in_widest_gap(
            processed, scan.angle_min, scan.angle_increment, i0, i1
        )
        
        # Calculate steering
        steering_cmd = self.calculate_steering(target_angle_lidar)
        
        # Check side clearance
        if not self.check_side_clearance(ranges, scan.angle_min, scan.angle_increment, steering_cmd):
            steering_cmd = 0.0
        
        # Calculate velocity
        velocity_cmd = self.calculate_velocity(steering_cmd, max_dist)
        
        # Publish visualization
        target_x = max_dist * math.cos(target_angle_lidar)
        target_y = max_dist * math.sin(target_angle_lidar)
        self.publish_target_marker(target_x, target_y, "follow_gap")
        
        return steering_cmd, velocity_cmd
    
    # Follow-the-Gap helper functions
    
    def preprocess_lidar(self, scan):
        """Preprocess LIDAR data."""
        ranges = np.array(scan.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)
        ranges = np.where(ranges < MIN_RANGE_OBSTACLE, MIN_RANGE_OBSTACLE, ranges)
        
        if LIDAR_SMOOTH_WINDOW > 1:
            ranges = self.smooth_lidar(ranges, LIDAR_SMOOTH_WINDOW)
        
        return ranges
    
    def smooth_lidar(self, ranges, window_size):
        """Apply moving average filter."""
        if window_size <= 1:
            return ranges
        kernel = np.ones(window_size) / window_size
        return np.convolve(ranges, kernel, mode='same')
    
    def find_disparities(self, ranges):
        """Find disparities in LIDAR data."""
        diffs = np.abs(np.diff(ranges))
        return list(np.where(diffs > DISPARITY_THRESHOLD)[0])
    
    def extend_disparities(self, ranges, angle_increment, disparities):
        """Extend disparities using bubble method."""
        out = np.copy(ranges)
        
        for idx in disparities:
            d1, d2 = out[idx], out[idx + 1]
            
            if d1 <= d2:
                closer_dist = d1
                closer_idx = idx
                farther_idx = idx + 1
                direction = +1
            else:
                closer_dist = d2
                closer_idx = idx + 1
                farther_idx = idx
                direction = -1
            
            radius = (CAR_WIDTH * 0.5) + SAFETY_MARGIN
            half_angle = bubble_half_angle(closer_dist)
            num_samples = int(half_angle / angle_increment)
            num_samples = clamp(num_samples, 0, 60)
            
            for k in range(num_samples + 1):
                j = farther_idx + direction * k
                if j < 0 or j >= len(out):
                    break
                if out[j] > closer_dist:
                    out[j] = closer_dist
        
        return out
    
    def front_window_indices(self, angle_min, angle_increment, n):
        """Get indices for front hemisphere."""
        car_min = math.radians(-90.0)
        car_max = math.radians(+90.0)
        lid_min = car_to_lidar_angle(car_min)
        lid_max = car_to_lidar_angle(car_max)
        
        i0 = int((lid_min - angle_min) / angle_increment)
        i1 = int((lid_max - angle_min) / angle_increment)
        i0 = clamp(i0, 0, n - 1)
        i1 = clamp(i1, 0, n - 1)
        return (i0, i1) if i0 <= i1 else (i1, i0)
    
    def find_target_direction_farthest_distance_in_widest_gap(self, ranges, angle_min, angle_increment, start_idx, end_idx):
        """Find target direction in widest gap."""
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        
        gaps = []
        window = ranges[start_idx:end_idx + 1]
        local_start = None
        
        for i, sample in enumerate(window):
            if sample >= MIN_FREE_DISTANCE:
                if local_start is None:
                    local_start = i
            else:
                if local_start is not None:
                    local_end = i
                    width = local_end - local_start
                    average_local_distance = np.mean(window[local_start:local_end])
                    gaps.append((width * average_local_distance, local_start, local_end))
                    local_start = None
        
        if local_start is not None:
            local_end = len(window)
            width = local_end - local_start
            average_local_distance = np.mean(window[local_start:local_end])
            gaps.append((width * average_local_distance, local_start, local_end))
        
        if gaps:
            width, local_start, local_end = max(gaps, key=lambda x: x[0])
            midpoint_local = int(local_start + (local_end - local_start) // 2)
            target_idx = start_idx + midpoint_local
            target_lidar_angle = angle_min + target_idx * angle_increment
            target_dist = ranges[target_idx]
            return (target_idx, target_lidar_angle, target_dist)
        else:
            mid_idx = (start_idx + end_idx) // 2
            return (mid_idx, angle_min + mid_idx * angle_increment, ranges[mid_idx])
    
    def calculate_steering(self, target_angle_lidar):
        """Calculate steering from target angle."""
        a_car = lidar_to_car_angle(target_angle_lidar)
        dev_deg = math.degrees(a_car)
        steering_cmd = (dev_deg / 45.0) * 100.0 * STEERING_GAIN
        steering_cmd = clamp(steering_cmd, MIN_STEERING, MAX_STEERING)
        steering_cmd *= STEERING_SIGN
        
        # Smoothing
        steering_cmd = ANGLE_SMOOTH_ALPHA * self.prev_steering + (1.0 - ANGLE_SMOOTH_ALPHA) * steering_cmd
        self.prev_steering = steering_cmd
        
        return steering_cmd
    
    def calculate_velocity(self, steering_cmd, max_distance_ahead):
        """Calculate velocity based on steering and distance."""
        steering_mag = abs(steering_cmd) / MAX_STEERING
        v = FTG_MIN_VELOCITY + (FTG_MAX_VELOCITY - FTG_MIN_VELOCITY) * (1.0 - steering_mag)
        
        if max_distance_ahead < 1.0:
            v *= 0.2
        elif max_distance_ahead < 2.0:
            v *= 0.75
        
        # Smoothing
        v = VELOCITY_SMOOTH_ALPHA * self.prev_velocity + (1.0 - VELOCITY_SMOOTH_ALPHA) * v
        self.prev_velocity = v
        
        return clamp(v, 0.0, 100.0)
    
    def check_side_clearance(self, ranges, angle_min, angle_increment, steering_cmd):
        """Check for obstacles on sides when cornering."""
        if abs(steering_cmd) < 5.0:
            return True
        
        n = len(ranges)
        turning_left = steering_cmd > 0
        
        if turning_left:
            car_min = math.radians(90.0)
            car_max = math.radians(180.0)
        else:
            car_min = math.radians(-180.0)
            car_max = math.radians(-90.0)
        
        lid_min = car_to_lidar_angle(car_min)
        lid_max = car_to_lidar_angle(car_max)
        
        i0 = int((lid_min - angle_min) / angle_increment)
        i1 = int((lid_max - angle_min) / angle_increment)
        i0 = clamp(i0, 0, n - 1)
        i1 = clamp(i1, 0, n - 1)
        
        if i0 > i1:
            i0, i1 = i1, i0
        
        infractions = 0
        for i in range(i0, i1 + 1):
            if ranges[i] < SIDE_SAFETY_DISTANCE:
                infractions += 1
                if infractions >= 10:
                    return False
        
        return True
    
    # =========================
    # Visualization
    # =========================
    
    def publish_path_marker(self):
        """Publish reference path marker."""
        if not self.plan:
            return
        
        path_marker = Marker()
        path_marker.header.frame_id = FRAME_ID
        path_marker.header.stamp = rospy.Time.now()
        path_marker.ns = "reference_path"
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.scale.x = 0.05
        path_marker.color.r = 0.0
        path_marker.color.g = 1.0
        path_marker.color.b = 0.0
        path_marker.color.a = 0.5
        
        for waypoint in self.plan:
            p = Point()
            p.x = waypoint[0]
            p.y = waypoint[1]
            p.z = 0.0
            path_marker.points.append(p)
        
        if len(self.plan) > 0:
            p = Point()
            p.x = self.plan[0][0]
            p.y = self.plan[0][1]
            p.z = 0.0
            path_marker.points.append(p)
        
        path_marker.lifetime = rospy.Duration(0)
        self.path_marker_pub.publish(path_marker)
    
    def publish_mode_marker(self):
        """Publish text marker showing current mode."""
        if self.current_pose is None:
            return
        
        mode_marker = Marker()
        mode_marker.header.frame_id = FRAME_ID
        mode_marker.header.stamp = rospy.Time.now()
        mode_marker.ns = "control_mode"
        mode_marker.id = 0
        mode_marker.type = Marker.TEXT_VIEW_FACING
        mode_marker.action = Marker.ADD
        
        mode_marker.pose.position.x = self.current_pose.pose.position.x
        mode_marker.pose.position.y = self.current_pose.pose.position.y
        mode_marker.pose.position.z = 1.0
        mode_marker.pose.orientation.w = 1.0
        
        mode_marker.scale.z = 0.3
        
        if self.mode == 'pure_pursuit':
            mode_marker.text = "MODE: PURE PURSUIT"
            mode_marker.color.r = 0.0
            mode_marker.color.g = 1.0
            mode_marker.color.b = 0.0
        else:
            mode_marker.text = "MODE: OVERTAKING"
            mode_marker.color.r = 1.0
            mode_marker.color.g = 0.5
            mode_marker.color.b = 0.0
        
        mode_marker.color.a = 1.0
        mode_marker.lifetime = rospy.Duration(0.2)
        
        self.mode_marker_pub.publish(mode_marker)
    
    def publish_opponent_marker(self, distance):
        """Publish marker showing detected opponent."""
        if self.current_pose is None:
            return
        
        marker_array = MarkerArray()
        
        # Sphere at opponent location (approximate)
        opponent_marker = Marker()
        opponent_marker.header.frame_id = FRAME_ID
        opponent_marker.header.stamp = rospy.Time.now()
        opponent_marker.ns = "opponent"
        opponent_marker.id = 0
        opponent_marker.type = Marker.SPHERE
        opponent_marker.action = Marker.ADD
        
        # Place marker ahead of car at detected distance
        opponent_marker.pose.position.x = self.current_pose.pose.position.x + distance * math.cos(self.current_heading)
        opponent_marker.pose.position.y = self.current_pose.pose.position.y + distance * math.sin(self.current_heading)
        opponent_marker.pose.position.z = 0.2
        opponent_marker.pose.orientation.w = 1.0
        
        opponent_marker.scale.x = 0.4
        opponent_marker.scale.y = 0.4
        opponent_marker.scale.z = 0.4
        
        opponent_marker.color.r = 1.0
        opponent_marker.color.g = 0.0
        opponent_marker.color.b = 0.0
        opponent_marker.color.a = 0.7
        
        opponent_marker.lifetime = rospy.Duration(0.2)
        
        marker_array.markers.append(opponent_marker)
        self.opponent_marker_pub.publish(marker_array)
    
    def publish_target_marker(self, target_x, target_y, mode):
        """Publish marker for target point."""
        target_marker = Marker()
        target_marker.header.frame_id = FRAME_ID if mode == "pure_pursuit" else "{}_laser".format(CAR_NAME)
        target_marker.header.stamp = rospy.Time.now()
        target_marker.ns = "target_point_{}".format(mode)
        target_marker.id = 0
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        
        target_marker.pose.position.x = target_x
        target_marker.pose.position.y = target_y
        target_marker.pose.position.z = 0.1
        target_marker.pose.orientation.w = 1.0
        
        target_marker.scale.x = 0.2
        target_marker.scale.y = 0.2
        target_marker.scale.z = 0.2
        
        if mode == "pure_pursuit":
            target_marker.color.r = 0.0
            target_marker.color.g = 0.0
            target_marker.color.b = 1.0
        else:
            target_marker.color.r = 1.0
            target_marker.color.g = 1.0
            target_marker.color.b = 0.0
        
        target_marker.color.a = 1.0
        target_marker.lifetime = rospy.Duration(0.1)
        
        self.target_marker_pub.publish(target_marker)


def main():
    rospy.init_node('hybrid_overtake', anonymous=True)
    rospy.loginfo("Hybrid Overtaking Node started")
    
    node = HybridOvertakeNode()
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Hybrid Overtaking Node terminated")
