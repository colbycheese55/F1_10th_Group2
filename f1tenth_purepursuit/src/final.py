#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
import numpy as np
import os
import sys
import csv

from ackermann_msgs.msg import AckermannDrive
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PolygonStamped
from geometry_msgs.msg import Point32
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import tf

# ============================================================
# GLOBAL CONFIGURATION AND STATE
# ============================================================

# LiDAR preprocessing parameters
LIDAR_SMOOTH_WINDOW = 7
MIN_RANGE_OBSTACLE = 0.5
OBSTACLE_DIST_FRONT = 1.5

# Path and trajectory
plan = []
path_resolution = []
frame_id = 'map'
trajectory_name = str(sys.argv[1])

# Vehicle physical parameters
WHEELBASE_LEN = 0.325
STEERING_RANGE = 100.0

# Adaptive lookahead parameters
L_MIN = 0.8
L_MAX = 2.0
V_MIN_LD = 8.0
V_MAX_LD = 25.0

# State variables
current_speed_cmd = 0.0
opponent_detected = False
opponent_detection_count = 0  # Counter for consecutive detections
wp_seq = 0
control_polygon = PolygonStamped()

# Progressive slowdown parameters
MAX_DETECTION_COUNT = 10  # Number of detections before full stop
MIN_SPEED_FACTOR = 0.1    # Minimum speed factor (10% of normal speed)

# Publishers
command_pub = None
polygon_pub = None
path_marker_pub = None
pose_marker_pub = None
target_marker_pub = None
steering_marker_pub = None
lookahead_marker_pub = None


# ============================================================
# LIDAR PROCESSING AND OBSTACLE DETECTION
# ============================================================

def smooth_lidar(ranges, window_size):
    """Apply moving average smoothing to LiDAR data."""
    if window_size <= 1:
        return ranges
    
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(ranges, kernel, mode='same')
    return smoothed


def preprocess_lidar(scan):
    """Clean and smooth LiDAR scan data."""
    ranges = np.array(scan.ranges, dtype=np.float32)
    # Replace NaN/Inf with large value (no obstacle)
    ranges = np.where(np.isfinite(ranges), ranges, 10.0)
    # Clip values too close
    ranges = np.where(ranges < MIN_RANGE_OBSTACLE, 10, ranges)
    
    # Apply smoothing filter
    if LIDAR_SMOOTH_WINDOW > 1:
        ranges = smooth_lidar(ranges, LIDAR_SMOOTH_WINDOW)
    
    return ranges


def check_for_obstacles(lidar_data, position, distance_threshold, angle_min, angle_increment):
    """
    Detect car-sized obstacles (not walls) by checking for consecutive detections.
    A car is typically 0.3-0.5m wide, identified by angular width between 5-15 degrees.
    """
    angle_min = math.radians(-30)
    
    # Define angular range based on position
    if position == "front":
        i0 = int((math.radians(65) - angle_min) / angle_increment)
        i1 = int((math.radians(115) - angle_min) / angle_increment)
    elif position == "right":
        i0 = int(-angle_min / angle_increment)
        i1 = int((math.radians(90) - angle_min) / angle_increment)
    elif position == "left":
        i0 = int((math.radians(90) - angle_min) / angle_increment)
        i1 = int((math.radians(180) - angle_min) / angle_increment)
    else:
        return False
    
    # Clamp indices
    i0 = max(0, min(i0, len(lidar_data) - 1))
    i1 = max(0, min(i1, len(lidar_data) - 1))
    if i0 > i1:
        i0, i1 = i1, i0
    
    window = lidar_data[i0:i1]
    
    # Find groups of consecutive detections below threshold
    groups = []
    in_group = False
    group_start = 0
    
    for i in xrange(len(window)):
        if window[i] < distance_threshold:
            if not in_group:
                in_group = True
                group_start = i
        else:
            if in_group:
                group_length = i - group_start
                groups.append(group_length)
                in_group = False
    
    # Handle case where group extends to end
    if in_group:
        groups.append(len(window) - group_start)
    
    if not groups:
        return False
    
    # Filter for car-sized objects (5-15 degrees angular width)
    CAR_MIN_ANGLE_DEG = 5
    CAR_MAX_ANGLE_DEG = 15
    
    for group_length in groups:
        angle_width_deg = math.degrees(group_length * angle_increment)
        
        if CAR_MIN_ANGLE_DEG <= angle_width_deg <= CAR_MAX_ANGLE_DEG:
            rospy.loginfo("Opponent car detected!")
            return True
    
    return False


def lidar_callback(data):
    """Process LiDAR data and update opponent detection state with progressive counting."""
    global opponent_detected, opponent_detection_count
    
    ranges = preprocess_lidar(data)
    angle_increment = data.angle_increment
    
    detected_now = check_for_obstacles(ranges, "front", OBSTACLE_DIST_FRONT, 
                                       data.angle_min, angle_increment)
    
    # Progressive detection counter
    if detected_now:
        opponent_detected = True
        opponent_detection_count = min(opponent_detection_count + 1, MAX_DETECTION_COUNT)
    else:
        opponent_detected = False
        # Decay the counter when no detection (recover speed gradually)
        opponent_detection_count = max(opponent_detection_count - 4, 0)


# ============================================================
# PATH LOADING AND VISUALIZATION
# ============================================================

def construct_path():
    """Load waypoints from CSV file and calculate path resolution."""
    file_path = os.path.expanduser('/home/nvidia/{}.csv'.format(trajectory_name))
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for waypoint in csv_reader:
            plan.append(waypoint)
    
    # Convert to floats
    for index in range(len(plan)):
        for point in range(len(plan[index])):
            plan[index][point] = float(plan[index][point])
    
    # Calculate distances between waypoints
    for index in range(1, len(plan)):
        dx = plan[index][0] - plan[index-1][0]
        dy = plan[index][1] - plan[index-1][1]
        path_resolution.append(math.sqrt(dx*dx + dy*dy))


def publish_path_marker():
    """Publish the reference path as a green LINE_STRIP in RViz."""
    path_marker = Marker()
    path_marker.header.frame_id = frame_id
    path_marker.header.stamp = rospy.Time.now()
    path_marker.ns = "reference_path"
    path_marker.id = 0
    path_marker.type = Marker.LINE_STRIP
    path_marker.action = Marker.ADD
    
    path_marker.scale.x = 0.05
    path_marker.color.r = 0.0
    path_marker.color.g = 1.0
    path_marker.color.b = 0.0
    path_marker.color.a = 0.8
    
    # Add all waypoints
    for waypoint in plan:
        p = Point()
        p.x = waypoint[0]
        p.y = waypoint[1]
        p.z = 0.0
        path_marker.points.append(p)
    
    # Close the loop
    if len(plan) > 0:
        p = Point()
        p.x = plan[0][0]
        p.y = plan[0][1]
        p.z = 0.0
        path_marker.points.append(p)
    
    path_marker.lifetime = rospy.Duration(0)
    path_marker_pub.publish(path_marker)


def publish_visualization_markers(odom_x, odom_y, heading, pose_x, pose_y, 
                                  target_x, target_y, steering_angle, lookahead_distance):
    """Publish visualization markers for pose, target, lookahead line, and steering arrow."""
    
    # Blue sphere for base projection on path
    pose_marker = Marker()
    pose_marker.header.frame_id = frame_id
    pose_marker.header.stamp = rospy.Time.now()
    pose_marker.ns = "base_projection"
    pose_marker.id = 1
    pose_marker.type = Marker.SPHERE
    pose_marker.action = Marker.ADD
    pose_marker.pose.position.x = pose_x
    pose_marker.pose.position.y = pose_y
    pose_marker.pose.position.z = 0.1
    pose_marker.pose.orientation.w = 1.0
    pose_marker.scale.x = 0.15
    pose_marker.scale.y = 0.15
    pose_marker.scale.z = 0.15
    pose_marker.color.r = 0.0
    pose_marker.color.g = 0.0
    pose_marker.color.b = 1.0
    pose_marker.color.a = 1.0
    pose_marker.lifetime = rospy.Duration(0.1)
    pose_marker_pub.publish(pose_marker)
    
    # Red sphere for target point
    target_marker = Marker()
    target_marker.header.frame_id = frame_id
    target_marker.header.stamp = rospy.Time.now()
    target_marker.ns = "target_point"
    target_marker.id = 2
    target_marker.type = Marker.SPHERE
    target_marker.action = Marker.ADD
    target_marker.pose.position.x = target_x
    target_marker.pose.position.y = target_y
    target_marker.pose.position.z = 0.1
    target_marker.pose.orientation.w = 1.0
    target_marker.scale.x = 0.2
    target_marker.scale.y = 0.2
    target_marker.scale.z = 0.2
    target_marker.color.r = 1.0
    target_marker.color.g = 0.0
    target_marker.color.b = 0.0
    target_marker.color.a = 1.0
    target_marker.lifetime = rospy.Duration(0.1)
    target_marker_pub.publish(target_marker)
    
    # Yellow line from car to target
    lookahead_marker = Marker()
    lookahead_marker.header.frame_id = frame_id
    lookahead_marker.header.stamp = rospy.Time.now()
    lookahead_marker.ns = "lookahead_line"
    lookahead_marker.id = 3
    lookahead_marker.type = Marker.LINE_STRIP
    lookahead_marker.action = Marker.ADD
    lookahead_marker.scale.x = 0.05
    lookahead_marker.color.r = 1.0
    lookahead_marker.color.g = 1.0
    lookahead_marker.color.b = 0.0
    lookahead_marker.color.a = 0.8
    
    p_start = Point()
    p_start.x = odom_x
    p_start.y = odom_y
    p_start.z = 0.1
    lookahead_marker.points.append(p_start)
    
    p_end = Point()
    p_end.x = target_x
    p_end.y = target_y
    p_end.z = 0.1
    lookahead_marker.points.append(p_end)
    
    lookahead_marker.lifetime = rospy.Duration(0.1)
    lookahead_marker_pub.publish(lookahead_marker)
    
    # Magenta arrow for steering direction
    steering_marker = Marker()
    steering_marker.header.frame_id = frame_id
    steering_marker.header.stamp = rospy.Time.now()
    steering_marker.ns = "steering_angle"
    steering_marker.id = 4
    steering_marker.type = Marker.ARROW
    steering_marker.action = Marker.ADD
    
    steering_heading = heading + steering_angle
    arrow_length = 0.5
    
    p_arrow_start = Point()
    p_arrow_start.x = odom_x + 0.2 * math.cos(heading)
    p_arrow_start.y = odom_y + 0.2 * math.sin(heading)
    p_arrow_start.z = 0.15
    
    p_arrow_end = Point()
    p_arrow_end.x = p_arrow_start.x + arrow_length * math.cos(steering_heading)
    p_arrow_end.y = p_arrow_start.y + arrow_length * math.sin(steering_heading)
    p_arrow_end.z = 0.15
    
    steering_marker.points.append(p_arrow_start)
    steering_marker.points.append(p_arrow_end)
    
    steering_marker.scale.x = 0.05
    steering_marker.scale.y = 0.1
    steering_marker.scale.z = 0.1
    
    steering_marker.color.r = 1.0
    steering_marker.color.g = 0.0
    steering_marker.color.b = 1.0
    steering_marker.color.a = 1.0
    
    steering_marker.lifetime = rospy.Duration(0.1)
    steering_marker_pub.publish(steering_marker)


# ============================================================
# PURE PURSUIT CONTROL
# ============================================================

def compute_lookahead_distance():
    """
    Adaptive lookahead based on current speed.
    Slower speeds use shorter lookahead, faster speeds use longer lookahead.
    """
    v = current_speed_cmd
    
    if v <= V_MIN_LD:
        return L_MIN
    if v >= V_MAX_LD:
        return L_MAX
    
    # Linear interpolation
    ratio = (v - V_MIN_LD) / float(V_MAX_LD - V_MIN_LD)
    return L_MIN + ratio * (L_MAX - L_MIN)


def purepursuit_control_node(data):
    """Main pure pursuit control loop."""
    global wp_seq, current_speed_cmd
    
    command = AckermannDrive()
    
    # Get current car position
    odom_x = data.pose.position.x
    odom_y = data.pose.position.y
    
    # Find closest point on path (base projection)
    min_distance = float('inf')
    base_index = 0
    
    for i in range(len(plan)):
        dx = plan[i][0] - odom_x
        dy = plan[i][1] - odom_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < min_distance:
            min_distance = distance
            base_index = i
    
    pose_x = plan[base_index][0]
    pose_y = plan[base_index][1]
    
    # Get car heading
    heading = tf.transformations.euler_from_quaternion((data.pose.orientation.x,
                                                        data.pose.orientation.y,
                                                        data.pose.orientation.z,
                                                        data.pose.orientation.w))[2]
    
    # Compute adaptive lookahead distance
    lookahead_distance = compute_lookahead_distance()
    
    # Find target point on path
    cumulative_distance = 0.0
    target_index = base_index
    num_points = len(plan)
    
    for j in range(num_points):
        i = (base_index + j) % num_points
        i_next = (i + 1) % num_points
        
        dx = plan[i_next][0] - plan[i][0]
        dy = plan[i_next][1] - plan[i][1]
        segment_distance = math.sqrt(dx*dx + dy*dy)
        
        if cumulative_distance + segment_distance >= lookahead_distance:
            remaining_distance = lookahead_distance - cumulative_distance
            ratio = remaining_distance / segment_distance
            target_x = plan[i][0] + ratio * dx
            target_y = plan[i][1] + ratio * dy
            target_index = i
            break
        
        cumulative_distance += segment_distance
        target_index = i_next
    else:
        target_x = plan[target_index][0]
        target_y = plan[target_index][1]
    
    # Transform target to vehicle frame
    dx = target_x - odom_x
    dy = target_y - odom_y
    
    target_x_vehicle = math.cos(-heading) * dx - math.sin(-heading) * dy
    target_y_vehicle = math.sin(-heading) * dx + math.cos(-heading) * dy
    
    # Calculate steering using pure pursuit formula
    curvature = 2.0 * target_y_vehicle / (lookahead_distance ** 2)
    steering_angle = math.atan(WHEELBASE_LEN * curvature)
    
    # Convert to steering command range
    max_steering_angle_rad = 0.4
    normalized_steering = steering_angle / max_steering_angle_rad
    command.steering_angle = max(-STEERING_RANGE, min(STEERING_RANGE, normalized_steering * STEERING_RANGE))
    
    # Dynamic velocity scaling based on steering angle
    abs_steering = abs(command.steering_angle)
    max_speed = 65.0
    min_speed = 10.0
    
    speed_scale = 1.0 - (abs_steering / STEERING_RANGE)
    command.speed = min_speed + (max_speed - min_speed) * speed_scale
    
    # Progressive slowdown based on detection count
    if opponent_detection_count > 0:
        # Calculate speed reduction factor based on detection count
        # Goes from 1.0 (no reduction) to MIN_SPEED_FACTOR (heavy reduction)
        slowdown_factor = 1.0 - (opponent_detection_count / float(MAX_DETECTION_COUNT)) * (1.0 - MIN_SPEED_FACTOR)
        command.speed = command.speed * slowdown_factor
        
        # Stop completely if detection count is maxed out
        if opponent_detection_count >= MAX_DETECTION_COUNT:
            command.speed = 0.0
            rospy.logwarn("Stopped: Obstacle persists (count: %d)", opponent_detection_count)
        else:
            rospy.loginfo("Slowing down: count=%d, speed=%.1f (factor=%.2f)", 
                         opponent_detection_count, command.speed, slowdown_factor)
    
    current_speed_cmd = command.speed
    
    # Publish command
    command_pub.publish(command)
    
    # Publish visualization markers
    publish_visualization_markers(odom_x, odom_y, heading, pose_x, pose_y, 
                                  target_x, target_y, steering_angle, lookahead_distance)
    
    # Publish control polygon
    base_link = Point32()
    nearest_pose = Point32()
    nearest_goal = Point32()
    base_link.x = odom_x
    base_link.y = odom_y
    nearest_pose.x = pose_x
    nearest_pose.y = pose_y
    nearest_goal.x = target_x
    nearest_goal.y = target_y
    
    control_polygon.header.frame_id = frame_id
    control_polygon.polygon.points = [nearest_pose, base_link, nearest_goal]
    control_polygon.header.seq = wp_seq
    control_polygon.header.stamp = rospy.Time.now()
    wp_seq = wp_seq + 1
    polygon_pub.publish(control_polygon)


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    global command_pub, polygon_pub, path_marker_pub, pose_marker_pub
    global target_marker_pub, steering_marker_pub, lookahead_marker_pub
    
    try:
        rospy.init_node('final_race', anonymous=True)
        rospy.loginfo("Final race node started.")
        
        # Initialize publishers
        command_pub = rospy.Publisher('/car_2/offboard/command', AckermannDrive, queue_size=1)
        polygon_pub = rospy.Publisher('/car_2/purepursuit_control/visualize', PolygonStamped, queue_size=1)
        path_marker_pub = rospy.Publisher('/car_2/purepursuit_control/path_marker', Marker, queue_size=1)
        pose_marker_pub = rospy.Publisher('/car_2/purepursuit_control/pose_marker', Marker, queue_size=1)
        target_marker_pub = rospy.Publisher('/car_2/purepursuit_control/target_marker', Marker, queue_size=1)
        steering_marker_pub = rospy.Publisher('/car_2/purepursuit_control/steering_marker', Marker, queue_size=1)
        lookahead_marker_pub = rospy.Publisher('/car_2/purepursuit_control/lookahead_marker', Marker, queue_size=1)
        
        # Load trajectory
        if not plan:
            rospy.loginfo('Loading trajectory from %s.csv', trajectory_name)
            construct_path()
            rospy.sleep(0.5)
            publish_path_marker()
            rospy.loginfo('Published reference path to RViz')
        
        # Subscribe to sensors
        rospy.Subscriber('/car_2/scan', LaserScan, lidar_callback)
        rospy.Subscriber('/car_2/particle_filter/viz/inferred_pose', PoseStamped, purepursuit_control_node)
        
        rospy.spin()
        
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()