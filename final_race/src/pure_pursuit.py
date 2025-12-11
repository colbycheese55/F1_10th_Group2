#!/usr/bin/env python

# Import necessary libraries
import rospy
import os
import sys
import csv
import math
from ackermann_msgs.msg import AckermannDrive
from geometry_msgs.msg import PolygonStamped
from geometry_msgs.msg import Point32
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import tf

import final

# Global variables for storing the path, path resolution, frame ID, and car details
plan                = []
path_resolution     = []
frame_id            = 'map'
trajectory_name     = str(sys.argv[1])

# Publishers for sending driving commands and visualizing the control polygon
command_pub         = rospy.Publisher('/car_2/offboard/command', AckermannDrive, queue_size = 1)
polygon_pub         = rospy.Publisher('/car_2/purepursuit_control/visualize', PolygonStamped, queue_size = 1)

# Publishers for RViz visualization markers
path_marker_pub     = rospy.Publisher('/car_2/purepursuit_control/path_marker', Marker, queue_size = 1)
pose_marker_pub     = rospy.Publisher('/car_2/purepursuit_control/pose_marker', Marker, queue_size = 1)
target_marker_pub   = rospy.Publisher('/car_2/purepursuit_control/target_marker', Marker, queue_size = 1)
steering_marker_pub = rospy.Publisher('/car_2/purepursuit_control/steering_marker', Marker, queue_size = 1)
lookahead_marker_pub = rospy.Publisher('/car_2/purepursuit_control/lookahead_marker', Marker, queue_size = 1)

# Global variables for waypoint sequence and current polygon
global wp_seq
global curr_polygon

wp_seq          = 0
control_polygon = PolygonStamped()

def construct_path():
    # Function to construct the path from a CSV file
    # TODO: Modify this path to match the folder where the csv file containing the path is located.
    file_path = os.path.expanduser('/home/nvidia/{}.csv'.format(trajectory_name))
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        for waypoint in csv_reader:
            plan.append(waypoint)

    # Convert string coordinates to floats and calculate path resolution
    for index in range(0, len(plan)):
        for point in range(0, len(plan[index])):
            plan[index][point] = float(plan[index][point])

    for index in range(1, len(plan)):
         dx = plan[index][0] - plan[index-1][0]
         dy = plan[index][1] - plan[index-1][1]
         path_resolution.append(math.sqrt(dx*dx + dy*dy))


def publish_path_marker():
    """Publish the reference path as a LINE_STRIP marker for RViz visualization."""
    path_marker = Marker()
    path_marker.header.frame_id = frame_id
    path_marker.header.stamp = rospy.Time.now()
    path_marker.ns = "reference_path"
    path_marker.id = 0
    path_marker.type = Marker.LINE_STRIP
    path_marker.action = Marker.ADD
    
    # Set the scale (line width)
    path_marker.scale.x = 0.05  # Line width
    
    # Set the color (green for reference path)
    path_marker.color.r = 0.0
    path_marker.color.g = 1.0
    path_marker.color.b = 0.0
    path_marker.color.a = 0.8
    
    # Add all waypoints to the marker
    for waypoint in plan:
        p = Point()
        p.x = waypoint[0]
        p.y = waypoint[1]
        p.z = 0.0
        path_marker.points.append(p)
    
    # Make the path a closed loop by connecting back to start
    if len(plan) > 0:
        p = Point()
        p.x = plan[0][0]
        p.y = plan[0][1]
        p.z = 0.0
        path_marker.points.append(p)
    
    path_marker.lifetime = rospy.Duration(0)  # Never expire
    path_marker_pub.publish(path_marker)


def publish_visualization_markers(odom_x, odom_y, heading, pose_x, pose_y, target_x, target_y, steering_angle, lookahead_distance):
    """Publish visualization markers for pose, target, and steering angle."""
    
    # --- Pose Marker (base projection on path) - Blue Sphere ---
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
    
    # --- Target Marker (lookahead point) - Red Sphere ---
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
    
    # --- Lookahead Line (from car to target) - Yellow Line ---
    lookahead_marker = Marker()
    lookahead_marker.header.frame_id = frame_id
    lookahead_marker.header.stamp = rospy.Time.now()
    lookahead_marker.ns = "lookahead_line"
    lookahead_marker.id = 3
    lookahead_marker.type = Marker.LINE_STRIP
    lookahead_marker.action = Marker.ADD
    lookahead_marker.scale.x = 0.05  # Line width
    lookahead_marker.color.r = 1.0
    lookahead_marker.color.g = 1.0
    lookahead_marker.color.b = 0.0
    lookahead_marker.color.a = 0.8
    
    # Start point (car position)
    p_start = Point()
    p_start.x = odom_x
    p_start.y = odom_y
    p_start.z = 0.1
    lookahead_marker.points.append(p_start)
    
    # End point (target)
    p_end = Point()
    p_end.x = target_x
    p_end.y = target_y
    p_end.z = 0.1
    lookahead_marker.points.append(p_end)
    
    lookahead_marker.lifetime = rospy.Duration(0.1)
    lookahead_marker_pub.publish(lookahead_marker)
    
    # --- Steering Angle Arrow - Magenta Arrow ---
    steering_marker = Marker()
    steering_marker.header.frame_id = frame_id
    steering_marker.header.stamp = rospy.Time.now()
    steering_marker.ns = "steering_angle"
    steering_marker.id = 4
    steering_marker.type = Marker.ARROW
    steering_marker.action = Marker.ADD
    
    # Calculate the steering direction in world frame
    # The steering angle is relative to the car's heading
    steering_heading = heading + steering_angle
    arrow_length = 0.5  # Arrow length in meters
    
    # Arrow start point (front of car)
    p_arrow_start = Point()
    p_arrow_start.x = odom_x + 0.2 * math.cos(heading)  # Slightly in front of car
    p_arrow_start.y = odom_y + 0.2 * math.sin(heading)
    p_arrow_start.z = 0.15
    
    # Arrow end point (in steering direction)
    p_arrow_end = Point()
    p_arrow_end.x = p_arrow_start.x + arrow_length * math.cos(steering_heading)
    p_arrow_end.y = p_arrow_start.y + arrow_length * math.sin(steering_heading)
    p_arrow_end.z = 0.15
    
    steering_marker.points.append(p_arrow_start)
    steering_marker.points.append(p_arrow_end)
    
    # Arrow dimensions
    steering_marker.scale.x = 0.05  # Shaft diameter
    steering_marker.scale.y = 0.1   # Head diameter
    steering_marker.scale.z = 0.1   # Head length
    
    # Magenta color
    steering_marker.color.r = 1.0
    steering_marker.color.g = 0.0
    steering_marker.color.b = 1.0
    steering_marker.color.a = 1.0
    
    steering_marker.lifetime = rospy.Duration(0.1)
    steering_marker_pub.publish(steering_marker)


# Steering Range from -100.0 to 100.0
STEERING_RANGE = 100.0

# vehicle physical parameters
WHEELBASE_LEN       = 0.325

# -------- Adaptive lookahead parameters --------
# Lookahead limits (meters)
L_MIN    = 0.8
L_MAX    = 2

# Speed range where we interpolate (same units as command.speed)
V_MIN_LD = 8.0
V_MAX_LD = 15.0

# last commanded speed (used as our speed estimate)
current_speed_cmd = 0.0


def compute_lookahead_distance():
    """
    Adaptive lookahead based on last commanded speed.
    Below V_MIN_LD -> L_MIN
    Above V_MAX_LD -> L_MAX
    In between     -> linear interpolation.
    """
    v = current_speed_cmd

    if v <= V_MIN_LD:
        return L_MIN
    if v >= V_MAX_LD:
        return L_MAX

    # Linear interpolation between L_MIN and L_MAX
    ratio = (v - V_MIN_LD) / float(V_MAX_LD - V_MIN_LD)
    return L_MIN + ratio * (L_MAX - L_MIN)


def purepursuit_control_node(data):
    # Main control function for pure pursuit algorithm

    # Create an empty ackermann drive message that we will populate later with the desired steering angle and speed.
    command = AckermannDrive()

    global wp_seq
    global curr_polygon
    global current_speed_cmd

    # Obtain the current position of the race car from the inferred_pose message
    odom_x = data.pose.position.x
    odom_y = data.pose.position.y


    # TODO 1: The reference path is stored in the 'plan' array.
    # Your task is to find the base projection of the car on this reference path.
    # The base projection is defined as the closest point on the reference path to the car's current position.
    # Calculate the index and position of this base projection on the reference path.
    
    # Find the closest point on the reference path to the car's current position
    min_distance = float('inf')
    base_index = 0
    
    for i in range(len(plan)):
        dx = plan[i][0] - odom_x
        dy = plan[i][1] - odom_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < min_distance:
            min_distance = distance
            base_index = i
    
    # Store the base projection coordinates
    pose_x = plan[base_index][0]
    pose_y = plan[base_index][1]

    
    # Calculate heading angle of the car (in radians)
    heading = tf.transformations.euler_from_quaternion((data.pose.orientation.x,
                                                        data.pose.orientation.y,
                                                        data.pose.orientation.z,
                                                        data.pose.orientation.w))[2]
    

    # TODO 2: You need to tune the value of the lookahead_distance
    # The lookahead distance affects how far ahead the car looks on the path
    # Smaller values make the car follow the path more tightly but may cause oscillations
    # Larger values make the car's path smoother but may cut corners
    # Start with 1.0m and tune based on performance (typical range: 0.5m - 2.0m)
    lookahead_distance = compute_lookahead_distance() # Fine-tuned from default 1.0m


    # TODO 3: Utilizing the base projection found in TODO 1, your next task is to identify the goal or target point for the car.
    # This target point should be determined based on the path and the base projection you have already calculated.
    # The target point is a specific point on the reference path that the car should aim towards - lookahead distance ahead of the base projection on the reference path.
    # Calculate the position of this goal/target point along the path.

    # Start from the base projection and move forward along the path
    cumulative_distance = 0.0
    target_index = base_index
    
    # Traverse the path until we've covered the lookahead distance
    for i in range(base_index, len(plan) - 1):
        dx = plan[i+1][0] - plan[i][0]
        dy = plan[i+1][1] - plan[i][1]
        segment_distance = math.sqrt(dx*dx + dy*dy)
        
        if cumulative_distance + segment_distance >= lookahead_distance:
            # Interpolate to find the exact target point
            remaining_distance = lookahead_distance - cumulative_distance
            ratio = remaining_distance / segment_distance
            target_x = plan[i][0] + ratio * dx
            target_y = plan[i][1] + ratio * dy
            target_index = i
            break
        
        cumulative_distance += segment_distance
        target_index = i + 1
    else:
        # If we've reached the end of the path, use the last point
        target_x = plan[-1][0]
        target_y = plan[-1][1]


    # TODO 4: Implement the pure pursuit algorithm to compute the steering angle given the pose of the car, target point, and lookahead distance.
    # Transform target point to vehicle frame of reference
    dx = target_x - odom_x
    dy = target_y - odom_y
    
    # Rotate the target point to the vehicle's coordinate frame
    target_x_vehicle = math.cos(-heading) * dx - math.sin(-heading) * dy
    target_y_vehicle = math.sin(-heading) * dx + math.cos(-heading) * dy
    
    # Calculate the curvature using pure pursuit formula
    # curvature = 2 * y / L^2, where y is lateral offset and L is lookahead distance
    curvature = 2.0 * target_y_vehicle / (lookahead_distance ** 2)
    
    # Calculate steering angle using bicycle model
    # steering_angle = atan(wheelbase * curvature)
    steering_angle = math.atan(WHEELBASE_LEN * curvature)


    # TODO 5: Ensure that the calculated steering angle is within the STEERING_RANGE and assign it to command.steering_angle
    # Convert steering angle from radians to the range [-100, 100]
    # Assuming maximum steering angle is around 0.4 radians (about 23 degrees)
    max_steering_angle_rad = 0.4
    
    # Normalize to [-1, 1] then scale to [-100, 100]
    normalized_steering = steering_angle / max_steering_angle_rad
    command.steering_angle = max(-STEERING_RANGE, min(STEERING_RANGE, normalized_steering * STEERING_RANGE))

    # TODO 6: Implement Dynamic Velocity Scaling instead of a constant speed
    # Scale velocity based on steering angle - slow down for sharp turns
    # Use absolute value of steering angle to determine speed
    abs_steering = abs(command.steering_angle)
    
    # Define speed parameters
    max_speed = 65.0  # Maximum speed on straightaways
    min_speed = 5.0   # Minimum speed for sharp turns
    
    # Linear velocity scaling based on steering angle
    # When steering is 0, speed is max_speed
    # When steering is at max (100), speed is min_speed
    speed_scale = 1.0 - (abs_steering / STEERING_RANGE)
    command.speed = min_speed + (max_speed - min_speed) * speed_scale
    current_speed_cmd = command.speed

    # coordination between modes function
    final.pure_pursuit_recent_cmd = command
    if not final.drive_mode == "pure_pursuit":
        return

    command_pub.publish(command)

    # Publish RViz visualization markers for reference path, pose, target, and steering angle
    publish_visualization_markers(odom_x, odom_y, heading, pose_x, pose_y, target_x, target_y, steering_angle, lookahead_distance)

    # Visualization code
    # Make sure the following variables are properly defined in your TODOs above:
    # - odom_x, odom_y: Current position of the car
    # - pose_x, pose_y: Position of the base projection on the reference path
    # - target_x, target_y: Position of the goal/target point


    base_link    = Point32()
    nearest_pose = Point32()
    nearest_goal = Point32()
    base_link.x    = odom_x
    base_link.y    = odom_y
    nearest_pose.x = pose_x
    nearest_pose.y = pose_y
    nearest_goal.x = target_x
    nearest_goal.y = target_y
    control_polygon.header.frame_id = frame_id
    control_polygon.polygon.points  = [nearest_pose, base_link, nearest_goal]
    control_polygon.header.seq      = wp_seq
    control_polygon.header.stamp    = rospy.Time.now()
    wp_seq = wp_seq + 1
    polygon_pub.publish(control_polygon)

if __name__ == '__main__':

    try:

        rospy.init_node('pure_pursuit', anonymous = True)
        if not plan:
            rospy.loginfo('obtaining trajectory')
            construct_path()
            # Publish the reference path marker once after loading
            rospy.sleep(0.5)  # Wait for publishers to be ready
            publish_path_marker()
            rospy.loginfo('Published reference path to RViz')

        # This node subsribes to the pose estimate provided by the Particle Filter. 
        # The message type of that pose message is PoseStamped which belongs to the geometry_msgs ROS package.
        rospy.Subscriber('/car_2/particle_filter/viz/inferred_pose', PoseStamped, purepursuit_control_node)
        rospy.spin()

    except rospy.ROSInterruptException:

        pass
