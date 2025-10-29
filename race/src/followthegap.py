#!/usr/bin/env python

import rospy
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDrive
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import tf2_ros
import geometry_msgs.msg

# Configuration parameters
DISPARITY_THRESHOLD = 0.1  # Threshold for detecting disparities (meters)
CAR_WIDTH = 0.20  # Width of the car (meters)
SAFETY_MARGIN = 0.15  # Additional safety margin around the car (meters)
MAX_VELOCITY = 40.0  # Maximum velocity on straightaways
MIN_VELOCITY = 15.0  # Minimum velocity during turns
MAX_STEERING = 100.0  # Maximum steering angle
MIN_STEERING = -100.0  # Minimum steering angle
CAR_LENGTH = 0.50  # Length of the car (meters)

# Publishers
command_pub = rospy.Publisher('/car_2/offboard/command', AckermannDrive, queue_size=1)
car_footprint_pub = rospy.Publisher('/follow_gap/car_footprint', Marker, queue_size=1)
steering_arrow_pub = rospy.Publisher('/follow_gap/steering_arrow', Marker, queue_size=1)
disparities_pub = rospy.Publisher('/follow_gap/disparities', MarkerArray, queue_size=1)
target_point_pub = rospy.Publisher('/follow_gap/target_point', Marker, queue_size=1)
processed_scan_pub = rospy.Publisher('/follow_gap/processed_scan', LaserScan, queue_size=1)


def preprocess_lidar(ranges, angle_min, angle_increment):
    """
    Preprocess LIDAR data: handle NaN and inf values, return valid ranges
    """
    processed_ranges = np.array(ranges)
    
    # Replace NaN and inf values with a large number (treat as no obstacle)
    processed_ranges[np.isnan(processed_ranges)] = 10.0
    processed_ranges[np.isinf(processed_ranges)] = 10.0
    
    return processed_ranges


def find_disparities(ranges, threshold=DISPARITY_THRESHOLD):
    """
    Find disparities in LIDAR readings where consecutive points differ by more than threshold
    Returns list of indices where disparities occur
    """
    disparities = []
    
    for i in range(len(ranges) - 1):
        diff = abs(ranges[i] - ranges[i + 1])
        if diff > threshold:
            disparities.append(i)
    
    return disparities


def calculate_bubble_radius(distance):
    """
    Calculate the number of LIDAR samples needed to cover half the car width plus safety margin
    at a given distance
    """
    # Radius needed to cover half car width plus safety margin
    radius = (CAR_WIDTH / 2.0) + SAFETY_MARGIN
    
    # Calculate angle subtended by this radius at the given distance
    # angle = arcsin(radius / distance)
    if distance < radius:
        # If obstacle is too close, use a large angle
        return 1.0  # radians
    
    angle = math.asin(min(radius / distance, 1.0))
    
    return angle


def extend_disparities(ranges, angle_increment, disparities):
    """
    For each disparity, extend the closer distance to cover the car's width plus safety margin
    """
    processed_ranges = np.copy(ranges)
    
    for disp_idx in disparities:
        # Get the two points around the disparity
        dist1 = ranges[disp_idx]
        dist2 = ranges[disp_idx + 1]
        
        # Determine which is closer and which is farther
        if dist1 < dist2:
            closer_dist = dist1
            closer_idx = disp_idx
            farther_idx = disp_idx + 1
            direction = 1  # Extend to the right (increasing indices)
        else:
            closer_dist = dist2
            closer_idx = disp_idx + 1
            farther_idx = disp_idx
            direction = -1  # Extend to the left (decreasing indices)
        
        # Calculate bubble radius in terms of angle
        bubble_angle = calculate_bubble_radius(closer_dist)
        
        # Calculate number of samples to cover
        num_samples = int(bubble_angle / angle_increment)
        
        # Extend from the farther point in the direction away from closer point
        for i in range(num_samples):
            idx = farther_idx + (direction * i)
            
            # Check bounds
            if idx < 0 or idx >= len(processed_ranges):
                break
            
            # Only overwrite if the current value is farther (don't overwrite closer obstacles)
            if processed_ranges[idx] > closer_dist:
                processed_ranges[idx] = closer_dist
    
    return processed_ranges


def get_angle_range_indices(angle_min, angle_increment, num_ranges):
    """
    Get indices corresponding to angles between -90 and +90 degrees
    (front-facing semicircle)
    
    LIDAR coordinate frame:
    - Scans from right to left with 240 FoV
    - angle_min = -30 (right side), angle_max = 210 (left side)
    - 0 is directly to the right
    - 90 is directly in front
    - 180 is directly to the left
    
    We want the front-facing semicircle, so we look at LIDAR angles from 
    approximately 30 (45 right of front) to 150 (45 left of front)
    to avoid looking too far to the sides or behind.
    """
    # In LIDAR frame, we want angles roughly from 30 to 150 
    # (covering front 120 with some margin)
    # This corresponds to -60 to +60 relative to car's forward direction
    
    target_min_angle = math.radians(30)   # 30 in LIDAR frame
    target_max_angle = math.radians(150)  # 150 in LIDAR frame
    
    # Find corresponding indices
    start_idx = int((target_min_angle - angle_min) / angle_increment)
    end_idx = int((target_max_angle - angle_min) / angle_increment)
    
    # Clamp to valid range
    start_idx = max(0, min(start_idx, num_ranges - 1))
    end_idx = max(0, min(end_idx, num_ranges - 1))
    
    return start_idx, end_idx


def find_best_gap(ranges, angle_min, angle_increment, start_idx, end_idx):
    """
    Find the best gap (deepest point in the widest/deepest gap) in the specified range
    Returns the index and angle of the target point
    """
    # Define minimum distance to consider as "free space"
    min_free_distance = 0.5  # meters
    
    # Find all gaps (contiguous sequences of free space)
    gaps = []
    in_gap = False
    gap_start = start_idx
    
    for i in range(start_idx, end_idx + 1):
        if ranges[i] > min_free_distance:
            if not in_gap:
                # Start of a new gap
                gap_start = i
                in_gap = True
        else:
            if in_gap:
                # End of current gap
                gaps.append((gap_start, i - 1))
                in_gap = False
    
    # Don't forget the last gap if it extends to the end
    if in_gap:
        gaps.append((gap_start, end_idx))
    
    # If no gaps found, return the farthest point as fallback
    if len(gaps) == 0:
        max_dist = 0.0
        max_idx = start_idx
        for i in range(start_idx, end_idx + 1):
            if ranges[i] > max_dist:
                max_dist = ranges[i]
                max_idx = i
        target_angle = angle_min + (max_idx * angle_increment)
        return max_idx, target_angle, max_dist
    
    # Find the best gap: deepest point in the largest gap
    best_gap = None
    best_gap_score = 0.0
    
    for gap_start, gap_end in gaps:
        # Calculate gap width (angular extent)
        gap_width = gap_end - gap_start + 1
        
        # Find the deepest (farthest) point in this gap
        max_dist_in_gap = 0.0
        max_idx_in_gap = gap_start
        for i in range(gap_start, gap_end + 1):
            if ranges[i] > max_dist_in_gap:
                max_dist_in_gap = ranges[i]
                max_idx_in_gap = i
        
        # Score combines gap width and depth
        # Prioritize depth more than width
        gap_score = max_dist_in_gap * 2.0 + gap_width * 0.01
        
        if gap_score > best_gap_score:
            best_gap_score = gap_score
            best_gap = (max_idx_in_gap, max_dist_in_gap)
    
    # Target the deepest point in the best gap
    target_idx = best_gap[0]
    max_distance = best_gap[1]
    target_angle = angle_min + (target_idx * angle_increment)
    
    return target_idx, target_angle, max_distance


def calculate_steering_angle(target_angle_rad):
    """
    Convert target angle (in radians) to steering command [-100, 100]
    
    LIDAR coordinate frame:
    - angle_min ≈ -30 (right side)
    - 0 is directly to the right
    - 90 is directly in front
    - 180 is directly to the left
    - angle_max ≈ 210 (left side)
    
    Steering:
    - 0 = straight ahead
    - Positive = turn left
    - Negative = turn right
    """
    # Convert radians to degrees
    target_angle_deg = math.degrees(target_angle_rad)
    
    # In LIDAR frame, 90 is straight ahead (directly in front)
    # Calculate deviation from straight (90)
    angle_deviation = target_angle_deg - 90.0
    
    # Map angle deviation to steering range
    # If target is > 90 (to the left), steering should be positive (turn left)
    # If target is < 90 (to the right), steering should be negative (turn right)
    # Assuming reasonable steering angles: 45 deviation maps to 100 steering
    steering_angle = (angle_deviation / 45.0) * 100.0
    
    # Clamp to valid range
    steering_angle = max(MIN_STEERING, min(MAX_STEERING, steering_angle))
    
    return steering_angle


def calculate_velocity(steering_angle, max_distance):
    """
    Calculate velocity based on steering angle and distance to obstacle
    Higher speed on straights, lower speed on turns
    """
    # Normalize steering angle to [0, 1]
    steering_magnitude = abs(steering_angle) / MAX_STEERING
    
    # Velocity inversely proportional to steering magnitude
    velocity = MIN_VELOCITY + (MAX_VELOCITY - MIN_VELOCITY) * (1.0 - steering_magnitude)
    
    # Also consider distance: slow down if obstacles are close
    if max_distance < 1.0:
        velocity *= 0.5
    elif max_distance < 2.0:
        velocity *= 0.75
    
    # Clamp to valid range
    velocity = max(0, min(100, velocity))
    
    return velocity


def publish_car_footprint():
    """
    Publish car footprint visualization as a rectangle in RViz
    """
    marker = Marker()
    marker.header.frame_id = "car_2_base_link"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "car_footprint"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    
    marker.scale.x = 0.05  # Line width
    
    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 1.0
    marker.color.a = 1.0
    
    # Define the rectangle corners (car footprint)
    half_length = CAR_LENGTH / 2.0
    half_width = CAR_WIDTH / 2.0
    
    # Create closed rectangle
    p1 = Point(half_length, -half_width, 0.0)  # Front-right
    p2 = Point(half_length, half_width, 0.0)   # Front-left
    p3 = Point(-half_length, half_width, 0.0)  # Rear-left
    p4 = Point(-half_length, -half_width, 0.0) # Rear-right
    
    marker.points = [p1, p2, p3, p4, p1]
    marker.lifetime = rospy.Duration(0)
    
    car_footprint_pub.publish(marker)


def publish_steering_arrow(steering_angle):
    """
    Publish steering direction visualization as an arrow in RViz
    """
    marker = Marker()
    marker.header.frame_id = "car_2_base_link"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "steering_arrow"
    marker.id = 1
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    
    # Arrow dimensions
    marker.scale.x = 0.05  # Shaft diameter
    marker.scale.y = 0.08  # Head diameter
    marker.scale.z = 0.1   # Head length
    
    # Color based on steering direction (green for straight, red for sharp turns)
    steering_magnitude = abs(steering_angle) / 100.0
    marker.color.r = steering_magnitude
    marker.color.g = 1.0 - steering_magnitude
    marker.color.b = 0.0
    marker.color.a = 1.0
    
    # Convert steering angle to radians
    # Map [-100, 100] to approximately [-45, 45] degrees
    angle_rad = (steering_angle / 100.0) * (math.pi / 4.0)
    
    # Start point (at front of car)
    start = Point(CAR_LENGTH / 2.0, 0.0, 0.1)
    
    # End point (steering direction)
    arrow_length = 0.5
    end = Point()
    end.x = start.x + arrow_length * math.cos(angle_rad)
    end.y = start.y + arrow_length * math.sin(angle_rad)
    end.z = start.z
    
    marker.points = [start, end]
    marker.lifetime = rospy.Duration(0.1)
    
    steering_arrow_pub.publish(marker)


def publish_disparities(disparities, ranges, angle_min, angle_increment):
    """
    Publish disparity locations as spheres in RViz
    """
    marker_array = MarkerArray()
    
    for i, disp_idx in enumerate(disparities):
        marker = Marker()
        marker.header.frame_id = "car_2_laser"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "disparities"
        marker.id = i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # Get the angle and distance for this disparity
        angle = angle_min + (disp_idx * angle_increment)
        distance = ranges[disp_idx]
        
        # Convert to Cartesian coordinates
        marker.pose.position.x = distance * math.cos(angle)
        marker.pose.position.y = distance * math.sin(angle)
        marker.pose.position.z = 0.0
        
        marker.pose.orientation.w = 1.0
        
        # Sphere size
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        
        # Color: red for disparities
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        marker.lifetime = rospy.Duration(0.1)
        marker_array.markers.append(marker)
    
    disparities_pub.publish(marker_array)


def publish_target_point(target_angle, target_distance):
    """
    Publish target point as a large sphere in RViz
    """
    marker = Marker()
    marker.header.frame_id = "car_2_laser"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "target_point"
    marker.id = 0
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    
    # Convert to Cartesian coordinates
    marker.pose.position.x = target_distance * math.cos(target_angle)
    marker.pose.position.y = target_distance * math.sin(target_angle)
    marker.pose.position.z = 0.0
    
    marker.pose.orientation.w = 1.0
    
    # Sphere size (larger than disparities)
    marker.scale.x = 0.2
    marker.scale.y = 0.2
    marker.scale.z = 0.2
    
    # Color: bright green for target
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    
    marker.lifetime = rospy.Duration(0.1)
    
    target_point_pub.publish(marker)


def publish_processed_scan(original_scan, processed_ranges):
    """
    Publish the processed laser scan (with safety bubbles) for visualization
    """
    processed_scan = LaserScan()
    processed_scan.header = original_scan.header
    processed_scan.angle_min = original_scan.angle_min
    processed_scan.angle_max = original_scan.angle_max
    processed_scan.angle_increment = original_scan.angle_increment
    processed_scan.time_increment = original_scan.time_increment
    processed_scan.scan_time = original_scan.scan_time
    processed_scan.range_min = original_scan.range_min
    processed_scan.range_max = original_scan.range_max
    processed_scan.ranges = processed_ranges.tolist()
    
    processed_scan_pub.publish(processed_scan)


def lidar_callback(data):
    """
    Main callback function for processing LIDAR data and publishing drive commands
    """
    # Step 1: Preprocess LIDAR data
    ranges = preprocess_lidar(data.ranges, data.angle_min, data.angle_increment)
    
    # Step 2: Find disparities
    disparities = find_disparities(ranges, DISPARITY_THRESHOLD)
    
    # Step 3: Extend disparities (safety bubbles)
    processed_ranges = extend_disparities(ranges, data.angle_increment, disparities)
    
    # Step 4: Get indices for front-facing range (-90 to +90 degrees)
    start_idx, end_idx = get_angle_range_indices(data.angle_min, data.angle_increment, len(processed_ranges))
    
    # Step 5: Find the best gap
    target_idx, target_angle, max_distance = find_best_gap(
        processed_ranges, data.angle_min, data.angle_increment, start_idx, end_idx
    )
    
    # Step 6: Calculate steering angle
    steering_angle = calculate_steering_angle(target_angle)
    
    # Step 7: Calculate velocity
    velocity = calculate_velocity(steering_angle, max_distance)
    
    # Step 8: Publish AckermannDrive message
    command = AckermannDrive()
    command.steering_angle = steering_angle
    command.speed = velocity
    
    command_pub.publish(command)
    
    # Step 9: Publish visualizations for RViz
    publish_car_footprint()
    publish_steering_arrow(steering_angle)
    publish_disparities(disparities, ranges, data.angle_min, data.angle_increment)
    publish_target_point(target_angle, max_distance)
    publish_processed_scan(data, processed_ranges)
    
    # Debug output with detailed information
    rospy.loginfo_throttle(1.0, 
        "LIDAR: angle_min=%.2f angle_max=%.2f | Target: idx=%d angle=%.2f | Steering: %.2f | Vel: %.2f | Dist: %.2f m | Disp: %d" % 
        (math.degrees(data.angle_min), math.degrees(data.angle_max), 
         target_idx, math.degrees(target_angle), steering_angle, velocity, max_distance, len(disparities)))


if __name__ == '__main__':
    try:
        rospy.init_node('follow_the_gap', anonymous=True)
        rospy.loginfo("Follow the Gap node started")
        
        # Subscribe to LIDAR scan topic
        rospy.Subscriber("/car_2/scan", LaserScan, lidar_callback)
        
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Follow the Gap node terminated")
