#!/usr/bin/env python

import rospy
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDrive

# Configuration parameters
DISPARITY_THRESHOLD = 0.1  # Threshold for detecting disparities (meters)
CAR_WIDTH = 0.20  # Width of the car (meters)
SAFETY_MARGIN = 0.15  # Additional safety margin around the car (meters)
MAX_VELOCITY = 40.0  # Maximum velocity on straightaways
MIN_VELOCITY = 15.0  # Minimum velocity during turns
MAX_STEERING = 100.0  # Maximum steering angle
MIN_STEERING = -100.0  # Minimum steering angle

# Publisher for moving the car
command_pub = rospy.Publisher('/car_2/offboard/command', AckermannDrive, queue_size=1)


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
    """
    # Calculate start and end angles for the range we care about
    target_min_angle = math.radians(-90)
    target_max_angle = math.radians(90)
    
    # Find corresponding indices
    start_idx = int((target_min_angle - angle_min) / angle_increment)
    end_idx = int((target_max_angle - angle_min) / angle_increment)
    
    # Clamp to valid range
    start_idx = max(0, min(start_idx, num_ranges - 1))
    end_idx = max(0, min(end_idx, num_ranges - 1))
    
    return start_idx, end_idx


def find_best_gap(ranges, angle_min, angle_increment, start_idx, end_idx):
    """
    Find the best gap (farthest point or center of widest gap) in the specified range
    Returns the index and angle of the target point
    """
    # Search for the farthest point in the front-facing range
    max_dist = 0.0
    max_idx = start_idx
    
    for i in range(start_idx, end_idx + 1):
        if ranges[i] > max_dist:
            max_dist = ranges[i]
            max_idx = i
    
    # Alternative: Find the widest gap (optional enhancement)
    # You can implement gap width detection here if desired
    
    # Calculate the angle corresponding to this index
    target_angle = angle_min + (max_idx * angle_increment)
    
    return max_idx, target_angle, max_dist


def calculate_steering_angle(target_angle_rad):
    """
    Convert target angle (in radians) to steering command [-100, 100]
    """
    # Convert radians to degrees for easier mapping
    target_angle_deg = math.degrees(target_angle_rad)
    
    # Map angle to steering range
    # Assuming reasonable steering angles: -45 to +45 degrees maps to -100 to +100
    steering_angle = (target_angle_deg / 45.0) * 100.0
    
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
    
    # Debug output
    rospy.loginfo_throttle(1.0, 
        "Target angle: %.2f deg, Steering: %.2f, Velocity: %.2f, Max distance: %.2f m" % 
        (math.degrees(target_angle), steering_angle, velocity, max_distance))


if __name__ == '__main__':
    try:
        rospy.init_node('follow_the_gap', anonymous=True)
        rospy.loginfo("Follow the Gap node started")
        
        # Subscribe to LIDAR scan topic
        rospy.Subscriber("/car_2/scan", LaserScan, lidar_callback)
        
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Follow the Gap node terminated")
