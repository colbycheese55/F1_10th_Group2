#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
import numpy as np

from ackermann_msgs.msg import AckermannDrive
from sensor_msgs.msg import LaserScan

rospy.init_node('final_race', anonymous = True)

# import followthegap2
import pure_pursuit2
import shared


LIDAR_SMOOTH_WINDOW = 7
MIN_RANGE_OBSTACLE = 0.5
OBSTACLE_DIST_FRONT = 2.0
OBSTACLE_DIST_SIDE = 0.4



def smooth_lidar(ranges, window_size):
    if window_size <= 1:
        return ranges
    
    # Create a normalized averaging kernel
    kernel = np.ones(window_size) / window_size
    
    # Apply convolution with 'same' mode to maintain array size
    # Use 'same' mode to keep the same length as input
    smoothed = np.convolve(ranges, kernel, mode='same')
    
    return smoothed

def preprocess_lidar(scan):
    ranges = np.array(scan.ranges, dtype=np.float32)
    # Replace NaN/Inf with a large value (treat as no obstacle)
    ranges = np.where(np.isfinite(ranges), ranges, 10.0)
    # Treat anything too close as blocked (clip)
    ranges = np.where(ranges < MIN_RANGE_OBSTACLE, 10, ranges)
    
    # Apply moving average smoothing filter
    if LIDAR_SMOOTH_WINDOW > 1:
        ranges = smooth_lidar(ranges, LIDAR_SMOOTH_WINDOW)
    
    return ranges

def check_for_obstacles(lidar_data, position, distance_threshold, angle_min, angle_increment):
    """
    Detect car-sized obstacles (not walls) by checking for consecutive detections.
    A car is typically 0.3-0.5m wide, while walls are much wider.
    """
    angle_min = math.radians(-30)
    if position == "right":
        i0 = int(-angle_min / angle_increment)
        i1 = int((math.radians(90) - angle_min) / angle_increment)
    elif position == "front":
        i0 = int((math.radians(60) - angle_min) / angle_increment)
        i1 = int((math.radians(120) - angle_min) / angle_increment)
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
    
    # Calculate angular width of each group and filter for car-sized objects
    # A car at 2m distance with 0.5m width spans ~14 degrees (~0.24 rad)
    # At closer distances, it spans more degrees
    # We'll accept groups between 5 and 60 degrees (car-sized, not wall-sized)
    CAR_MIN_ANGLE_DEG = 5   # Minimum angular width for a car
    CAR_MAX_ANGLE_DEG = 20  # Maximum angular width (walls are typically wider)
    
    for group_length in groups:
        angle_width_deg = math.degrees(group_length * angle_increment)
        
        # If we find a group that's car-sized, return True
        if CAR_MIN_ANGLE_DEG <= angle_width_deg <= CAR_MAX_ANGLE_DEG:
            print(f"Car detected: {angle_width_deg:.1f} degrees wide, {group_length} indices")
            return True
    
    # All groups are either too small (noise) or too wide (walls)
    print(f"Only walls/noise detected. Group widths: {[math.degrees(g * angle_increment) for g in groups]}")
    return False


def lidar_callback(data):
    ranges = preprocess_lidar(data)
    angle_increment = data.angle_increment

    shared.opponent_detected = check_for_obstacles(ranges, "front", OBSTACLE_DIST_FRONT, data.angle_min, angle_increment)

def main():
    try:
        rospy.loginfo("Final race node started.")
        rospy.Subscriber("/car_2/scan", LaserScan, lidar_callback)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass




if __name__ == '__main__':
    main()