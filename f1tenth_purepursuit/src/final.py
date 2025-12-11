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

def check_for_obstacles(lidar_data, position, range, angle_min, angle_increment):
    # print("\n\n\n")
    # print(lidar_data)
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

    # print(lidar_data[i0:i1])
    # print(i0, i1, len(lidar_data), angle_min, angle_increment)
    # print(position, i0, i1, len(lidar_data), angle_min)

    print(np.sum(lidar_data[i0:i1] < range))
    return np.sum(lidar_data[i0:i1] < range) > 90


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