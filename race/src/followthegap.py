#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDrive
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class FollowTheGapNode:
    def __init__(self):
        # Car parameters
        self.CAR_WIDTH = 0.20  # meters (Traxxas Rally width)
        self.SAFETY_MARGIN = 0.15  # Additional safety margin on each side
        self.DISPARITY_THRESHOLD = 0.5  # meters - threshold to detect disparities
        
        # Speed parameters
        self.MAX_SPEED = 5.0  # Maximum speed in m/s
        self.MIN_SPEED = 1.5  # Minimum speed in m/s
        
        # Angle range for consideration (forward-facing only: -90 to +90 degrees)
        self.ANGLE_RANGE_MIN = -90.0  # degrees
        self.ANGLE_RANGE_MAX = 90.0   # degrees
        
        # LIDAR parameters (will be updated from scan message)
        self.angle_min = None
        self.angle_max = None
        self.angle_increment = None
        self.range_min = None
        self.range_max = None
        
        # Publisher for drive commands
        self.drive_pub = rospy.Publisher('/car_2/offboard/command', AckermannDrive, queue_size=1)
        
        # Publishers for visualization
        self.marker_pub = rospy.Publisher('/followthegap_markers', MarkerArray, queue_size=1)
        self.car_footprint_pub = rospy.Publisher('/car_footprint', Marker, queue_size=1)
        self.steering_arrow_pub = rospy.Publisher('/steering_arrow', Marker, queue_size=1)
        self.laserscan_viz_pub = rospy.Publisher('/laserscan_viz', Marker, queue_size=1)
        
        # Subscriber to LIDAR scan
        rospy.Subscriber('/car_2/scan', LaserScan, self.lidar_callback)
        
        rospy.loginfo("Follow the Gap node initialized")
        rospy.loginfo("Car width: {}m, Safety margin: {}m".format(self.CAR_WIDTH, self.SAFETY_MARGIN))
        rospy.loginfo("Disparity threshold: {}m".format(self.DISPARITY_THRESHOLD))
    
    def preprocess_lidar(self, ranges):
        """
        Preprocess LIDAR data: handle NaN, inf, and out-of-range values
        """
        processed_ranges = np.array(ranges, dtype=float)
        
        # Replace NaN and inf with max range
        processed_ranges[np.isnan(processed_ranges)] = self.range_max
        processed_ranges[np.isinf(processed_ranges)] = self.range_max
        
        # Clip to valid range
        processed_ranges = np.clip(processed_ranges, self.range_min, self.range_max)
        
        return processed_ranges
    
    def find_disparities(self, ranges):
        """
        Find disparities in LIDAR readings.
        Returns list of tuples: (index, closer_distance, farther_distance)
        """
        disparities = []
        
        for i in range(len(ranges) - 1):
            dist_diff = abs(ranges[i] - ranges[i + 1])
            
            if dist_diff > self.DISPARITY_THRESHOLD:
                closer_dist = min(ranges[i], ranges[i + 1])
                farther_dist = max(ranges[i], ranges[i + 1])
                
                # Store the index of the pair and which one is closer
                if ranges[i] < ranges[i + 1]:
                    disparities.append((i, i + 1, closer_dist, farther_dist))
                else:
                    disparities.append((i + 1, i, closer_dist, farther_dist))
        
        return disparities
    
    def calculate_bubble_radius(self, distance):
        """
        Calculate the number of LIDAR samples needed to cover
        half the car width plus safety margin at a given distance.
        """
        # Total width to protect (half car + safety margin)
        protected_width = (self.CAR_WIDTH / 2.0) + self.SAFETY_MARGIN
        
        # Calculate angle subtended by this width at the given distance
        # angle = arctan(width / distance)
        if distance < 0.01:  # Avoid division by zero
            return 100  # Large number of samples
        
        angle_rad = math.atan(protected_width / distance)
        
        # Convert to number of samples
        num_samples = int(angle_rad / self.angle_increment)
        
        return max(1, num_samples)  # At least 1 sample
    
    def extend_disparities(self, ranges, disparities):
        """
        Extend disparities using the disparity extender approach.
        """
        extended_ranges = ranges.copy()
        
        for closer_idx, farther_idx, closer_dist, farther_dist in disparities:
            # Calculate bubble radius based on closer distance
            bubble_radius = self.calculate_bubble_radius(closer_dist)
            
            # Determine direction to extend (away from closer point, starting at farther point)
            if closer_idx < farther_idx:
                # Extend to the right (increasing indices)
                start_idx = farther_idx
                direction = 1
            else:
                # Extend to the left (decreasing indices)
                start_idx = farther_idx
                direction = -1
            
            # Extend the disparity
            for j in range(bubble_radius):
                idx = start_idx + (j * direction)
                
                # Check bounds
                if idx < 0 or idx >= len(extended_ranges):
                    break
                
                # Only overwrite if current value is farther (never overwrite closer distances)
                if extended_ranges[idx] > closer_dist:
                    extended_ranges[idx] = closer_dist
        
        return extended_ranges
    
    def find_best_gap(self, ranges):
        """
        Find the best gap (direction with farthest reachable distance)
        within the forward-facing angle range (-90 to +90 degrees).
        """
        # Convert angle range to indices
        # angle = angle_min + index * angle_increment
        # index = (angle - angle_min) / angle_increment
        
        min_angle_rad = math.radians(self.ANGLE_RANGE_MIN)
        max_angle_rad = math.radians(self.ANGLE_RANGE_MAX)
        
        # Calculate index range
        min_idx = int((min_angle_rad - self.angle_min) / self.angle_increment)
        max_idx = int((max_angle_rad - self.angle_min) / self.angle_increment)
        
        # Clip to valid range
        min_idx = max(0, min_idx)
        max_idx = min(len(ranges) - 1, max_idx)
        
        # Find the index with maximum distance in the forward range
        max_dist = 0.0
        max_idx_found = (min_idx + max_idx) // 2  # Default to center
        
        for i in range(min_idx, max_idx + 1):
            if ranges[i] > max_dist:
                max_dist = ranges[i]
                max_idx_found = i
        
        # Convert index to angle
        target_angle = self.angle_min + (max_idx_found * self.angle_increment)
        
        return target_angle, max_dist, max_idx_found
    
    def calculate_steering_and_speed(self, target_angle, max_distance):
        """
        Calculate steering angle and speed based on target direction.
        """
        # Convert target angle from radians to steering angle
        # Steering angle in radians (positive = left, negative = right)
        steering_angle = target_angle
        
        # Limit steering angle
        max_steering = math.radians(30.0)  # 30 degrees max
        steering_angle = np.clip(steering_angle, -max_steering, max_steering)
        
        # Calculate speed based on steering angle and distance
        # Reduce speed for sharp turns
        steering_factor = 1.0 - (abs(steering_angle) / max_steering) * 0.5
        distance_factor = min(max_distance / 5.0, 1.0)  # Scale based on distance
        
        speed = self.MIN_SPEED + (self.MAX_SPEED - self.MIN_SPEED) * steering_factor * distance_factor
        
        return steering_angle, speed
    
    def publish_car_footprint(self):
        """
        Publish car footprint visualization.
        """
        marker = Marker()
        marker.header.frame_id = "car_2_laser"
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
        half_length = 0.50 / 2.0  # Car length
        half_width = self.CAR_WIDTH / 2.0
        
        # Front-right corner
        p1 = Point()
        p1.x = half_length
        p1.y = -half_width
        p1.z = 0.0
        
        # Front-left corner
        p2 = Point()
        p2.x = half_length
        p2.y = half_width
        p2.z = 0.0
        
        # Rear-left corner
        p3 = Point()
        p3.x = -half_length
        p3.y = half_width
        p3.z = 0.0
        
        # Rear-right corner
        p4 = Point()
        p4.x = -half_length
        p4.y = -half_width
        p4.z = 0.0
        
        # Create closed rectangle
        marker.points.append(p1)
        marker.points.append(p2)
        marker.points.append(p3)
        marker.points.append(p4)
        marker.points.append(p1)
        
        marker.lifetime = rospy.Duration(0)
        
        self.car_footprint_pub.publish(marker)
    
    def publish_steering_arrow(self, steering_angle):
        """
        Publish steering direction arrow.
        """
        marker = Marker()
        marker.header.frame_id = "car_2_laser"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "steering_arrow"
        marker.id = 1
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Arrow dimensions
        marker.scale.x = 0.05   # Shaft diameter
        marker.scale.y = 0.08   # Head diameter
        marker.scale.z = 0.08   # Head length
        
        # Set color based on steering direction
        steering_normalized = abs(steering_angle) / math.radians(30.0)
        marker.color.r = steering_normalized
        marker.color.g = 1.0 - steering_normalized
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        # Start point (at front of car)
        start = Point()
        start.x = 0.25  # Front of car
        start.y = 0.0
        start.z = 0.1
        
        # End point (steering direction)
        arrow_length = 0.5
        end = Point()
        end.x = start.x + arrow_length * math.cos(steering_angle)
        end.y = start.y + arrow_length * math.sin(steering_angle)
        end.z = start.z
        
        marker.points.append(start)
        marker.points.append(end)
        
        marker.lifetime = rospy.Duration(0.1)
        
        self.steering_arrow_pub.publish(marker)
    
    def publish_laserscan_viz(self, ranges):
        """
        Publish LaserScan visualization as points.
        """
        marker = Marker()
        marker.header.frame_id = "car_2_laser"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "laserscan_points"
        marker.id = 2
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        
        marker.scale.x = 0.03  # Point width
        marker.scale.y = 0.03  # Point height
        
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.5
        
        # Add all scan points
        for i, distance in enumerate(ranges):
            if distance < self.range_max and distance > self.range_min:
                angle = self.angle_min + i * self.angle_increment
                
                p = Point()
                p.x = distance * math.cos(angle)
                p.y = distance * math.sin(angle)
                p.z = 0.0
                
                marker.points.append(p)
        
        marker.lifetime = rospy.Duration(0.1)
        
        self.laserscan_viz_pub.publish(marker)
    
    def publish_visualization(self, ranges, extended_ranges, disparities, target_idx, steering_angle):
        """
        Publish all visualization markers for RViz.
        """
        marker_array = MarkerArray()
        timestamp = rospy.Time.now()
        
        # 1. Target point (spherical marker at the selected gap)
        target_marker = Marker()
        target_marker.header.frame_id = "car_2_laser"
        target_marker.header.stamp = timestamp
        target_marker.ns = "target_point"
        target_marker.id = 0
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        
        target_angle = self.angle_min + (target_idx * self.angle_increment)
        target_dist = extended_ranges[target_idx]
        
        target_marker.pose.position.x = target_dist * math.cos(target_angle)
        target_marker.pose.position.y = target_dist * math.sin(target_angle)
        target_marker.pose.position.z = 0.0
        target_marker.pose.orientation.w = 1.0
        
        target_marker.scale.x = 0.2
        target_marker.scale.y = 0.2
        target_marker.scale.z = 0.2
        
        target_marker.color.r = 0.0
        target_marker.color.g = 1.0
        target_marker.color.b = 0.0
        target_marker.color.a = 1.0
        
        target_marker.lifetime = rospy.Duration(0.1)
        marker_array.markers.append(target_marker)
        
        # 2. Target direction arrow
        arrow_marker = Marker()
        arrow_marker.header.frame_id = "car_2_laser"
        arrow_marker.header.stamp = timestamp
        arrow_marker.ns = "target_arrow"
        arrow_marker.id = 1
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        
        start = Point()
        start.x = 0.0
        start.y = 0.0
        start.z = 0.0
        
        end = Point()
        end.x = target_dist * math.cos(target_angle)
        end.y = target_dist * math.sin(target_angle)
        end.z = 0.0
        
        arrow_marker.points.append(start)
        arrow_marker.points.append(end)
        
        arrow_marker.scale.x = 0.05  # Arrow shaft diameter
        arrow_marker.scale.y = 0.1   # Arrow head diameter
        arrow_marker.scale.z = 0.1   # Arrow head length
        
        arrow_marker.color.r = 0.0
        arrow_marker.color.g = 1.0
        arrow_marker.color.b = 0.0
        arrow_marker.color.a = 0.8
        
        arrow_marker.lifetime = rospy.Duration(0.1)
        marker_array.markers.append(arrow_marker)
        
        # 3. Disparity markers (show each disparity location)
        for i, (closer_idx, farther_idx, closer_dist, farther_dist) in enumerate(disparities):
            # Mark the closer point
            closer_marker = Marker()
            closer_marker.header.frame_id = "car_2_laser"
            closer_marker.header.stamp = timestamp
            closer_marker.ns = "disparities"
            closer_marker.id = 100 + i * 2
            closer_marker.type = Marker.SPHERE
            closer_marker.action = Marker.ADD
            
            closer_angle = self.angle_min + (closer_idx * self.angle_increment)
            
            closer_marker.pose.position.x = closer_dist * math.cos(closer_angle)
            closer_marker.pose.position.y = closer_dist * math.sin(closer_angle)
            closer_marker.pose.position.z = 0.0
            closer_marker.pose.orientation.w = 1.0
            
            closer_marker.scale.x = 0.1
            closer_marker.scale.y = 0.1
            closer_marker.scale.z = 0.1
            
            closer_marker.color.r = 1.0
            closer_marker.color.g = 0.0
            closer_marker.color.b = 0.0
            closer_marker.color.a = 0.8
            
            closer_marker.lifetime = rospy.Duration(0.1)
            marker_array.markers.append(closer_marker)
            
            # Mark the farther point
            farther_marker = Marker()
            farther_marker.header.frame_id = "car_2_laser"
            farther_marker.header.stamp = timestamp
            farther_marker.ns = "disparities"
            farther_marker.id = 100 + i * 2 + 1
            farther_marker.type = Marker.SPHERE
            farther_marker.action = Marker.ADD
            
            farther_angle = self.angle_min + (farther_idx * self.angle_increment)
            
            farther_marker.pose.position.x = farther_dist * math.cos(farther_angle)
            farther_marker.pose.position.y = farther_dist * math.sin(farther_angle)
            farther_marker.pose.position.z = 0.0
            farther_marker.pose.orientation.w = 1.0
            
            farther_marker.scale.x = 0.1
            farther_marker.scale.y = 0.1
            farther_marker.scale.z = 0.1
            
            farther_marker.color.r = 1.0
            farther_marker.color.g = 0.5
            farther_marker.color.b = 0.0
            farther_marker.color.a = 0.8
            
            farther_marker.lifetime = rospy.Duration(0.1)
            marker_array.markers.append(farther_marker)
            
            # Draw a line between disparity points
            line_marker = Marker()
            line_marker.header.frame_id = "car_2_laser"
            line_marker.header.stamp = timestamp
            line_marker.ns = "disparity_lines"
            line_marker.id = 200 + i
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            
            p1 = Point()
            p1.x = closer_dist * math.cos(closer_angle)
            p1.y = closer_dist * math.sin(closer_angle)
            p1.z = 0.0
            
            p2 = Point()
            p2.x = farther_dist * math.cos(farther_angle)
            p2.y = farther_dist * math.sin(farther_angle)
            p2.z = 0.0
            
            line_marker.points.append(p1)
            line_marker.points.append(p2)
            
            line_marker.scale.x = 0.02
            
            line_marker.color.r = 1.0
            line_marker.color.g = 0.0
            line_marker.color.b = 0.0
            line_marker.color.a = 0.6
            
            line_marker.lifetime = rospy.Duration(0.1)
            marker_array.markers.append(line_marker)
        
        self.marker_pub.publish(marker_array)
    
    def lidar_callback(self, data):
        """
        Main callback function for LIDAR data processing.
        Implements the Follow the Gap algorithm with disparity extender.
        """
        # Update LIDAR parameters from scan message
        if self.angle_min is None:
            self.angle_min = data.angle_min
            self.angle_max = data.angle_max
            self.angle_increment = data.angle_increment
            self.range_min = data.range_min
            self.range_max = data.range_max
            rospy.loginfo("LIDAR parameters initialized: angle_min={:.1f}deg, angle_max={:.1f}deg, increment={:.3f}deg".format(
                         math.degrees(self.angle_min), math.degrees(self.angle_max), math.degrees(self.angle_increment)))
        
        # Step 1: Preprocess LIDAR data
        ranges = self.preprocess_lidar(data.ranges)
        
        # Step 2: Find disparities
        disparities = self.find_disparities(ranges)
        rospy.loginfo_throttle(1.0, "Found {} disparities".format(len(disparities)))
        
        # Step 3: Extend disparities
        extended_ranges = self.extend_disparities(ranges, disparities)
        
        # Step 4: Find the best gap (target direction)
        target_angle, max_distance, target_idx = self.find_best_gap(extended_ranges)
        
        # Step 5: Calculate steering and speed
        steering_angle, speed = self.calculate_steering_and_speed(target_angle, max_distance)
        
        # Log information
        rospy.loginfo_throttle(1.0, "Target angle: {:.1f}deg, Distance: {:.2f}m, Steering: {:.1f}deg, Speed: {:.2f}m/s".format(
                              math.degrees(target_angle), max_distance, math.degrees(steering_angle), speed))
        
        # Publish drive command
        drive_msg = AckermannDrive()
        drive_msg.steering_angle = steering_angle
        drive_msg.speed = speed
        self.drive_pub.publish(drive_msg)
        
        # Publish all visualizations
        self.publish_car_footprint()
        self.publish_steering_arrow(steering_angle)
        self.publish_laserscan_viz(extended_ranges)
        self.publish_visualization(ranges, extended_ranges, disparities, target_idx, steering_angle)


def main():
    rospy.init_node('follow_the_gap', anonymous=True)
    rospy.loginfo("Follow the Gap node started (Disparity Extender)")
    _ = FollowTheGapNode()
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Follow the Gap node terminated")

