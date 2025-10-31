#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDrive
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# =========================
# Configuration parameters
# =========================

# LiDAR mounting / frame alignment:
# If your LaserScan has 0° ≈ straight ahead, leave this at 0.
# If your LaserScan has 0° ≈ to the right (and ~90° straight ahead), set this to 90.
LIDAR_ZERO_DEG_IN_CAR_FRAME = 0.0  # degrees

# If your car turns opposite of what you expect, flip this sign to -1.0
STEERING_SIGN = +1.0

# Disparity / bubble
DISPARITY_THRESHOLD = 0.10      # m
CAR_WIDTH = 0.20                # m
CAR_LENGTH = 0.50               # m
SAFETY_MARGIN = 0.15            # m
MIN_RANGE_OBSTACLE = 0.18       # m (anything closer than this is considered blocked)

# Speed / steering scaling (course-specific scale: -100..+100)
MAX_VELOCITY = 40.0             # straightaways
MIN_VELOCITY = 15.0             # tight turns
MAX_STEERING = 100.0
MIN_STEERING = -100.0

# Gap & target selection
MIN_FREE_DISTANCE = 0.50        # m (threshold to consider a ray as "free")

# Smoothing
ANGLE_SMOOTH_ALPHA = 0.5        # 0..1 (higher = smoother); applied to steering command

# Visualization rate
VIS_LIFETIME = 0.1              # seconds

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def bubble_half_angle(distance):
    """Numerically stable half-angle (radians) that covers (car half-width + margin)."""
    radius = (CAR_WIDTH * 0.5) + SAFETY_MARGIN
    if distance <= 1e-3:
        return math.radians(45.0)
    # atan is stable for all distances; for small d it approaches ~90 deg
    return math.atan(radius / max(distance, 1e-3))

def car_to_lidar_angle(a_car_rad):
    """Convert an angle in the car frame (0 = straight ahead, left positive) to LiDAR frame."""
    # LiDAR angle = car angle + (LiDAR's 0° offset measured in car frame) - 90°
    # Explanation: in many setups 90° lidar == front in your old assumption; here we generalize.
    return a_car_rad + math.radians(LIDAR_ZERO_DEG_IN_CAR_FRAME)

def lidar_to_car_angle(a_lidar_rad):
    """Convert an angle in the LiDAR frame to the car frame (0 = straight ahead, left positive)."""
    return a_lidar_rad - math.radians(LIDAR_ZERO_DEG_IN_CAR_FRAME)

class FollowTheGapNode(object):
    def __init__(self):
        # Publishers
        self.command_pub = rospy.Publisher('/car_2/offboard/command', AckermannDrive, queue_size=1)
        self.car_footprint_pub = rospy.Publisher('/follow_gap/car_footprint', Marker, queue_size=1)
        self.steering_arrow_pub = rospy.Publisher('/follow_gap/steering_arrow', Marker, queue_size=1)
        self.disparities_pub = rospy.Publisher('/follow_gap/disparities', MarkerArray, queue_size=1)
        self.target_point_pub = rospy.Publisher('/follow_gap/target_point', Marker, queue_size=1)
        self.processed_scan_pub = rospy.Publisher('/follow_gap/processed_scan', LaserScan, queue_size=1)

        self.prev_steering = 0.0  # for smoothing

        rospy.Subscriber("/car_2/scan", LaserScan, self.lidar_callback)

    # ---------- Preprocess ----------

    def preprocess_lidar(self, scan):
        ranges = np.array(scan.ranges, dtype=np.float32)
        # Replace NaN/Inf with a large value (treat as no obstacle)
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)
        # Treat anything too close as blocked (clip)
        ranges = np.where(ranges < MIN_RANGE_OBSTACLE, MIN_RANGE_OBSTACLE, ranges)
        return ranges

    # ---------- Disparity & bubble inflation ----------

    def find_disparities(self, ranges):
        diffs = np.abs(np.diff(ranges))
        return list(np.where(diffs > DISPARITY_THRESHOLD)[0])

    def apply_safety_bubble_to_nearest(self, ranges, angle_increment):
        """Step 1: Find nearest LIDAR point and put a safety bubble around it."""
        out = np.copy(ranges)
        
        # Find the nearest point (minimum distance)
        nearest_idx = int(np.argmin(ranges))
        nearest_dist = ranges[nearest_idx]
        
        # Calculate bubble radius in terms of angle
        half_ang = bubble_half_angle(nearest_dist)
        num_samples = int(half_ang / angle_increment)
        num_samples = clamp(num_samples, 0, 100)
        
        # Set all points within the bubble to 0 (blocked)
        for k in range(-num_samples, num_samples + 1):
            j = nearest_idx + k
            if 0 <= j < len(out):
                if out[j] <= nearest_dist + SAFETY_MARGIN:
                    out[j] = 0.0
        
        return out

    def extend_disparities(self, ranges, angle_increment, disparities):
        """Step 2-3: For each disparity, extend safety bubble from closer point."""
        out = np.copy(ranges)
        for idx in disparities:
            d1, d2 = ranges[idx], ranges[idx + 1]
            # Identify closer vs farther point
            if d1 <= d2:
                closer_dist, closer_idx, farther_idx, direction = d1, idx, idx + 1, +1
            else:
                closer_dist, closer_idx, farther_idx, direction = d2, idx + 1, idx, -1

            # Calculate number of samples to cover car width + margin at closer distance
            half_ang = bubble_half_angle(closer_dist)
            num = int(half_ang / angle_increment)
            num = clamp(num, 0, 60)  # cap to avoid wiping out the scan

            # Starting at the farther point, extend toward closer point's distance
            # Do not overwrite points that are already closer!
            for k in range(num + 1):
                j = farther_idx + direction * k
                if j < 0 or j >= len(out):
                    break
                if out[j] > closer_dist:
                    out[j] = closer_dist
        return out

    # ---------- Window selection (front hemisphere in CAR frame) ----------

    def front_window_indices(self, angle_min, angle_increment, n):
        # Car frame front window: [-90°, +90°]
        car_min = math.radians(-90.0)
        car_max = math.radians(+90.0)
        # Convert those to LiDAR frame
        lid_min = car_to_lidar_angle(car_min)
        lid_max = car_to_lidar_angle(car_max)

        i0 = int((lid_min - angle_min) / angle_increment)
        i1 = int((lid_max - angle_min) / angle_increment)
        i0 = clamp(i0, 0, n - 1)
        i1 = clamp(i1, 0, n - 1)
        return (i0, i1) if i0 <= i1 else (i1, i0)

    # ---------- Gap finding ----------

    def find_best_gap(self, ranges, angle_min, angle_increment, start_idx, end_idx):
        """
        Step 3: Find maximum length sequence of consecutive non-zeros (free space).
        Step 4: Choose the furthest point in the selected gap.
        """
        # Identify contiguous "free" sequences (non-zero points after bubble processing)
        gaps = []
        in_gap = False
        gap_start = start_idx

        for i in range(start_idx, end_idx + 1):
            # Consider point "free" if it's above minimum threshold (treated as non-zero)
            if ranges[i] > MIN_FREE_DISTANCE:
                if not in_gap:
                    in_gap = True
                    gap_start = i
            else:
                if in_gap:
                    gaps.append((gap_start, i - 1))
                    in_gap = False
        if in_gap:
            gaps.append((gap_start, end_idx))

        # No gap → fallback to farthest point in range
        if not gaps:
            max_idx = start_idx + int(np.argmax(ranges[start_idx:end_idx + 1]))
            max_dist = ranges[max_idx]
            target_angle_lidar = angle_min + max_idx * angle_increment
            return (max_idx, max_idx, target_angle_lidar, max_dist)

        # Find the best gap based on algorithm requirements
        # Option 1: Maximum length gap (longest consecutive free space)
        # Option 2: Deepest gap (gap with furthest point)
        # Let's try both and allow configuration
        
        best_gap = None
        best_score = -1.0
        
        for gs, ge in gaps:
            width = ge - gs + 1
            
            # Find deepest point in this gap
            sub = ranges[gs:ge + 1]
            local_max_off = int(np.argmax(sub))
            deepest_idx = gs + local_max_off
            deepest_dist = ranges[deepest_idx]
            
            # Scoring strategy (can be adjusted):
            # Prioritize deepest gaps but also consider width
            # This balances between "widest gap" and "deepest gap"
            score = deepest_dist * 2.0 + width * 0.1
            
            if score > best_score:
                best_score = score
                best_gap = (gs, ge, deepest_idx, deepest_dist)
        
        gs, ge, deepest_idx, deepest_dist = best_gap
        
        # Step 4: Target the FURTHEST point in the selected gap (the deepest point)
        # This follows the algorithm: "Choose the furthest point in free space"
        target_idx = deepest_idx
        target_angle_lidar = angle_min + target_idx * angle_increment
        
        return (gs, ge, target_angle_lidar, ranges[target_idx])

    # ---------- Control ----------

    def calculate_steering(self, target_angle_lidar):
        # Convert to car frame: 0 rad = forward, left positive
        a_car = lidar_to_car_angle(target_angle_lidar)
        # Map angle deviation (deg) to [-100, 100] with ~45° → 100
        dev_deg = math.degrees(a_car)
        steering_cmd = (dev_deg / 45.0) * 100.0
        steering_cmd = clamp(steering_cmd, MIN_STEERING, MAX_STEERING)
        steering_cmd *= STEERING_SIGN
        # Smooth it
        steering_cmd = ANGLE_SMOOTH_ALPHA * self.prev_steering + (1.0 - ANGLE_SMOOTH_ALPHA) * steering_cmd
        self.prev_steering = steering_cmd
        return steering_cmd

    def calculate_velocity(self, steering_cmd, max_distance_ahead):
        steering_mag = abs(steering_cmd) / MAX_STEERING
        v = MIN_VELOCITY + (MAX_VELOCITY - MIN_VELOCITY) * (1.0 - steering_mag)
        if max_distance_ahead < 1.0:
            v *= 0.5
        elif max_distance_ahead < 2.0:
            v *= 0.75
        return clamp(v, 0.0, 100.0)

    # ---------- Visualization ----------

    def publish_car_footprint(self):
        m = Marker()
        m.header.frame_id = "car_2_base_link"
        m.header.stamp = rospy.Time.now()
        m.ns = "car_footprint"
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.05
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.0, 1.0, 1.0

        hl = CAR_LENGTH * 0.5
        hw = CAR_WIDTH * 0.5
        p1 = Point( hl, -hw, 0.0)
        p2 = Point( hl,  hw, 0.0)
        p3 = Point(-hl,  hw, 0.0)
        p4 = Point(-hl, -hw, 0.0)
        m.points = [p1, p2, p3, p4, p1]
        m.lifetime = rospy.Duration(0)
        self.car_footprint_pub.publish(m)

    def publish_steering_arrow(self, steering_cmd):
        m = Marker()
        m.header.frame_id = "car_2_base_link"
        m.header.stamp = rospy.Time.now()
        m.ns = "steering_arrow"
        m.id = 1
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.scale.x = 0.05  # shaft diameter
        m.scale.y = 0.08  # head diameter
        m.scale.z = 0.10  # head length

        mag = abs(steering_cmd) / 100.0
        m.color.r = mag
        m.color.g = 1.0 - mag
        m.color.b = 0.0
        m.color.a = 1.0

        # Map [-100,100] to about [-45°, +45°]
        angle_rad = (steering_cmd / 100.0) * (math.pi / 4.0)
        # Arrow starts near the front
        start = Point(CAR_LENGTH * 0.5, 0.0, 0.1)
        L = 0.5
        end = Point()
        end.x = start.x + L * math.cos(angle_rad)
        end.y = start.y + L * math.sin(angle_rad)
        end.z = start.z
        m.points = [start, end]
        m.lifetime = rospy.Duration(VIS_LIFETIME)
        self.steering_arrow_pub.publish(m)

    def publish_disparities(self, disparities, ranges, angle_min, angle_increment):
        arr = MarkerArray()
        t = rospy.Time.now()
        for i, idx in enumerate(disparities):
            m = Marker()
            m.header.frame_id = "car_2_laser"
            m.header.stamp = t
            m.ns = "disparities"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            ang = angle_min + idx * angle_increment
            dist = ranges[idx]
            m.pose.position.x = dist * math.cos(ang)
            m.pose.position.y = dist * math.sin(ang)
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.10
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 0.8
            m.lifetime = rospy.Duration(VIS_LIFETIME)
            arr.markers.append(m)
        self.disparities_pub.publish(arr)

    def publish_target_point(self, target_angle_lidar, target_distance):
        m = Marker()
        m.header.frame_id = "car_2_laser"
        m.header.stamp = rospy.Time.now()
        m.ns = "target_point"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = target_distance * math.cos(target_angle_lidar)
        m.pose.position.y = target_distance * math.sin(target_angle_lidar)
        m.pose.position.z = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.20
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 1.0
        m.lifetime = rospy.Duration(VIS_LIFETIME)
        self.target_point_pub.publish(m)

    def publish_processed_scan(self, original_scan, processed_ranges):
        out = LaserScan()
        # Use the original header as a base but ensure stamp and frame_id are valid for RViz.
        out.header = original_scan.header
        # If the incoming scan has an empty frame_id (some drivers do), set a sane default
        if not out.header.frame_id:
            out.header.frame_id = "car_2_laser"
        # Use current time to avoid RViz ignoring an old/zero timestamp
        out.header.stamp = rospy.Time.now()
        out.angle_min = original_scan.angle_min
        out.angle_max = original_scan.angle_max
        out.angle_increment = original_scan.angle_increment
        out.time_increment = original_scan.time_increment
        out.scan_time = original_scan.scan_time
        out.range_min = original_scan.range_min
        out.range_max = original_scan.range_max
        # Ensure ranges are native Python floats (not numpy types)
        out.ranges = [float(x) for x in processed_ranges.tolist()]
        self.processed_scan_pub.publish(out)

    # ---------- Callback ----------

    def lidar_callback(self, scan):
        # ALGORITHM IMPLEMENTATION:
        
        # Step 0: Preprocess raw LIDAR data
        ranges = self.preprocess_lidar(scan)

        # Step 1: Find nearest LIDAR point and put safety bubble around it
        ranges = self.apply_safety_bubble_to_nearest(ranges, scan.angle_increment)

        # Step 2: Find disparities (large jumps in distance)
        disparities = self.find_disparities(ranges)

        # Step 3: Extend safety bubbles around disparities
        # Starting at farther point, overwrite with closer distance (don't overwrite already closer points)
        processed = self.extend_disparities(ranges, scan.angle_increment, disparities)

        # Step 4: Search through filtered distances in front hemisphere (-90° to +90° in car frame)
        i0, i1 = self.front_window_indices(scan.angle_min, scan.angle_increment, len(processed))

        # Step 5: Find the maximum gap (longest sequence of free space points)
        # Step 6: Choose the furthest point in that gap as target
        gs, ge, target_angle_lidar, max_dist = self.find_best_gap(
            processed, scan.angle_min, scan.angle_increment, i0, i1
        )

        # Step 7: Calculate steering angle toward target point
        steering_cmd = self.calculate_steering(target_angle_lidar)
        velocity_cmd = self.calculate_velocity(steering_cmd, max_dist)

        # Step 8: Publish AckermannDrive message with steering [-100,100] and velocity [0,100]
        cmd = AckermannDrive()
        cmd.steering_angle = steering_cmd  # course-specific interface expects [-100,100]
        cmd.speed = velocity_cmd
        self.command_pub.publish(cmd)

        # 9) Visualizations
        self.publish_car_footprint()
        self.publish_steering_arrow(steering_cmd)
        self.publish_disparities(disparities, ranges, scan.angle_min, scan.angle_increment)
        self.publish_target_point(target_angle_lidar, max_dist)
        self.publish_processed_scan(scan, processed)

        # Debug
        rospy.loginfo_throttle(
            1.0,
            "Gap: [%d,%d] | Target(lidar)=%.1f° | Steering=%.1f | Speed=%.1f | Dist=%.2f m",
            gs, ge,
            math.degrees(target_angle_lidar),
            steering_cmd,
            velocity_cmd,
            max_dist
        )

def main():
    rospy.init_node('follow_the_gap', anonymous=True)
    rospy.loginfo("Follow the Gap node started (frame-safe)")
    _ = FollowTheGapNode()
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Follow the Gap node terminated")

