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
LIDAR_ZERO_DEG_IN_CAR_FRAME = 00.0  # degrees

# If your car turns opposite of what you expect, flip this sign to -1.0
STEERING_SIGN = +1.0

# Disparity / bubble
DISPARITY_THRESHOLD = 0.20      # m (0.1-0.2m is good starting point)
CAR_WIDTH = 0.31                # m
CAR_LENGTH = 0.50               # m
SAFETY_MARGIN = 0.05            # m (tolerance around car half-width)
MIN_RANGE_OBSTACLE = 0.18       # m (anything closer than this is considered blocked)

# Speed / steering scaling (course-specific scale: -100..+100)
MAX_VELOCITY = 20.0             # straightaways
MIN_VELOCITY = 10.0             # tight turns
MAX_STEERING = 100.0
MIN_STEERING = -100.0

# Gap & target selection
MIN_FREE_DISTANCE = 2.0        # m (threshold to consider a ray as "free")
CENTER_BIAS = 0.15              # 0..1 (0=center of gap; 1=deepest point). Use a mix.
                                 # We'll aim at weighted center: (1-CENTER_BIAS)*gap_center + CENTER_BIAS*deepest

# Speed / steering scaling
STEERING_GAIN = 2.2             # Multiplier for steering response

# Smoothing
ANGLE_SMOOTH_ALPHA = 0.5        # 0..1 (higher = smoother); applied to steering command
VELOCITY_SMOOTH_ALPHA = 0.5     # 0..1 (higher = smoother); applied to velocity command
LIDAR_SMOOTH_WINDOW = 5         # Window size for moving average filter (odd number recommended)

# Cornering safety
SIDE_SAFETY_DISTANCE = 0.2     # m (minimum safe distance for side/rear obstacles when cornering)

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

        self.prev_velocity = 0.0  # for velocity smoothing

        rospy.Subscriber("/car_2/scan", LaserScan, self.lidar_callback)

    # ---------- Preprocess ----------

    def preprocess_lidar(self, scan):
        ranges = np.array(scan.ranges, dtype=np.float32)
        # Replace NaN/Inf with a large value (treat as no obstacle)
        ranges = np.where(np.isfinite(ranges), ranges, 10.0)
        # Treat anything too close as blocked (clip)
        ranges = np.where(ranges < MIN_RANGE_OBSTACLE, MIN_RANGE_OBSTACLE, ranges)
        
        # Apply moving average smoothing filter
        if LIDAR_SMOOTH_WINDOW > 1:
            ranges = self.smooth_lidar(ranges, LIDAR_SMOOTH_WINDOW)
        
        return ranges
    
    def smooth_lidar(self, ranges, window_size):
        """
        Apply a moving average filter to smooth lidar data.
        Uses numpy convolution for efficiency.
        
        Args:
            ranges: numpy array of lidar ranges
            window_size: size of the smoothing window (odd number recommended)
        
        Returns:
            Smoothed numpy array of the same size
        """
        if window_size <= 1:
            return ranges
        
        # Create a normalized averaging kernel
        kernel = np.ones(window_size) / window_size
        
        # Apply convolution with 'same' mode to maintain array size
        # Use 'same' mode to keep the same length as input
        smoothed = np.convolve(ranges, kernel, mode='same')
        
        return smoothed

    # ---------- Disparity & bubble inflation ----------

    def find_disparities(self, ranges):
        """Step 1: Find disparities - two subsequent points that differ by > threshold."""
        diffs = np.abs(np.diff(ranges))
        return list(np.where(diffs > DISPARITY_THRESHOLD)[0])

    def extend_disparities(self, ranges, angle_increment, disparities):
        """
        Steps 2-4: Disparity Extender Algorithm
        
        For each disparity:
        2. Pick the point at the closer distance
        3. Calculate number of samples to cover (car half-width + tolerance)
        4. Starting at the more distant point, overwrite samples with closer distance
           (Do not overwrite any points that are already closer!)
        """
        out = np.copy(ranges)
        
        for idx in disparities:
            d1, d2 = out[idx], out[idx + 1]
            
            # Step 2: Identify closer vs farther point
            if d1 <= d2:
                closer_dist = d1
                closer_idx = idx
                farther_idx = idx + 1
                direction = +1  # extend forward
            else:
                closer_dist = d2
                closer_idx = idx + 1
                farther_idx = idx
                direction = -1  # extend backward
            
            # Calculate number of samples to cover car half-width + tolerance at closer distance
            radius = (CAR_WIDTH * 0.5) + SAFETY_MARGIN
            half_angle = bubble_half_angle(closer_dist)
            num_samples = int(half_angle / angle_increment)
            num_samples = clamp(num_samples, 0, 60)  # cap for safety
            
            # Step 3-4: Starting at farther point, extend in direction away from closer point
            # Overwrite with closer distance, but DON'T overwrite already closer points
            for k in range(num_samples + 1):
                j = farther_idx + direction * k
                if j < 0 or j >= len(out):
                    break
                # Step 4: Only overwrite if current point is farther than closer_dist
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

    # ---------- Cornering safety check ----------

    def check_side_clearance(self, ranges, angle_min, angle_increment, steering_cmd):
        """
        Check for obstacles on the sides/rear of the car to prevent corner strikes.
        
        Scans LIDAR samples below -90° or above +90° (in car frame) to check the sides
        and back of the car. If any point is below the safe distance on the side of the
        car in the direction the car is turning, returns False to override steering.
        
        Args:
            ranges: preprocessed lidar ranges
            angle_min: minimum angle of lidar scan (radians)
            angle_increment: angular resolution (radians)
            steering_cmd: current steering command (positive = left, negative = right)
        
        Returns:
            True if safe to turn, False if should go straight instead
        """
        n = len(ranges)
        
        # Determine which side to check based on steering direction
        # Positive steering = turning left, negative = turning right
        if abs(steering_cmd) < 5.0:  # Nearly straight, no need to check
            return True
        
        turning_left = steering_cmd > 0
        
        # Define side/rear regions in car frame
        # Left side/rear: +90° to +180°
        # Right side/rear: -90° to -180°
        if turning_left:
            # Check left side/rear (+90° to +180° in car frame)
            car_min = math.radians(90.0)
            car_max = math.radians(180.0)
        else:
            # Check right side/rear (-90° to -180° in car frame)
            car_min = math.radians(-180.0)
            car_max = math.radians(-90.0)
        
        # Convert to LiDAR frame
        lid_min = car_to_lidar_angle(car_min)
        lid_max = car_to_lidar_angle(car_max)
        
        # Get indices for this region
        i0 = int((lid_min - angle_min) / angle_increment)
        i1 = int((lid_max - angle_min) / angle_increment)
        i0 = clamp(i0, 0, n - 1)
        i1 = clamp(i1, 0, n - 1)
        
        # Ensure we scan in the right direction
        if i0 > i1:
            i0, i1 = i1, i0
        
        # Check all points in this region
        for i in range(i0, i1 + 1):
            if ranges[i] < SIDE_SAFETY_DISTANCE:
                # Obstacle too close on the side we're turning toward
                rospy.logwarn_throttle(
                    0.5,
                    "Corner safety: Obstacle detected at %.2fm on %s side. Overriding turn.",
                    ranges[i],
                    "left" if turning_left else "right"
                )
                return False
        
        return True

    # ---------- Gap finding ----------

    def find_target_direction_farthest_gap(self, ranges, angle_min, angle_increment, start_idx, end_idx):
        """
        Find the sample with the farthest distance in the valid range.
        This is the target direction.
        
        Returns: (target_idx, target_angle_lidar, target_distance)
        """
        # Simply find the maximum distance in the range
        if start_idx >= end_idx:
            # Fallback
            mid_idx = (start_idx + end_idx) // 2
            return (mid_idx, angle_min + mid_idx * angle_increment, ranges[mid_idx])
        
        # Find index of maximum distance in the valid window
        window = ranges[start_idx:end_idx + 1]
        local_max_idx = int(np.argmax(window))
        target_idx = start_idx + local_max_idx
        target_dist = ranges[target_idx]
        target_angle_lidar = angle_min + target_idx * angle_increment
        
        return (target_idx, target_angle_lidar, target_dist)
    
    def find_target_direction_widest_gap(self, ranges, angle_min, angle_increment, start_idx, end_idx):
        # Ensure start_idx <= end_idx
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        # make a list of gaps where every sample is >= MIN_FREE_DISTANCE
        gaps = []  # elements are tuples (width, local_start, local_end_exclusive)
        window = ranges[start_idx:end_idx + 1]
        local_start = None
        for i, sample in enumerate(window):
            if sample >= MIN_FREE_DISTANCE:
                if local_start is None:
                    local_start = i
            else:
                if local_start is not None:
                    local_end = i  # exclusive
                    width = local_end - local_start
                    gaps.append((width, local_start, local_end))
                    local_start = None

        # trailing gap to end of window
        if local_start is not None:
            local_end = len(window)
            width = local_end - local_start
            gaps.append((width, local_start, local_end))

        # Find the widest gap (largest width). If multiple equal-width gaps exist, pick the first.
        if gaps:
            width, local_start, local_end = max(gaps, key=lambda x: x[0])
            # midpoint in local window coordinates (use integer floor)
            midpoint_local = int(local_start + (local_end - local_start) // 2)
            # convert to global index in the full ranges array
            target_idx = start_idx + midpoint_local
            target_lidar_angle = angle_min + target_idx * angle_increment
            target_dist = ranges[target_idx]
            return (target_idx, target_lidar_angle, target_dist)
        else:
            rospy.logwarn("No valid gaps found; defaulting to center")
            mid_idx = (start_idx + end_idx) // 2
            return (mid_idx, angle_min + mid_idx * angle_increment, ranges[mid_idx])

    def find_target_direction_farthest_distance_in_widest_gap(self, ranges, angle_min, angle_increment, start_idx, end_idx):
        # Ensure start_idx <= end_idx
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        # make a list of gaps where every sample is >= MIN_FREE_DISTANCE
        gaps = []  # elements are tuples (width, local_start, local_end_exclusive)
        window = ranges[start_idx:end_idx + 1]
        #print(window)
        local_start = None
        for i, sample in enumerate(window):
            if sample >= MIN_FREE_DISTANCE:
                if local_start is None:
                    local_start = i
            else:
                if local_start is not None:
                    local_end = i  # exclusive
                    width = local_end - local_start
                    average_local_distance = np.mean(window[local_start:local_end + 1])
                    gaps.append((width*average_local_distance, local_start, local_end))
                    local_start = None

        # trailing gap to end of window
        if local_start is not None:
            local_end = len(window)
            width = local_end - local_start

            # Find index of maximum distance in the valid window
            average_local_distance = np.mean(window[local_start:local_end+1])

            gaps.append((width*average_local_distance, local_start, local_end))

        # Find the widest gap (largest width). If multiple equal-width gaps exist, pick the first.
        if gaps:
            width, local_start, local_end = max(gaps, key=lambda x: x[0])
            # midpoint in local window coordinates (use integer floor)
            midpoint_local = int(local_start + (local_end - local_start) // 2)
            # convert to global index in the full ranges array
            target_idx = start_idx + midpoint_local
            target_lidar_angle = angle_min + target_idx * angle_increment
            target_dist = ranges[target_idx]
            return (target_idx, target_lidar_angle, target_dist)
        else:
            rospy.logwarn("No valid gaps found; defaulting to center")
            mid_idx = (start_idx + end_idx) // 2
            return (mid_idx, angle_min + mid_idx * angle_increment, ranges[mid_idx])

    # ---------- Control ----------

    def calculate_steering(self, target_angle_lidar):
        """Calculate steering command from target angle."""
        # Convert to car frame: 0 rad = forward, left positive
        a_car = lidar_to_car_angle(target_angle_lidar)
        
        # Map angle to steering command [-100, 100]
        # Use proportional control with tunable gain
        dev_deg = math.degrees(a_car)
        steering_cmd = (dev_deg / 45.0) * 100.0 * STEERING_GAIN
        steering_cmd = clamp(steering_cmd, MIN_STEERING, MAX_STEERING)
        steering_cmd *= STEERING_SIGN
        
        # Apply smoothing
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
        
        # Apply smoothing to velocity for gradual speed scaling
        v = VELOCITY_SMOOTH_ALPHA * self.prev_velocity + (1.0 - VELOCITY_SMOOTH_ALPHA) * v
        self.prev_velocity = v
        
        return clamp(v, 0.0, 100.0)

    # ---------- Visualization ----------

    def publish_car_footprint(self):
        m = Marker()
        m.header.frame_id = "car_2_laser"
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
        m.header.frame_id = "car_2_laser"
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
        """
        DISPARITY EXTENDER ALGORITHM:
        
        1. Take raw array of LIDAR samples
        2. Find disparities (large jumps between consecutive points)
        3. For each disparity:
           - Pick the closer point
           - Calculate samples needed to cover car half-width + tolerance
           - Starting at farther point, overwrite with closer distance
           - Never overwrite already closer points
        4. Search filtered distances in front hemisphere (-90° to +90°)
        5. Find sample with farthest distance → that's target direction
        6. Actuate car toward target
        
        Target: < 10ms processing time
        """
        
        # Step 1: Take raw array and preprocess
        ranges = self.preprocess_lidar(scan)

        # Step 2: Find disparities
        disparities = self.find_disparities(ranges)

        # Step 3: Extend disparities (repeat for every disparity)
        processed = self.extend_disparities(ranges, scan.angle_increment, disparities)

        # Step 4: Search through filtered distances between -90° and +90° (car frame)
        i0, i1 = self.front_window_indices(scan.angle_min, scan.angle_increment, len(processed))

        # Step 5: Find the sample with farthest distance → target direction
        # target_idx, target_angle_lidar, max_dist = self.find_target_direction_farthest_gap(
        #     processed, scan.angle_min, scan.angle_increment, i0, i1
        # )
        target_idx, target_angle_lidar, max_dist = self.find_target_direction_farthest_distance_in_widest_gap(
            processed, scan.angle_min, scan.angle_increment, i0, i1
        )

        # Step 6: Actuate - calculate steering and velocity
        steering_cmd = self.calculate_steering(target_angle_lidar)


        
        # Check side clearance for cornering safety
        #if not self.check_side_clearance(processed, scan.angle_min, scan.angle_increment, steering_cmd):
            # Override steering to go straight if obstacle detected on turning side
            #steering_cmd = 0.0
        
        velocity_cmd = self.calculate_velocity(steering_cmd, max_dist)

        # Publish AckermannDrive message (steering [-100,100], velocity [0,100])
        cmd = AckermannDrive()
        cmd.steering_angle = steering_cmd
        cmd.speed = velocity_cmd
        self.command_pub.publish(cmd)

        # Visualizations
        self.publish_car_footprint()
        self.publish_steering_arrow(steering_cmd)
        self.publish_disparities(disparities, ranges, scan.angle_min, scan.angle_increment)
        self.publish_target_point(target_angle_lidar, max_dist)
        self.publish_processed_scan(scan, processed)

        # Debug logging
        rospy.loginfo_throttle(
            1.0,
            "Disparities: %d | Target: %.1f° (%.2fm) | Steering: %.1f | Speed: %.1f",
            len(disparities),
            math.degrees(target_angle_lidar),
            max_dist,
            steering_cmd,
            velocity_cmd
        )

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

