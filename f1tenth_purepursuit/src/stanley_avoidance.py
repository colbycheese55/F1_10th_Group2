#!/usr/bin/env python

"""
Pure Pursuit Avoidance Controller for F1Tenth

This node implements the Pure Pursuit steering method with obstacle avoidance
using an occupancy grid generated from LiDAR scans.
"""

# Import necessary libraries
import rospy
import os
import sys
import csv
import math
import copy
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

from ackermann_msgs.msg import AckermannDrive
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Point, Point32, PolygonStamped
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
import tf
from tf.transformations import euler_from_quaternion, quaternion_matrix

# ============================================================================
# PARAMETERS - Tune these values to adjust controller behavior
# ============================================================================

# Steering Range from -100.0 to 100.0
STEERING_RANGE = 100.0

# Vehicle physical parameters
WHEELBASE_LEN = 0.325

# Pure pursuit controller gains
K_P = 1.0       # Pure pursuit gain
K_P_OBSTACLE = 0.8  # Pure pursuit gain during obstacle avoidance

# Lookahead parameters
MIN_LOOKAHEAD = 0.8     # Minimum lookahead distance (meters)
MAX_LOOKAHEAD = 2.0     # Maximum lookahead distance (meters)
MIN_LOOKAHEAD_SPEED = 8.0   # Speed at minimum lookahead (0-100 range)
MAX_LOOKAHEAD_SPEED = 15.0   # Speed at maximum lookahead (0-100 range)

# Speed control parameters
MAX_SPEED = 40.0      # Maximum speed (0-100 range)
MIN_SPEED = 5.0       # Minimum speed for sharp turns
VELOCITY_SCALE_FACTOR = 10.0  # Converts m/s from waypoint to 0-100 range
VELOCITY_PERCENTAGE = 10.0     # Percentage of target velocity to use

# Steering parameters
MAX_STEERING_ANGLE_RAD = 0.4  # Maximum steering angle in radians
STEERING_LIMIT_DEG = 25.0     # Steering limit in degrees

# Occupancy grid parameters
GRID_WIDTH_METERS = 3.0     # Width of occupancy grid in meters
CELLS_PER_METER = 10        # Resolution of occupancy grid
COLLISION_MARGIN_METERS = 0.18  # Safety margin for collision checking (car is ~0.3m wide)

# ============================================================================
# END PARAMETERS
# ============================================================================

# Global variables for storing the path, path resolution, frame ID, and car details
plan = []
velocities = []
path_resolution = []
frame_id = 'map'
car_name = str(sys.argv[1])
trajectory_name = str(sys.argv[2])

# Publishers
command_pub = None
polygon_pub = None
path_marker_pub = None
pose_marker_pub = None
target_marker_pub = None
steering_marker_pub = None
lookahead_marker_pub = None
occupancy_grid_pub = None
avoidance_path_pub = None
avoidance_path_array_pub = None
current_waypoint_pub = None

# Global state variables
current_pose = None
current_heading = None
goal_pos = None
closest_waypoint_index = 0
target_velocity = 0.0
obstacle_detected = False
obstacle_detected_count = 0  # Hysteresis counter for obstacle detection
OBSTACLE_HYSTERESIS_THRESHOLD = 3  # Number of frames before switching state
current_speed_cmd = 0.0
occupancy_grid = None
last_steering_cmd = 0.0  # For steering smoothing
STEERING_SMOOTHING_FACTOR = 0.3  # Lower = smoother but slower response
last_avoidance_direction = 0  # Track which direction we're avoiding (-1=left, 1=right, 0=none)

# Grid parameters (computed at runtime)
grid_height = 0
grid_width = 0
cell_y_offset = 0

# TF listener
tf_listener = None


class Utils:
    """Utility class for visualization and grid traversal."""
    
    @staticmethod
    def draw_marker(frame_id, stamp, position, publisher, color="red", marker_id=0):
        """Draw a sphere marker at the given position."""
        if position is None or publisher is None:
            return
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.a = 1.0
        if color == "red":
            marker.color.r = 1.0
        elif color == "green":
            marker.color.g = 1.0
        elif color == "blue":
            marker.color.b = 1.0
        elif color == "yellow":
            marker.color.r = 1.0
            marker.color.g = 1.0
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.lifetime = rospy.Duration(0.1)
        publisher.publish(marker)

    @staticmethod
    def draw_marker_array(frame_id, stamp, positions, publisher):
        """Draw multiple sphere markers."""
        if publisher is None:
            return
        marker_array = MarkerArray()
        for i, position in enumerate(positions):
            if position is None:
                continue
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.pose.position.x = position[0]
            marker.pose.position.y = position[1]
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.lifetime = rospy.Duration(0.1)
            marker_array.markers.append(marker)
        publisher.publish(marker_array)

    @staticmethod
    def draw_lines(frame_id, stamp, path, publisher):
        """Draw lines connecting points in the path."""
        if publisher is None or len(path) < 2:
            return
        points = []
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            point = Point()
            point.x = a[0]
            point.y = a[1]
            points.append(copy.deepcopy(point))
            point.x = b[0]
            point.y = b[1]
            points.append(copy.deepcopy(point))

        line_list = Marker()
        line_list.header.frame_id = frame_id
        line_list.header.stamp = stamp
        line_list.id = 0
        line_list.type = Marker.LINE_LIST
        line_list.action = Marker.ADD
        line_list.scale.x = 0.1
        line_list.color.a = 1.0
        line_list.color.r = 0.0
        line_list.color.g = 1.0
        line_list.color.b = 0.0
        line_list.points = points
        line_list.lifetime = rospy.Duration(0.1)
        publisher.publish(line_list)

    @staticmethod
    def traverse_grid(start, end):
        """
        Bresenham's line algorithm for fast voxel traversal.
        
        CREDIT TO: Rogue Basin
        CODE TAKEN FROM: http://www.roguebasin.com/index.php/Bresenham%27s_Line_Algorithm
        """
        x1, y1 = int(start[0]), int(start[1])
        x2, y2 = int(end[0]), int(end[1])
        dx = x2 - x1
        dy = y2 - y1

        is_steep = abs(dy) > abs(dx)

        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2

        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1

        dx = x2 - x1
        dy = y2 - y1

        error = int(dx / 2.0)
        ystep = 1 if y1 < y2 else -1

        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx
        return points


utils = Utils()


def construct_path():
    """Load waypoints from CSV file."""
    global plan, velocities, path_resolution
    
    file_path = os.path.expanduser('/home/nvidia/{}.csv'.format(trajectory_name))
    
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for waypoint in csv_reader:
            if len(waypoint) >= 2:
                plan.append(waypoint)

    # Convert string coordinates to floats
    for index in range(len(plan)):
        for point in range(len(plan[index])):
            plan[index][point] = float(plan[index][point])
        # If velocity is not provided, append a default value
        if len(plan[index]) < 3:
            plan[index].append(0.0)
        velocities.append(plan[index][2] if len(plan[index]) > 2 else 0.0)

    # Calculate path resolution
    for index in range(1, len(plan)):
        dx = plan[index][0] - plan[index - 1][0]
        dy = plan[index][1] - plan[index - 1][1]
        path_resolution.append(math.sqrt(dx * dx + dy * dy))


def publish_path_marker():
    """Publish the reference path as a LINE_STRIP marker for RViz visualization."""
    if path_marker_pub is None:
        return
        
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
    
    for waypoint in plan:
        p = Point()
        p.x = waypoint[0]
        p.y = waypoint[1]
        p.z = 0.0
        path_marker.points.append(p)
    
    if len(plan) > 0:
        p = Point()
        p.x = plan[0][0]
        p.y = plan[0][1]
        p.z = 0.0
        path_marker.points.append(p)
    
    path_marker.lifetime = rospy.Duration(0)
    path_marker_pub.publish(path_marker)


def compute_lookahead_distance():
    """Compute adaptive lookahead distance based on current speed command."""
    v = current_speed_cmd
    
    if v <= MIN_LOOKAHEAD_SPEED:
        return MIN_LOOKAHEAD
    if v >= MAX_LOOKAHEAD_SPEED:
        return MAX_LOOKAHEAD
    
    ratio = (v - MIN_LOOKAHEAD_SPEED) / (MAX_LOOKAHEAD_SPEED - MIN_LOOKAHEAD_SPEED)
    return MIN_LOOKAHEAD + ratio * (MAX_LOOKAHEAD - MIN_LOOKAHEAD)


def transform_waypoints_to_vehicle(waypoints, car_x, car_y, heading):
    """Transform waypoints from world frame to vehicle frame."""
    transformed = []
    cos_h = math.cos(-heading)
    sin_h = math.sin(-heading)
    
    for wp in waypoints:
        # Translation
        dx = wp[0] - car_x
        dy = wp[1] - car_y
        # Rotation
        x_vehicle = cos_h * dx - sin_h * dy
        y_vehicle = sin_h * dx + cos_h * dy
        transformed.append([x_vehicle, y_vehicle])
    
    return transformed


def get_closest_waypoint(car_x, car_y):
    """Find the closest waypoint to the car's current position."""
    min_dist = float('inf')
    closest_idx = 0
    
    for i, wp in enumerate(plan):
        dx = wp[0] - car_x
        dy = wp[1] - car_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    
    return closest_idx


def get_lookahead_waypoint(car_x, car_y, heading, lookahead_distance, base_index):
    """
    Get the waypoint at the lookahead distance.
    Uses the same algorithm as pure_pursuit.py - traverse forward from base projection.
    Returns (waypoint_in_vehicle_frame, waypoint_in_world_frame, index)
    """
    # Start from the base projection and move forward along the path
    cumulative_distance = 0.0
    target_index = base_index
    
    # Traverse the path until we've covered the lookahead distance
    # Use modulo to wrap around the path (treat it as a closed loop)
    num_points = len(plan)
    target_x_world = None
    target_y_world = None
    
    for j in range(num_points):
        i = (base_index + j) % num_points
        i_next = (i + 1) % num_points
        
        dx = plan[i_next][0] - plan[i][0]
        dy = plan[i_next][1] - plan[i][1]
        segment_distance = math.sqrt(dx*dx + dy*dy)
        
        if cumulative_distance + segment_distance >= lookahead_distance:
            # Interpolate to find the exact target point
            remaining_distance = lookahead_distance - cumulative_distance
            ratio = remaining_distance / segment_distance if segment_distance > 0 else 0
            target_x_world = plan[i][0] + ratio * dx
            target_y_world = plan[i][1] + ratio * dy
            target_index = i
            break
        
        cumulative_distance += segment_distance
        target_index = i_next
    else:
        # Fallback (shouldn't happen with wrap-around)
        target_x_world = plan[target_index][0]
        target_y_world = plan[target_index][1]
    
    if target_x_world is None:
        return None, None, -1
    
    # Transform target point to vehicle frame
    dx = target_x_world - car_x
    dy = target_y_world - car_y
    target_x_vehicle = math.cos(-heading) * dx - math.sin(-heading) * dy
    target_y_vehicle = math.sin(-heading) * dx + math.cos(-heading) * dy
    
    return ([target_x_vehicle, target_y_vehicle], 
            [target_x_world, target_y_world], 
            target_index)





# Occupancy grid functions
def init_grid_params():
    """Initialize grid parameters based on lookahead distance."""
    global grid_height, grid_width, cell_y_offset
    grid_height = int(MAX_LOOKAHEAD * CELLS_PER_METER)
    grid_width = int(GRID_WIDTH_METERS * CELLS_PER_METER)
    cell_y_offset = (grid_width // 2) - 1


def local_to_grid(x, y):
    """Convert local (vehicle frame) coordinates to grid indices."""
    i = int(x * -CELLS_PER_METER + (grid_height - 1))
    j = int(y * -CELLS_PER_METER + cell_y_offset)
    return (i, j)


def local_to_grid_parallel(x, y):
    """Vectorized conversion from local to grid coordinates."""
    i = np.round(x * -CELLS_PER_METER + (grid_height - 1)).astype(int)
    j = np.round(y * -CELLS_PER_METER + cell_y_offset).astype(int)
    return i, j


def grid_to_local(point):
    """Convert grid indices to local (vehicle frame) coordinates."""
    i, j = point[0], point[1]
    x = (i - (grid_height - 1)) / -CELLS_PER_METER
    y = (j - cell_y_offset) / -CELLS_PER_METER
    return (x, y)


def populate_occupancy_grid(ranges, angle_increment, angle_min):
    """Populate occupancy grid using LiDAR scans."""
    global occupancy_grid
    
    IS_OCCUPIED = 100
    IS_FREE = 0
    
    occupancy_grid = np.full(shape=(grid_height, grid_width), fill_value=IS_FREE, dtype=int)
    
    ranges = np.array(ranges)
    indices = np.arange(len(ranges))
    
    # Calculate angles (adjusting for sensor orientation)
    thetas = angle_min + indices * angle_increment
    
    # Filter valid ranges
    valid_mask = (ranges > 0.1) & (ranges < MAX_LOOKAHEAD)
    
    # Convert to local coordinates (in vehicle frame)
    xs = ranges * np.cos(thetas)
    ys = ranges * np.sin(thetas)
    
    # Convert to grid coordinates
    i, j = local_to_grid_parallel(xs, ys)
    
    # Mark occupied cells
    occupied_indices = np.where(
        valid_mask & 
        (i >= 0) & (i < grid_height) & 
        (j >= 0) & (j < grid_width)
    )
    occupancy_grid[i[occupied_indices], j[occupied_indices]] = IS_OCCUPIED


def convolve_occupancy_grid():
    """Apply convolution to expand obstacles in the occupancy grid."""
    global occupancy_grid
    kernel = np.ones(shape=[2, 2])
    occupancy_grid = signal.convolve2d(
        occupancy_grid.astype('int'), kernel.astype('int'), 
        boundary='symm', mode='same'
    )
    occupancy_grid = np.clip(occupancy_grid, -1, 100)


def publish_occupancy_grid(frame_id, stamp):
    """Publish the occupancy grid for visualization."""
    if occupancy_grid_pub is None:
        return
        
    oc = OccupancyGrid()
    oc.header.frame_id = frame_id
    oc.header.stamp = stamp
    # Center the grid on the vehicle/LiDAR
    # The grid extends from -MAX_LOOKAHEAD behind to 0 ahead (in vehicle frame, x points forward)
    # And from -GRID_WIDTH_METERS/2 to +GRID_WIDTH_METERS/2 laterally
    oc.info.origin.position.x = -MAX_LOOKAHEAD  # Grid starts MAX_LOOKAHEAD behind vehicle
    oc.info.origin.position.y = -GRID_WIDTH_METERS / 2.0  # Center laterally
    oc.info.origin.position.z = 0.0
    oc.info.width = grid_height
    oc.info.height = grid_width
    oc.info.resolution = 1.0 / CELLS_PER_METER
    oc.data = np.fliplr(np.rot90(occupancy_grid, k=1)).flatten().tolist()
    occupancy_grid_pub.publish(oc)


def check_collision(cell_a, cell_b, margin=0):
    """Check if path between two cells has a collision."""
    IS_OCCUPIED = 100
    
    for i in range(-margin, margin + 1):
        cell_a_margin = (cell_a[0], cell_a[1] + i)
        cell_b_margin = (cell_b[0], cell_b[1] + i)
        for cell in utils.traverse_grid(cell_a_margin, cell_b_margin):
            if (cell[0] < 0 or cell[1] < 0 or 
                cell[0] >= grid_height or cell[1] >= grid_width):
                continue
            try:
                if occupancy_grid[cell[0], cell[1]] == IS_OCCUPIED:
                    return True
            except IndexError:
                return True
    return False


def check_collision_loose(cell_a, cell_b, margin=0):
    """Check collision with looser constraints (only checks second half of path)."""
    IS_OCCUPIED = 100
    
    for i in range(-margin, margin + 1):
        cell_a_margin = (
            int((cell_a[0] + cell_b[0]) / 2), 
            int((cell_a[1] + cell_b[1]) / 2) + i
        )
        cell_b_margin = (cell_b[0], cell_b[1] + i)
        for cell in utils.traverse_grid(cell_a_margin, cell_b_margin):
            if (cell[0] < 0 or cell[1] < 0 or 
                cell[0] >= grid_height or cell[1] >= grid_width):
                continue
            try:
                if occupancy_grid[cell[0], cell[1]] == IS_OCCUPIED:
                    return True
            except IndexError:
                return True
    return False


def drive_pure_pursuit(target_point_vehicle, k_p, target_velocity_value):
    """
    Compute steering using pure pursuit algorithm.
    Uses the same algorithm as pure_pursuit.py.
    """
    global current_speed_cmd, last_steering_cmd
    
    command = AckermannDrive()
    
    # Calculate lookahead distance
    L = math.sqrt(target_point_vehicle[0]**2 + target_point_vehicle[1]**2)
    if L < 0.01:
        L = 0.01
    
    # Pure pursuit curvature calculation
    # curvature = 2 * y / L^2
    y = target_point_vehicle[1]
    curvature = 2.0 * y / (L ** 2)
    
    # Calculate steering angle using bicycle model
    # steering_angle = atan(wheelbase * curvature)
    steering_angle = math.atan(WHEELBASE_LEN * curvature)
    
    # Clip to maximum steering angle
    steering_angle = np.clip(steering_angle, -math.radians(STEERING_LIMIT_DEG), math.radians(STEERING_LIMIT_DEG))
    
    # Normalize steering to [-100, 100] range
    normalized_steering = steering_angle / MAX_STEERING_ANGLE_RAD
    raw_steering_cmd = max(-STEERING_RANGE, min(STEERING_RANGE, normalized_steering * STEERING_RANGE))
    
    # Apply steering smoothing to reduce oscillation
    # Exponential moving average filter
    smoothed_steering = (STEERING_SMOOTHING_FACTOR * raw_steering_cmd + 
                         (1.0 - STEERING_SMOOTHING_FACTOR) * last_steering_cmd)
    command.steering_angle = smoothed_steering
    last_steering_cmd = smoothed_steering
    
    # Use velocity from waypoint if available, otherwise scale based on steering
    if target_velocity_value > 0.0:
        velocity = target_velocity_value * VELOCITY_SCALE_FACTOR * VELOCITY_PERCENTAGE
        velocity = min(velocity, MAX_SPEED)
    else:
        # Dynamic velocity scaling based on steering angle
        abs_angle_deg = abs(math.degrees(steering_angle))
        if abs_angle_deg < 10.0:
            velocity = MAX_SPEED
        elif abs_angle_deg < 20.0:
            velocity = (MAX_SPEED + MIN_SPEED) / 2
        else:
            velocity = MIN_SPEED
        velocity = velocity * VELOCITY_PERCENTAGE
    
    command.speed = velocity
    current_speed_cmd = command.speed
    return command





def pose_callback(data):
    """Callback for pose updates from particle filter."""
    global current_pose, current_heading, target_velocity, closest_waypoint_index
    
    current_pose = data.pose
    
    # Get heading from quaternion
    current_heading = euler_from_quaternion([
        data.pose.orientation.x,
        data.pose.orientation.y,
        data.pose.orientation.z,
        data.pose.orientation.w
    ])[2]
    
    car_x = current_pose.position.x
    car_y = current_pose.position.y
    
    # Get closest waypoint and velocity
    closest_waypoint_index = get_closest_waypoint(car_x, car_y)
    target_velocity = velocities[closest_waypoint_index] if velocities[closest_waypoint_index] > 0 else 1.0
    
    # Visualize closest waypoint
    utils.draw_marker(
        frame_id, data.header.stamp,
        [plan[closest_waypoint_index][0], plan[closest_waypoint_index][1]],
        current_waypoint_pub, color="blue"
    )


def scan_callback(scan_msg):
    """
    LiDAR scan callback - main control loop.
    Updates occupancy grid and performs obstacle avoidance.
    """
    global obstacle_detected, obstacle_detected_count, goal_pos, last_avoidance_direction
    
    if current_pose is None or len(plan) == 0:
        return
    
    car_x = current_pose.position.x
    car_y = current_pose.position.y
    
    # Compute lookahead distance based on current speed
    lookahead_distance = compute_lookahead_distance()
    
    # Get lookahead waypoint starting from closest waypoint
    goal_vehicle, goal_world, goal_idx = get_lookahead_waypoint(
        car_x, car_y, current_heading, lookahead_distance, closest_waypoint_index
    )
    
    if goal_vehicle is None:
        rospy.logwarn("Could not find lookahead waypoint")
        return
    
    goal_pos = goal_vehicle
    
    # Visualize goal point
    utils.draw_marker(
        frame_id, scan_msg.header.stamp,
        goal_world, target_marker_pub, color="red"
    )
    
    scan_msg.ranges = np.array(scan_msg.ranges)
    scan_msg.ranges[~np.isfinite(scan_msg.ranges)] = 10.0

    # Populate occupancy grid
    populate_occupancy_grid(
        scan_msg.ranges, 
        scan_msg.angle_increment,
        scan_msg.angle_min
    )
    convolve_occupancy_grid()
    
    # Publish occupancy grid for visualization (use base_link frame, not laser frame)
    publish_occupancy_grid('{}/base_link'.format(car_name), scan_msg.header.stamp)
    
    # Check for obstacles and compute avoidance path
    path_local = []
    current_pos_grid = np.array(local_to_grid(0, 0))
    goal_pos_grid = np.array(local_to_grid(goal_pos[0], goal_pos[1]))
    target = None
    path_local = [grid_to_local(current_pos_grid)]
    
    MARGIN = int(CELLS_PER_METER * COLLISION_MARGIN_METERS)
    
    # Check for collision
    collision_now = check_collision(current_pos_grid, goal_pos_grid, margin=MARGIN)
    
    # Apply hysteresis to obstacle detection to prevent rapid switching
    if collision_now:
        obstacle_detected_count = min(obstacle_detected_count + 1, OBSTACLE_HYSTERESIS_THRESHOLD + 1)
    else:
        obstacle_detected_count = max(obstacle_detected_count - 1, 0)
    
    # Only change obstacle_detected state after hysteresis threshold
    if obstacle_detected_count >= OBSTACLE_HYSTERESIS_THRESHOLD:
        obstacle_detected = True
    elif obstacle_detected_count == 0:
        obstacle_detected = False
    # Otherwise, keep previous state (hysteresis)
    
    if obstacle_detected:
        print
        # Generate shifts - prefer continuing in the same direction to avoid oscillation
        # Start with the last successful avoidance direction
        if last_avoidance_direction != 0:
            # Continue in the same direction first, then try the other side
            primary_dir = last_avoidance_direction
            shifts = []
            for i in range(1, 21):
                shifts.append(i * primary_dir)  # Primary direction first
            for i in range(1, 21):
                shifts.append(i * -primary_dir)  # Then opposite direction
        else:
            # No previous direction, try right first (positive shifts), then left
            shifts = list(range(1, 21)) + list(range(-1, -21, -1))
        
        found = False
        for shift in shifts:
            new_goal = goal_pos_grid + np.array([0, shift])
            
            if not check_collision(current_pos_grid, new_goal, margin=int(1.5 * MARGIN)):
                target = grid_to_local(new_goal)
                found = True
                path_local.append(target)
                # Remember which direction worked
                last_avoidance_direction = 1 if shift > 0 else -1
                print("Found avoidance path (condition 1)")
                break
        
        if not found:
            # Obstacle is very close, try steeper turns with middle point
            middle_grid_point = np.array(
                current_pos_grid + (goal_pos_grid - current_pos_grid) / 2
            ).astype(int)
            
            for shift in shifts:
                new_goal = middle_grid_point + np.array([0, shift])
                if not check_collision(current_pos_grid, new_goal, margin=int(1.5 * MARGIN)):
                    target = grid_to_local(new_goal)
                    found = True
                    path_local.append(target)
                    last_avoidance_direction = 1 if shift > 0 else -1
                    print("Found avoidance path (condition 2)")
                    break
        
        if not found:
            # Try with looser collision checking
            for shift in shifts:
                new_goal = middle_grid_point + np.array([0, shift])
                if not check_collision_loose(current_pos_grid, new_goal, margin=MARGIN):
                    target = grid_to_local(new_goal)
                    found = True
                    path_local.append(target)
                    last_avoidance_direction = 1 if shift > 0 else -1
                    print("Found avoidance path (condition 3)")
                    break
    else:
        # No obstacle - reset avoidance direction for next obstacle encounter
        last_avoidance_direction = 0
        target = grid_to_local(goal_pos_grid)
        path_local.append(target)
    
    # Compute and publish drive command
    if target:
        # Use pure pursuit for both normal and obstacle avoidance scenarios
        # Blend gains smoothly to avoid abrupt steering changes
        if obstacle_detected:
            print("Obstacle detected - using pure pursuit avoidance")
            # Use slightly higher gain during avoidance for more responsive steering
            effective_k_p = K_P_OBSTACLE
        else:
            # Use standard pure pursuit when path is clear
            effective_k_p = K_P
        
        command = drive_pure_pursuit(target, effective_k_p, target_velocity)
        
        command_pub.publish(command)
        
        print("Obstacle: {} | Lookahead: {:.2f} | Speed: {:.2f} | Steering: {:.2f}".format(
            obstacle_detected, lookahead_distance, command.speed, command.steering_angle
        ))
    else:
        rospy.logwarn("Could not find target path - stopping vehicle")
        command = AckermannDrive()
        command.speed = 0.0
        command.steering_angle = 0.0
        command_pub.publish(command)
    
    # Visualization
    utils.draw_marker_array(
        scan_msg.header.frame_id, scan_msg.header.stamp, 
        path_local, avoidance_path_array_pub
    )
    utils.draw_lines(
        scan_msg.header.frame_id, scan_msg.header.stamp, 
        path_local, avoidance_path_pub
    )


if __name__ == '__main__':
    try:
        rospy.init_node('stanley_avoidance', anonymous=True)
        
        # Initialize grid parameters
        init_grid_params()
        
        # Initialize publishers
        command_pub = rospy.Publisher(
            '/{}/offboard/command'.format(car_name), 
            AckermannDrive, queue_size=1
        )
        polygon_pub = rospy.Publisher(
            '/{}/purepursuit_control/visualize'.format(car_name), 
            PolygonStamped, queue_size=1
        )
        path_marker_pub = rospy.Publisher(
            '/{}/purepursuit_control/path_marker'.format(car_name), 
            Marker, queue_size=1
        )
        pose_marker_pub = rospy.Publisher(
            '/{}/purepursuit_control/pose_marker'.format(car_name), 
            Marker, queue_size=1
        )
        target_marker_pub = rospy.Publisher(
            '/{}/purepursuit_control/target_marker'.format(car_name), 
            Marker, queue_size=1
        )
        steering_marker_pub = rospy.Publisher(
            '/{}/purepursuit_control/steering_marker'.format(car_name), 
            Marker, queue_size=1
        )
        lookahead_marker_pub = rospy.Publisher(
            '/{}/purepursuit_control/lookahead_marker'.format(car_name), 
            Marker, queue_size=1
        )
        current_waypoint_pub = rospy.Publisher(
            '/{}/stanley_avoidance/current_waypoint'.format(car_name), 
            Marker, queue_size=1
        )
        occupancy_grid_pub = rospy.Publisher(
            '/{}/stanley_avoidance/occupancy_grid'.format(car_name), 
            OccupancyGrid, queue_size=1
        )
        avoidance_path_pub = rospy.Publisher(
            '/{}/stanley_avoidance/avoidance_path'.format(car_name), 
            Marker, queue_size=1
        )
        avoidance_path_array_pub = rospy.Publisher(
            '/{}/stanley_avoidance/avoidance_path_array'.format(car_name), 
            MarkerArray, queue_size=1
        )
        
        # Load waypoints
        if not plan:
            rospy.loginfo('Loading trajectory: {}'.format(trajectory_name))
            construct_path()
            rospy.sleep(0.5)
            publish_path_marker()
            rospy.loginfo('Loaded {} waypoints'.format(len(plan)))
        
        # Subscribe to pose and scan topics
        rospy.Subscriber(
            '/{}/particle_filter/viz/inferred_pose'.format(car_name), 
            PoseStamped, pose_callback
        )
        rospy.Subscriber(
            '/{}/scan'.format(car_name), 
            LaserScan, scan_callback
        )
        
        rospy.loginfo('Pure Pursuit Avoidance node initialized for {}'.format(car_name))
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
