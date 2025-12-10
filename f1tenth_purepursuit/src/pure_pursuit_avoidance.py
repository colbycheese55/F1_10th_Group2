#!/usr/bin/env python

"""
Pure Pursuit with Obstacle Avoidance

This node runs pure pursuit steering but slows down/stops if there's an obstacle
between the car and the path.
"""

import rospy
import os
import sys
import csv
import math
import numpy as np
from scipy import signal

from ackermann_msgs.msg import AckermannDrive
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
import tf

# ============================================================================
# PARAMETERS
# ============================================================================

# Vehicle physical parameters
WHEELBASE_LEN = 0.325
STEERING_RANGE = 100.0

# Occupancy grid parameters
GRID_WIDTH_METERS = 3.0     # Width of occupancy grid in meters
CELLS_PER_METER = 10        # Resolution of occupancy grid
COLLISION_MARGIN_METERS = 0.18  # Safety margin for collision checking

# Speed parameters
MAX_SPEED = 65.0            # Maximum speed
MIN_SPEED = 0.0             # Speed when obstacle detected
NORMAL_SPEED = 30.0         # Normal cruising speed

# Lookahead parameters
L_MIN = 0.8
L_MAX = 2.0
V_MIN_LD = 8.0
V_MAX_LD = 15.0

# ============================================================================
# Global variables
# ============================================================================

plan = []
path_resolution = []
frame_id = 'map'
car_name = str(sys.argv[1])
trajectory_name = str(sys.argv[2])

# LiDAR data
latest_scan = None
occupancy_grid = None
grid_height = 0
grid_width = 0
cell_y_offset = 0

# State
current_speed_cmd = 0.0
seq = 0
prev_x = 0.0
prev_y = 0.0

# Publishers
command_pub = None

# Grid constants
IS_OCCUPIED = 100
IS_FREE = 0


# ============================================================================
# Utility functions
# ============================================================================

def construct_path():
    """Load waypoints from CSV file."""
    global plan
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

    # Calculate path resolution
    for index in range(1, len(plan)):
        dx = plan[index][0] - plan[index - 1][0]
        dy = plan[index][1] - plan[index - 1][1]
        path_resolution.append(math.sqrt(dx * dx + dy * dy))
    
    rospy.loginfo('Loaded {} waypoints from {}'.format(len(plan), trajectory_name))


def init_grid_params():
    """Initialize grid parameters."""
    global grid_height, grid_width, cell_y_offset
    grid_height = int(L_MAX * CELLS_PER_METER)
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


def populate_occupancy_grid(ranges, angle_increment, angle_min):
    """Populate occupancy grid using LiDAR scans."""
    global occupancy_grid
    
    occupancy_grid = np.full(shape=(grid_height, grid_width), fill_value=IS_FREE, dtype=int)
    
    ranges = np.array(ranges)
    indices = np.arange(len(ranges))
    
    # Calculate angles
    thetas = angle_min + indices * angle_increment
    
    # Filter valid ranges
    valid_mask = (ranges > 0.1) & (ranges < L_MAX)
    
    # Convert to local coordinates (vehicle frame)
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


def traverse_grid(start, end):
    """
    Bresenham's line algorithm for fast voxel traversal.
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


def check_collision(car_pos_cell, target_pos_cell, margin=0):
    """Check if path between car and target has a collision."""
    if occupancy_grid is None:
        return False
    
    margin_cells = int(COLLISION_MARGIN_METERS * CELLS_PER_METER)
    
    for i in range(-margin_cells, margin_cells + 1):
        car_margin = (car_pos_cell[0], car_pos_cell[1] + i)
        target_margin = (target_pos_cell[0], target_pos_cell[1] + i)
        
        for cell in traverse_grid(car_margin, target_margin):
            if (cell[0] < 0 or cell[1] < 0 or 
                cell[0] >= grid_height or cell[1] >= grid_width):
                continue
            try:
                if occupancy_grid[cell[0], cell[1]] == IS_OCCUPIED:
                    return True
            except IndexError:
                return True
    
    return False


def is_path_clear_to_target(target_x, target_y, car_x, car_y, heading):
    """
    Check if there's a clear path from car to target point.
    Returns True if path is clear (no obstacles).
    """
    if occupancy_grid is None:
        return True
    
    # Transform target to vehicle frame
    dx = target_x - car_x
    dy = target_y - car_y
    target_x_vehicle = math.cos(-heading) * dx - math.sin(-heading) * dy
    target_y_vehicle = math.sin(-heading) * dx + math.cos(-heading) * dy
    
    # Car position in grid (center of vehicle)
    car_cell = local_to_grid(0.0, 0.0)
    
    # Target position in grid
    target_cell = local_to_grid(target_x_vehicle, target_y_vehicle)
    
    # Clamp to grid bounds
    car_cell = (max(0, min(grid_height-1, car_cell[0])),
                max(0, min(grid_width-1, car_cell[1])))
    target_cell = (max(0, min(grid_height-1, target_cell[0])),
                   max(0, min(grid_width-1, target_cell[1])))
    
    # Check for collision
    return not check_collision(car_cell, target_cell)


def compute_lookahead_distance(speed):
    """Compute adaptive lookahead distance based on speed."""
    if speed <= V_MIN_LD:
        return L_MIN
    if speed >= V_MAX_LD:
        return L_MAX
    
    ratio = (speed - V_MIN_LD) / float(V_MAX_LD - V_MIN_LD)
    return L_MIN + ratio * (L_MAX - L_MIN)


# ============================================================================
# Callbacks
# ============================================================================

def lidar_callback(data):
    """Process LiDAR scans and populate occupancy grid."""
    global latest_scan, occupancy_grid
    latest_scan = data
    
    try:
        populate_occupancy_grid(data.ranges, data.angle_increment, data.angle_min)
        convolve_occupancy_grid()
    except Exception as e:
        rospy.logerr("Error processing LiDAR: {}".format(str(e)))


def control_callback(data):
    """Main control callback - runs pure pursuit with obstacle avoidance."""
    global current_speed_cmd, seq
    
    try:
        # Get current pose
        odom_x = data.pose.position.x
        odom_y = data.pose.position.y
        heading = tf.transformations.euler_from_quaternion((
            data.pose.orientation.x,
            data.pose.orientation.y,
            data.pose.orientation.z,
            data.pose.orientation.w
        ))[2]
        
        # Find closest point on path
        min_distance = float('inf')
        base_index = 0
        
        for i in range(len(plan)):
            dx = plan[i][0] - odom_x
            dy = plan[i][1] - odom_y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < min_distance:
                min_distance = distance
                base_index = i
        
        # Clamp to safe range
        base_index = min(base_index, len(plan) - 2)
        
        # Get lookahead distance
        lookahead_distance = compute_lookahead_distance(current_speed_cmd)
        
        # Find target point
        cumulative_distance = 0.0
        target_x = plan[base_index][0]
        target_y = plan[base_index][1]
        
        for i in range(base_index, len(plan) - 1):
            dx = plan[i+1][0] - plan[i][0]
            dy = plan[i+1][1] - plan[i][1]
            segment_distance = math.sqrt(dx*dx + dy*dy)
            
            if segment_distance < 1e-6:
                continue
            
            if cumulative_distance + segment_distance >= lookahead_distance:
                remaining_distance = lookahead_distance - cumulative_distance
                ratio = min(1.0, remaining_distance / segment_distance)
                target_x = plan[i][0] + ratio * dx
                target_y = plan[i][1] + ratio * dy
                break
            
            cumulative_distance += segment_distance
            target_x = plan[i+1][0]
            target_y = plan[i+1][1]
        else:
            target_x = plan[-1][0]
            target_y = plan[-1][1]
        
        # Create command
        command = AckermannDrive()
        
        # Transform target to vehicle frame
        dx = target_x - odom_x
        dy = target_y - odom_y
        target_x_vehicle = math.cos(-heading) * dx - math.sin(-heading) * dy
        target_y_vehicle = math.sin(-heading) * dx + math.cos(-heading) * dy
        
        # Pure pursuit steering calculation
        curvature = 2.0 * target_y_vehicle / (lookahead_distance ** 2)
        steering_angle = math.atan(WHEELBASE_LEN * curvature)
        
        # Convert to range [-100, 100]
        max_steering_angle_rad = 0.4
        normalized_steering = steering_angle / max_steering_angle_rad
        command.steering_angle = max(-STEERING_RANGE, min(STEERING_RANGE, normalized_steering * STEERING_RANGE))
        
        # **CHECK FOR OBSTACLE BETWEEN CAR AND TARGET**
        path_is_clear = is_path_clear_to_target(target_x, target_y, odom_x, odom_y, heading)
        
        if not path_is_clear:
            # Obstacle detected between car and path - STOP
            rospy.logwarn("Obstacle detected in path! Stopping...")
            command.speed = 0.0
            current_speed_cmd = 0.0
        else:
            # Path is clear - proceed with normal speed control
            abs_steering = abs(command.steering_angle)
            speed_scale = 1.0 - (abs_steering / STEERING_RANGE)
            command.speed = (MIN_SPEED + (NORMAL_SPEED - MIN_SPEED) * speed_scale)
            current_speed_cmd = command.speed
        
        # Publish command
        command_pub.publish(command)
        
    except Exception as e:
        rospy.logerr("Error in control callback: {}".format(str(e)))


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    try:
        rospy.init_node('pure_pursuit_avoidance', anonymous=True)
        
        # Initialize grid
        init_grid_params()
        
        # Load path
        if not plan:
            rospy.loginfo('Loading trajectory...')
            construct_path()
        
        # Initialize publisher
        command_pub = rospy.Publisher('/{}/offboard/command'.format(car_name), 
                                     AckermannDrive, queue_size=1)
        
        # Subscribe to LiDAR and pose
        rospy.Subscriber('/{}/scan'.format(car_name), LaserScan, lidar_callback)
        rospy.Subscriber('/{}/particle_filter/viz/inferred_pose'.format(car_name), 
                        PoseStamped, control_callback)
        
        rospy.loginfo("Pure Pursuit with Obstacle Avoidance node started")
        rospy.spin()
        
    except rospy.ROSInterruptException:
        pass
