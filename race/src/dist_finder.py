#!/usr/bin/env python

import rospy 
import math
from sensor_msgs.msg import LaserScan
from race.msg import pid_input 

# Some useful variable declarations.
angle_range = 240	# Hokuyo 4LX has 240 degrees FoV for scan
forward_projection = 1.5	# distance (in m) that we project the car forward for correcting the error. You have to adjust this.
desired_distance = 0.9	# distance from the wall (in m). (defaults to right wall). You need to change this for the track
vel = 15 		# this vel variable is not really used here.
error = 0.0		# initialize the error
car_length = 0.50 # Traxxas Rally is 20 inches or 0.5 meters. Useful variable.

# Handle to the publisher that will publish on the error topic, messages of the type 'pid_input'
pub = rospy.Publisher('error', pid_input, queue_size=10)


def getRange(data,angle):
	# data: single message from topic /scan
    # angle: between -30 to 210 degrees, where 0 degrees is directly to the right, and 90 degrees is directly in front
    # Outputs length in meters to object with angle in lidar scan field of view
    # Make sure to take care of NaNs etc.
	
	# Convert our angle (thetha = right) to LIDAR frame (thetha = right is at 30 in LIDAR)
	# LIDAR scans from right to left with 240 FoV

	# angle_min corresponds to -30 (right side), angle_max corresponds to 210 (left side)
	lidar_angle = angle + 30.0  # Offset by 30 degrees to convert to LIDAR frame
	
	# index = (desired_angle - angle_min) / angle_increment
	index = (math.radians(lidar_angle)) / data.angle_increment
	index = int(index)
	
	# check bounds
	if index < 0 or index >= len(data.ranges):
		return 0.0
	
	# Get the range value
	distance = data.ranges[index]
	
	# Handle NaN and inf values
	if math.isnan(distance) or math.isinf(distance):
		return 0.0
	
	return distance



def callback(data):
	global forward_projection, error

	theta = 50 # you need to try different values for theta
	a = getRange(data,theta) # obtain the ray distance for theta
	b = getRange(data,0)	# obtain the ray distance for 0 degrees (i.e. directly to the right of the car)
	swing = math.radians(theta)

	## Your code goes here to determine the projected error as per the alrorithm
	# Compute Alpha, AB, and CD..and finally the error.
	
	# alpha = arctan((a*cos(theta) - b) / (a*sin(theta)))
	numerator = a * math.cos(swing) - b
	denominator = a * math.sin(swing)
	
	# Handle division by zero
	if denominator == 0:
		alpha = 0
	else:
		alpha = math.atan(numerator / denominator)
	
	# Calculate AB (current distance from wall perpendicular to car)
	AB = b * math.cos(alpha)
	
	# Calculate CD (projected distance at lookahead point)
	CD = AB + forward_projection * math.sin(alpha)
	
	# Calculate error (difference between desired and projected distance)
	error = desired_distance - CD

	msg = pid_input()	# An empty msg is created of the type pid_input
	# this is the error that you want to send to the PID for steering correction.
	msg.pid_error = error
	msg.pid_vel = vel		# velocity error can also be sent.
	pub.publish(msg)


if __name__ == '__main__':
	print("Hokuyo LIDAR node started")
	rospy.init_node('dist_finder',anonymous = True)
	rospy.Subscriber("/car_2/scan",LaserScan,callback)
	rospy.spin()
