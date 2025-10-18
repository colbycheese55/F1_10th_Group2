#!/usr/bin/env python
import math
import rospy
from race.msg import pid_input
from ackermann_msgs.msg import AckermannDrive
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# PID Control Params
kp = 0.0 #TODO
kd = 0.0 #TODO
ki = 0.0 #TODO
servo_offset = 0.0	# zero correction offset in case servo is misaligned and has a bias in turning.
prev_error = 0.0
integral_error = 0.0


# This code can input desired velocity from the user.
# velocity must be between [0,100] to move forward.
# The following velocity values correspond to different speed profiles.
# 15: Very Slow (Good for debug mode)
# 25: Slow and steady
# 35: Nice Autonomous Pace
# > 40: Careful, what you do here. Only use this if your autonomous steering is very reliable.
max_velocity = 40.0	# Maximum velocity on straightaways (low error)
min_velocity = 20.0	# Minimum velocity during turns (high error)
velocity_scale_factor = 1.0  # Scaling factor for velocity adjustment based on error

# Publisher for moving the car.
# TODO: Use the coorect topic /car_x/offboard/command. The multiplexer listens to this topic
command_pub = rospy.Publisher('/car_2/offboard/command', AckermannDrive, queue_size = 1)

# Publishers for RViz visualization
car_footprint_pub = rospy.Publisher('/car_footprint', Marker, queue_size=1)
steering_arrow_pub = rospy.Publisher('/steering_arrow', Marker, queue_size=1)

# Car dimensions (Traxxas Rally dimensions)
CAR_LENGTH = 0.50  # 20 inches = 0.5 meters
CAR_WIDTH = 0.20   # approximately 10 inches = 0.25 meters

def publish_car_footprint():
	"""
	Publish a polygon marker representing the car's footprint in RViz
	"""
	marker = Marker()
	marker.header.frame_id = "car_2_base_link"  # Car's coordinate frame
	marker.header.stamp = rospy.Time.now()
	marker.ns = "car_footprint"
	marker.id = 0
	marker.type = Marker.LINE_STRIP
	marker.action = Marker.ADD
	
	# Set the scale (line width)
	marker.scale.x = 0.05  # Line thickness
	
	# Set the color (blue)
	marker.color.r = 0.0
	marker.color.g = 0.0
	marker.color.b = 1.0
	marker.color.a = 1.0  # Alpha (transparency)
	
	# Define the rectangle corners (car footprint)
	# Assuming car center is at origin, front is +x direction
	half_length = CAR_LENGTH / 2.0
	half_width = CAR_WIDTH / 2.0
	
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
	
	# Add points to create closed rectangle
	marker.points.append(p1)
	marker.points.append(p2)
	marker.points.append(p3)
	marker.points.append(p4)
	marker.points.append(p1)  # Close the loop
	
	# Set lifetime (0 means forever)
	marker.lifetime = rospy.Duration(0)
	
	car_footprint_pub.publish(marker)

def publish_steering_arrow(steering_angle):
	"""
	Publish an arrow marker showing the steering direction in RViz
	
	Args:
		steering_angle: The steering angle from the controller (in degrees or radians, normalized)
	"""
	marker = Marker()
	marker.header.frame_id = "car_2_base_link"  # Car's coordinate frame
	marker.header.stamp = rospy.Time.now()
	marker.ns = "steering_arrow"
	marker.id = 1
	marker.type = Marker.ARROW
	marker.action = Marker.ADD
	
	# Arrow dimensions
	marker.scale.x = 0.05   # Arrow length
	marker.scale.y = 0.05  # Arrow width
	marker.scale.z = 0.05  # Arrow height
	
	# Set color based on steering direction (green for straight, red for sharp turns)
	marker.color.r = 1
	marker.color.g = 0
	marker.color.b = 0.0
	marker.color.a = 1.0
	
	# Convert steering angle to radians (assuming steering_angle is in range [-100, 100])
	# Map [-100, 100] to approximately [-45, 45] degrees = [-0.785, 0.785] radians
	angle_rad = (steering_angle / 100.0) * (math.pi / 4.0)
	
	# Start point (at front of car)
	start = Point()
	start.x = CAR_LENGTH / 2.0
	start.y = 0.0
	start.z = 0.1  # Slightly above ground
	
	# End point (pointing in steering direction)
	arrow_length = 0.5
	end = Point()
	end.x = start.x + arrow_length * math.cos(angle_rad)
	end.y = start.y + arrow_length * math.sin(angle_rad)
	end.z = start.z
	
	marker.points.append(start)
	marker.points.append(end)
	
	# Set lifetime
	marker.lifetime = rospy.Duration(0.1)  # Update frequently
	
	steering_arrow_pub.publish(marker)

def control(data):
	global prev_error
	global integral_error
	global max_velocity
	global min_velocity
	global velocity_scale_factor
	global kp
	global kd
	global ki

	pid_error = data.pid_error

	print("PID Control Node is Listening to error: %.3f" % pid_error)

	## PID controller implementation
	# Proportional term
	P = kp * pid_error
	
	# Integral term
	integral_error += pid_error
	I = ki * integral_error
	
	# Derivative term
	D = kd * (pid_error - prev_error)
	
	# Update previous error
	prev_error = pid_error

	# Calculate steering angle using PID equation
	angle = P + I + D + servo_offset

	# An empty AckermannDrive message is created. You will populate the steering_angle and the speed fields.
	command = AckermannDrive()

	# Make sure the steering value is within bounds [-100,100]
	if angle > 100:
		angle = 100
	elif angle < -100:
		angle = -100
	
	command.steering_angle = angle

	# Dynamic velocity scaling based on error
	# Use exponential decay for smooth velocity reduction
	# velocity = max_velocity * e^(-scale_factor * |error|^2) + min_velocity
	# This ensures velocity decreases smoothly as error increases
	error_magnitude = abs(pid_error)
	velocity_reduction = math.exp(-velocity_scale_factor * error_magnitude * error_magnitude)
	velocity = min_velocity + (max_velocity - min_velocity) * velocity_reduction
	
	# Alternative linear approach (commented out):
	# velocity = max_velocity - velocity_scale_factor * error_magnitude
	# velocity = max(min_velocity, min(max_velocity, velocity))
	
	# Make sure the velocity is within bounds [0,100]
	if velocity > 100:
		velocity = 100
	elif velocity < 0:
		velocity = 0
	
	command.speed = velocity
	print("Velocity: %.2f" % velocity)

	# Move the car autonomously
	command_pub.publish(command)
	
	# Publish RViz visualization markers
	publish_car_footprint()
	publish_steering_arrow(angle)

if __name__ == '__main__':

    # This code tempalte asks for the values for the gains from the user upon start, but you are free to set them as ROS parameters as well.
	global kp
	global kd
	global ki
	global max_velocity
	global min_velocity
	global velocity_scale_factor
	
	print("=== PID Controller Configuration ===")
	kp = float(input("Enter Kp Value: "))
	kd = float(input("Enter Kd Value: "))
	ki = float(input("Enter Ki Value: "))
	
	print("\n=== Dynamic Velocity Configuration ===")
	#max_velocity = float(input("Enter maximum velocity (straightaways): "))
	#min_velocity = float(input("Enter minimum velocity (turns): "))
	velocity_scale_factor = float(input("Enter velocity scale factor (suggested: 10-50): "))
	
	rospy.init_node('pid_controller', anonymous=True)
    # subscribe to the error topic
	rospy.Subscriber("error", pid_input, control)
	rospy.spin()
