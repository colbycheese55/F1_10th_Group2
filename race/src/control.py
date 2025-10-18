#!/usr/bin/env python
import math
import rospy
from race.msg import pid_input
from ackermann_msgs.msg import AckermannDrive

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
