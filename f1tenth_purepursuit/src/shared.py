from ackermann_msgs.msg import AckermannDrive
from sensor_msgs.msg import LaserScan


# globals variables used to coordinate
pure_pursuit_recent_cmd = AckermannDrive()
follow_gap_recent_cmd = AckermannDrive()
drive_mode = "pure_pursuit"  # default mode