import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Point, Twist
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

import sys
import time
import math
import numpy as np
import cv2
from cv_bridge import CvBridge

class getLidarData(Node):

    def __init__(self):

        # Creates the node.
        super().__init__('object_range')

        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos_profile.durability = QoSDurabilityPolicy.VOLATILE

        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, qos_profile)
        self.lidar_sub

        self.publish_distance = self.create_publisher(Float32, '/head_dist', 1)

        self.distance = Float32()


    def lidar_callback(self, msg):
        ranges = msg.ranges
        
        #Publishing distance at angle 0
        
        self.distance.data = ranges[0]

        self.publish_distance.publish(self.distance)



def main():
    rclpy.init() #init routine needed for ROS2.
    get_lidar_data = getLidarData() #Create class object to be used.

    rclpy.spin(get_lidar_data) # Trigger callback processing.		

    #Clean up and shutdown.
    get_lidar_data.destroy_node()  
    rclpy.shutdown()


if __name__ == '__main__':
	main()