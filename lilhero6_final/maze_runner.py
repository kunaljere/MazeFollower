import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import Float32, Float32MultiArray, Int32
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Point, Twist, PoseStamped
from nav_msgs.msg import Odometry
from action_msgs.msg import GoalStatusArray
from cv_bridge import CvBridge

import cv2 as cv
import sys
import csv
import time
import math
import numpy as np


class mazeRunner(Node):

    def __init__(self):

        # Creates the node.
        super().__init__('maze_runner')

        self.head_dist = float('inf')
        self.label = -1
        self.previous_label = -1
        self.thresh_angle = 0.05
        self.head_dist_thresh = 0.5

        self.linear_speed = 0.21
        self.angular_speed = 0.5 

        self.turn_reached = False 

        self.dist_sub = self.create_subscription(Float32, '/head_dist', self.dist_callback, 1)
        self.dist_sub

        self.label_sub = self.create_subscription(Float32, '/label', self.label_callback, 1)
        self.label_sub

        self.odom_sub = self.create_subscription(Odometry, '/odom', self.maze_runner, 1)
        self.odom_sub

        self.classify_img_pub = self.create_publisher(Int32, '/classify_img', 5)
        self.classify_flag = Int32()

        self.state = 'DRIVE'

        self.Init = True
        self.Init_pos = Point()
        self.Init_pos.x = 0.0
        self.Init_pos.y = 0.0
        self.Init_ang = 0.0
        self.globalPos = Point()
        self.globalAng = 0.0
        self.offset_angle = 0.0
    

        self.move_robot_pub = self.create_publisher(Twist, '/cmd_vel', 5)


    def dist_callback(self, msg):

        self.head_dist = msg.data

    def label_callback(self, msg):
        self.label = msg.data


    def maze_runner(self, data):

        self.update_Odometry(data)

        move = Twist()

        #Angle Wrapping
        while self.globalAng < -1*np.pi:
            self.globalAng += 2*np.pi
        while self.globalAng > np.pi:
            self.globalAng -= 2*np.pi

        # print('Head Distance:', self.head_dist)
        # print('Global Angle', self.globalAng)
        

        # Condition during 90 degree turn to stop image classification and stay in turn state
        if self.label >=1 and self.label <=4:
            self.previous_label = self.label
            self.classify_flag.data = 0
            self.classify_img_pub.publish(self.classify_flag)
        elif (self.label == -1 and self.state == 'TURN'):
            self.label = self.previous_label
        

        # STATE MACHINE

        # GOAL REACHED STATE
        if self.label == 5.0:
            self.shutdown_motors()
            time.sleep(50000)

        elif self.label == 0:
            self.turn_reached = False
            self.turn_reached = self.correct_offset()
            if self.turn_reached:
                self.offset_angle = 0
                self.state = 'TURN'
                self.turn_reached = False
                self.Init = True

        elif self.state == "DRIVE":

            print('Global Angle', self.globalAng)

            if self.head_dist > self.head_dist_thresh:
                
                #Making sure the TurtleBot drives on a straight path
                if self.globalAng > 0 and self.globalAng%(np.pi/2) > self.thresh_angle:
                    #turn right 
                    print("correcting right")
                    angular_vel = -self.angular_speed
                elif self.globalAng < 0 and -1*(self.globalAng%(np.pi/2)) < -self.thresh_angle:
                    #turn left
                    print("correcting left")
                    angular_vel = self.angular_speed
                else:
                    print("driving straight")
                    angular_vel = 0.0
                    
                linear_vel = self.linear_speed
                self.classify_flag.data = 0 # Robot will not read label yet

            elif self.head_dist <= (self.head_dist_thresh + 0.05):
                self.classify_flag.data = 1 # Robot will read label
                linear_vel = 0.0
                angular_vel = 0.0

                if self.label > -1:
                    self.state = 'TURN'


            self.classify_img_pub.publish(self.classify_flag)
            move.linear.x = linear_vel
            move.angular.z = angular_vel
            self.move_robot_pub.publish(move)
            print()

        # TURN STATE, RIGHT, LEFT, 180, or STOP
        elif self.state == 'TURN':

            self.classify_flag.data = 0
            self.classify_img_pub.publish(self.classify_flag)

            if self.turn_reached == False:
                print('Label:', self.label)

                if self.label == 1.0:   # LEFT
                    self.turn_reached = self.turn_left()
                    if self.turn_reached:
                        if self.globalAng > 0:
                            self.offset_angle += self.globalAng%np.pi/2
                        else: 
                            self.offset_angle -= self.globalAng%np.pi/2
                        self.state = 'DRIVE'
                        self.turn_reached = False
                        self.Init = True #Resetting global angle to 0.0

                elif self.label == 2.0: # RIGHT
                    self.turn_reached = self.turn_right()
                    if self.turn_reached:
                        if self.globalAng > 0:
                            self.offset_angle += self.globalAng%np.pi/2
                        else: 
                            self.offset_angle -= self.globalAng%np.pi/2
                        self.state = 'DRIVE'
                        self.turn_reached = False
                        self.Init = True
                        
                elif self.label == 3.0: # TURN 180
                    self.turn_reached = self.turn_180()
                    if self.turn_reached:
                        if self.globalAng > 0:
                            self.offset_angle += self.globalAng%np.pi/2
                        else: 
                            self.offset_angle -= self.globalAng%np.pi/2
                        self.state = 'DRIVE'
                        self.turn_reached = False
                        self.Init = True

                elif self.label == 4.0: # STOP
                    self.turn_reached = self.turn_180()
                    if self.turn_reached:
                        if self.globalAng > 0:
                            self.offset_angle += self.globalAng%np.pi/2
                        else: 
                            self.offset_angle -= self.globalAng%np.pi/2
                        self.state = 'DRIVE'
                        self.turn_reached = False
                        self.Init = True



    def turn_left(self):
        while self.globalAng < -1*np.pi:
            self.globalAng += 2*np.pi
        while self.globalAng > np.pi:
            self.globalAng -= 2*np.pi

        goal_angle = np.pi/2 - 0.05

        print('Angle to Turn To: ', goal_angle)

        move = Twist()
        if self.globalAng < goal_angle:
            move.angular.z = 0.5
            self.move_robot_pub.publish(move)
            return False
        else:
            move.angular.z = 0.0
            self.move_robot_pub.publish(move)

            print('starting timer')
            time.sleep(0.5)
            print('ending timer')

            self.state = 'DRIVE'
            return True


    def turn_right(self):
        while self.globalAng < -1*np.pi:
            self.globalAng += 2*np.pi
        while self.globalAng > np.pi:
            self.globalAng -= 2*np.pi


        goal_angle = -np.pi/2 + 0.05

        print('Angle to Turn To: ', goal_angle)

        move = Twist()
        if self.globalAng > goal_angle:
            move.angular.z = -0.5
            self.move_robot_pub.publish(move)
            return False
        else:
            move.angular.z = 0.0
            self.move_robot_pub.publish(move)

            print('starting timer')
            time.sleep(0.5)
            print('ending timer')

            self.state = 'DRIVE'
            return True


    def turn_180(self):
        while self.globalAng < -1*np.pi:
            self.globalAng += 2*np.pi
        while self.globalAng > np.pi:
            self.globalAng -= 2*np.pi


        goal_angle = np.pi - 0.05

        print('Angle to Turn To: ', goal_angle)

        move = Twist()
        if self.globalAng < goal_angle:
            move.angular.z = 0.5
            self.move_robot_pub.publish(move)
            return False
        else:
            move.angular.z = 0.0
            self.move_robot_pub.publish(move)

            print('starting timer')
            time.sleep(0.5)
            print('ending timer')

            self.state = 'DRIVE'
            return True


    def correct_offset(self):
        while self.globalAng < -1*np.pi:
            self.globalAng += 2*np.pi
        while self.globalAng > np.pi:
            self.globalAng -= 2*np.pi

        print('Offset Angle to turn to: ', self.offset_angle)

        move = Twist()
        if self.offset_angle < self.globalAng:
            move.angular.z = 0.5
            self.move_robot_pub.publish(move)
            return False
        else:
            move.angular.z = 0.0
            self.move_robot_pub.publish(move)

            print('starting timer')
            time.sleep(0.5)
            print('ending timer')

            self.state = 'DRIVE'
            return True

    '''
    Auxiliary
    '''
    def update_Odometry(self,Odom):
        position = Odom.pose.pose.position
        
        #Orientation uses the quaternion aprametrization.
        #To get the angular position along the z-axis, the following equation is required.
        q = Odom.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))

        if self.Init:
            #The initial data is stored to by subtracted to all the other values as we want to start at position (0,0) and orientation 0
            self.Init = False
            self.Init_ang = orientation
            self.globalAng = self.Init_ang
            Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        
            self.Init_pos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y
            self.Init_pos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y
            self.Init_pos.z = position.z
        Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        

        #We subtract the initial values
        self.globalPos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y - self.Init_pos.x

        self.globalPos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y - self.Init_pos.y
        self.globalAng = orientation - self.Init_ang   
    
        # self.get_logger().info('Transformed global pose is x:{}, y:{}, a:{}'.format(self.globalPos.x,self.globalPos.y,self.globalAng))

    
    def shutdown_motors(self):
        t = Twist()
        self.move_robot_pub.publish(t)



def main():
    rclpy.init() #init routine needed for ROS2.
    maze_runner = mazeRunner() #Create class object to be used.
    
    rclpy.spin(maze_runner) # Trigger callback processing.

    #Clean up and shutdown.
    maze_runner.destroy_node()  
    rclpy.shutdown()


if __name__ == '__main__':
	main()