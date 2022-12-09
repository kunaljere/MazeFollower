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

class classifyImage(Node):

    def __init__(self):

        # Creates the node.
        super().__init__('classify_image')

        # self.imageDirectory = './images/'
        # self.imageDirectory = '/home/ubuntu/Documents/ros2_ws/src/lilhero6_final/lilhero6_final/images/'
        self.imageDirectory = '/home/burger/turtlebot3_ws/src/lilhero6_final/lilhero6_final/images/'
        self.imread_mode = 1 # 0 for bw

        ## knn params
        self.k = 5
        self.dim = 3 # image channel

        self.morphKernel = 5  # Closing and Opening Kernel Size
        self.maxObjects = 1  # Max number of object to detect.
        self.minObjectArea = 300  # Min number of pixels for an object to be recognized.


        ## Resize parameters
        self.raw_w = 410 # images are 410 x 308 pixels
        self.raw_h = 308

        self.resize_ratio = 0.2
        self.resize_h = int(self.raw_h * self.resize_ratio)
        self.resize_w = int(self.raw_w * self.resize_ratio)

        ## Rgb filter
        self.light_thresh = 60
        self.contrast = 30

        self.debug = False


        self._video_subscriber = self.create_subscription(
				CompressedImage,
				'/camera/image/compressed',
				self.image_callback,
				1)
        self._video_subscriber # Prevents unused variable warning.

        
        # self.publish_object_center = self.create_publisher(Float32, '/center', 5)
        # self.center = Float32()


        self.publish_label = self.create_publisher(Float32, '/label', 5)
        self.label = Float32()

        self.classify_img_sub = self.create_subscription(Int32, '/classify_img', self.classify_callback, 1)
        self.classify_img_sub

        self.classify_flag = -1



    def findObjects(self, image):
        # Finds the location of the desired object in the image.
        contours,_ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Contours the image to find blobs of the same color
        cont = sorted(contours, key=cv.contourArea, reverse=True)[:self.maxObjects]  # Sorts the blobs by size (Largest to smallest)
        x = y = w = h = 1
        # Find the center of mass of the blob if there are any
        if len(cont) > 0:
            M = cv.moments(cont[0])
            if M['m00'] > self.minObjectArea:
                # print(M['m00'])
                x, y, w, h = cv.boundingRect(cont[0])

        return x, y, w, h


    def morphOps(self, image, kernelSize):

        kernel = np.ones((kernelSize, kernelSize), np.uint8)
        # highlight the edge
        element2 = cv.getStructuringElement(cv.MORPH_RECT, (kernelSize, kernelSize))
        fix = cv.dilate(image, element2, iterations=1)
        # Fill in holes
        fix = cv.morphologyEx(fix, cv.MORPH_CLOSE, kernel)
        return fix



    def rgbThreshold(self, img, upper, fill=None):

        if fill is None:
            fill = [255, 255, 255] #white 
        if len(img.shape) == 2:
            return img

        h,w,_ = img.shape
        for i in range(h):
            for j in range(w):
                if img[i][j][0] > upper and img[i][j][1] > upper and img[i][j][2] > upper:
                    img[i][j] = fill
                elif np.var(np.array(img[i][j])) < self.contrast:
                    img[i][j] = fill      
        return img



    def preprocess(self, img):
        # downsample to speed up
        img = cv.resize(img, (int(0.3*self.raw_w), int(0.3*self.raw_h)))
        # filter the background and noise
        img_rgb = self.rgbThreshold(img, self.light_thresh)

        # detect edges
        canny_img = cv.Canny(img_rgb, 100, 150)

        # morphology
        morph_img = self.morphOps(canny_img, self.morphKernel)

        mask = np.ones((img.shape[0], img.shape[1]), np.uint8)

        mask[0:20, 0:123] = 0
        mask[75:92, 0:123] = 0

        masked_img = cv.bitwise_and(morph_img, morph_img, mask=mask)

        # detect the object
        x, y, w, h = self.findObjects(masked_img)

        # crop the original image to the object area
        img = img_rgb[y:y + h, x:x + w]

        img = cv.resize(img, (self.resize_w, self.resize_h))

        return img



    def train(self):

        with open(self.imageDirectory + 'train.txt', 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)

        # this line reads in all images listed in the file in COLOR and does preprocessing function 
        train = np.array([np.array(self.preprocess(cv.imread(self.imageDirectory +lines[i][0]+".png",self.imread_mode))) for i in range(len(lines))])

        # here we reshape each image into a long vector and ensure the data type is a float (which is what KNN wants)
        train_data = train.flatten().reshape(len(train),self.resize_w*self.resize_h*self.dim)
        train_data = train_data.astype(np.float32)

        # read in training labels
        train_labels = np.array([np.int32(lines[i][1]) for i in range(len(lines))])


        ### Train classifier
        knn = cv.ml.KNearest_create()
        knn.train(train_data, cv.ml.ROW_SAMPLE, train_labels)
        knn.save("knnModel") #save as file that can be read



    def classify_callback(self, msg):
        self.classify_flag = msg.data


    def image_callback(self, CompressedImage):

        imgBGR = CvBridge().compressed_imgmsg_to_cv2(CompressedImage, "bgr8")

        if self.classify_flag == 1:
            img = cv.resize(imgBGR, (308, 410), interpolation=cv.INTER_LINEAR)
            self.label.data = self.classify_raspi_image(img)
            print(self.label.data)

            self.publish_label.publish(self.label)

        else:
            self.label.data = -1.0
            self.publish_label.publish(self.label)



    def classify_raspi_image(self, original_img):
        # load the model
        knn_test = cv.ml.KNearest_create()
        model = knn_test.load("knnModel")

        test_img = np.array(self.preprocess(original_img))
        test_img = test_img.flatten().reshape(1,self.resize_w*self.resize_h*self.dim)

        test_img = test_img.astype(np.float32)

        ret, results, neighbours, dist = model.findNearest(test_img, self.k)

        return ret

    

    #only use for training new model
    def classify_test_images(self):

        self.train()

        # load the model
        knn_test = cv.ml.KNearest_create()
        model = knn_test.load("knnModel")

        # load test set
        with open(self.imageDirectory + 'test.txt', 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)

        correct = 0.0
        confusion_matrix = np.zeros((6,6))

        for i in range(len(lines)):
            original_img = cv.imread(self.imageDirectory+ lines[i][0] + ".png", self.imread_mode)
            test_img = np.array(self.preprocess(original_img))

            test_img = test_img.flatten().reshape(1,self.resize_w*self.resize_h*self.dim)
            test_img = test_img.astype(np.float32)

            test_label = np.int32(lines[i][1])


            ret, results, neighbours, dist = model.findNearest(test_img, self.k)

            if test_label == ret:
                print(str(lines[i][0]) + " Correct, " + str(ret))
                correct += 1
                confusion_matrix[np.int32(ret)][np.int32(ret)] += 1
            else:
                confusion_matrix[test_label][np.int32(ret)] += 1

                print(str(lines[i][0]) + " Wrong, " + str(test_label) + " classified as " + str(ret))
                print("\tneighbours: " + str(neighbours))
                print("\tdistances: " + str(dist))
            
        print(confusion_matrix)

        


def main():
    rclpy.init() #init routine needed for ROS2.
    classify_image = classifyImage() #Create class object to be used.
    
    # Comment out next line for training new model
    rclpy.spin(classify_image) # Trigger callback processing.

    # Uncomment next line for training new model
    # rclpy.spin(classify_image.classify_test_images()) # Trigger callback processing.		

    #Clean up and shutdown.
    classify_image.destroy_node()  
    rclpy.shutdown()


if __name__ == '__main__':
	main()