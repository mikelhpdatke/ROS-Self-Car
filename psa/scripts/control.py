#! /usr/bin/python
# Copyright (c) 2015, Rethink Robotics, Inc.

# Using this CvBridge Tutorial for converting
# ROS images to OpenCV2 images
# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

# Using this OpenCV2 tutorial for saving Images:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html

# rospy for the subscriber
import rospy
import math
import ImgProcessing
from ImgProcessing import *
import os
# ROS Image message
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2

class CarControl:
    def __init__(self):
        self.x = 120
        self.y = 300
        self.angle = 0
        self.pubSteer = rospy.Publisher('/Team1_steerAngle', Float32, queue_size=10)
        self.pubSpeed = rospy.Publisher('/Team1_speed', Float32, queue_size=10)

    def errorAngle(self, nextX, nextY):
        x1 = nextX
        y1 = nextY
        if (self.x == x1):
            return 0
        if (self.y == y1):
            return -90 if (x1 < self.x) else 90
        pi = math.acos(-1.0)
        dx = x1 - self.x
        dy = self.y - y1
        if (dx < 0):
            return (-1) * math.atan(-dx / dy) * 180 / pi
        else:
            return math.atan(dx / dy) * 180 / pi

    def driver(self, arrLeft, arrRight, speed):
        n = len(arrLeft)
        m = len(arrRight)
        pos = n - 10
        error = 0
        if (pos >= 0):
            midX = (arrLeft[pos][0] + arrRight[pos][0]) / 2
            midY = (arrLeft[pos][1] + arrRight[pos][1]) / 2
            error = self.errorAngle(midX, midY)
            print "Debug: %f %f %f" % (midX, midY, error)
        else:
            '''
            Ko detect duoc
            '''
            print("err")
            self.angle = 0
        self.angle = error
        rr = Float32(50)
        self.pubSpeed.publish(rr)
        self.pubSteer.publish(self.angle)
        #print "Publish ..%f" %(self.angle)

# Instantiate CvBridge
bridge = CvBridge()
countImg = 0

def image_callback(msg):
    global countImg
    print("Received an image!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "rgb8")
        arr_left, arr_right, globalImg = processImg(cv2_img)
        car.driver(arr_left, arr_right, 20)
    except CvBridgeError, e:
        print(e)
    else:
        countImg = countImg + 1
        if not os.path.exists('/home/ais/images'):
            os.makedirs('/home/ais/images')
        cv2.imwrite('/home/ais/images/' + str(countImg) + '.jpeg', cv2_img)
        #print(cv2)
        '''
        (countImg) = (countImg) + 1
        cv2.namedWindow('Original')
        cv2.imshow('Original', cv2_img)
        cv2.waitKey(1)

        cv2.namedWindow('threshholded')
        img = thresholder.threshold(cv2_img)
        cv2.imshow('threshholded', img)
        cv2.waitKey(1)

        cv2.namedWindow('warperd')
        img = warper.warp(img)
        cv2.imshow('warperd', img)
        cv2.waitKey(1)
        
        cv2.namedWindow('final')
        cv2.imshow('final', globalImg)
        cv2.waitKey(1)
        '''
        

def main():
    rospy.init_node('image_listener')
    # Define your image topic
    image_topic = "image_decompressed"
    # Set up your subscriber and define its callback
    #cv2.namedWindow('Parameter-Trackbars')
    
    rospy.Subscriber(image_topic, Image, image_callback)
    #rospy.Subscriber(image_topic, Image, showImage)
    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    globalImg = None

    car = CarControl()
    main()