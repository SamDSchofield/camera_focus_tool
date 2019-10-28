#!/usr/bin/env python
"""ROS node for estimating the sharpness of an image.

Copyright (c) 2019. Sam Schofield. This file is subject to the 3-clause BSD
license, as found in the LICENSE file in the top-level directory of this
distribution and at https://github.com/sds53/camera_focus_tool/LICENSE.
No part of camera_focus_tool, including this file, may be copied, modified,
propagated, or distributed except according to the terms contained in the
LICENSE file.
"""
from __future__ import print_function
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge, CvBridgeError
import argparse
from dynamic_reconfigure.server import Server
from camera_focus_tool.cfg import FocusToolConfig


def calculate_focus_laplace(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def percents_to_pixels(image_width, image_height, row_percent, col_percent, width_percent, height_percent):
    """
    Convert from percents of image to pixel values
    Args:
        image_width:
        image_height:
        row_percent:
        col_percent:
        width_percent:
        height_percent:

    Returns:
        (int, int, int, int): row, col, height width in pixels.
    """
    row = int(image_height * row_percent)
    col = int(image_width * col_percent)
    width = int(image_width * width_percent)
    height = int(image_height * height_percent)
    return row, col, height, width


def draw_roi_fde(image, output_image, row_percent, col_percent, width_percent, height_percent, color=(0, 0, 255)):
    row, col, height, width = percents_to_pixels(
        image.shape[1], image.shape[0],
        row_percent, col_percent, width_percent, height_percent
    )
    pt1 = (col, row)
    pt2 = col + width, row + height
    cv2.rectangle(output_image, pt1, pt2, color, thickness=5)

    roi = image[row:row + height, col:col + width]
    lap_var = calculate_focus_laplace(roi)
    text = "fde: {0}".format(np.sum(lap_var))
    cv2.putText(output_image, text, (col + width, row + height), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=color,
                thickness=2)
    return output_image, lap_var


class CameraFocus:
    def __init__(self, topic):
        self.topic = topic
        self.row1_percent = None
        self.col1_percent = None
        self.width1_percent = 0
        self.height1_percent = 0

        self.row2_percent = None
        self.col2_percent = None
        self.width2_percent = 0
        self.height2_percent = 0

        self.image_pub = rospy.Publisher("camera_focuser/image", Image, queue_size=1)
        self.var_pub1 = rospy.Publisher("camera_focuser/full_laplace_var", Float32, queue_size=1)
        self.var_pub2 = rospy.Publisher("camera_focuser/roi_laplace_var1", Float32, queue_size=1)
        self.var_pub3 = rospy.Publisher("camera_focuser/roi_laplace_var2", Float32, queue_size=1)

        self.reconfigure_server = Server(FocusToolConfig, self.reconfigure_callback)
        self.windowNameOrig = "Camera: {0}".format(self.topic)
        cv2.namedWindow(self.windowNameOrig, 2)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(self.topic, Image, self.callback)

    def reconfigure_callback(self, config, _):
        self.width1_percent = config.width_percent_1
        self.height1_percent = config.height_percent_1
        self.row1_percent = config.row_percent_1
        self.col1_percent = config.col_percent_1

        self.width2_percent = config.width_percent_2
        self.height2_percent = config.height_percent_2
        self.row2_percent = config.row_percent_2
        self.col2_percent = config.col_percent_2
        return config

    def callback(self, msg):
        # convert image to opencv
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            np_image = np.array(cv_image)
        except CvBridgeError, e:
            print("Could not convert ros message to opencv image: ", e)
            return

        output_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2BGR)

        lap_var = calculate_focus_laplace(np_image)
        var1_msg = Float32(lap_var)
        self.var_pub1.publish(var1_msg)
        text = "Laplacian var: {0} (Higher the better)".format(np.sum(lap_var))
        cv2.putText(output_image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 255), thickness=2)

        if self.width1_percent != 0 and self.height1_percent != 0:
            output_image, var = draw_roi_fde(np_image, output_image, self.row1_percent, self.col1_percent,
                                        self.width1_percent, self.height1_percent)
            var1_msg = Float32(var)
            self.var_pub2.publish(var1_msg)

        if self.width2_percent != 0 and self.height2_percent != 0:
            output_image, var = draw_roi_fde(np_image, output_image, self.row2_percent, self.col2_percent,
                                        self.width2_percent, self.height2_percent, (0, 255, 0))
            var1_msg = Float32(var)
            self.var_pub3.publish(var1_msg)

        image_msg = self.bridge.cv2_to_imgmsg(output_image, encoding="bgr8")
        self.image_pub.publish(image_msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate the intrinsics of a camera.')
    parser.add_argument('--topic', nargs='+', dest='topics', help='camera topic', required=True)
    parsed = parser.parse_args()

    rospy.init_node('camera_focuser', anonymous=True)

    for topic in parsed.topics:
        camval = CameraFocus(topic)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

