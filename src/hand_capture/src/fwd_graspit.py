#!/usr/bin/env python3

# forward joint angles to GraspIt

import rospy

from graspit_interface.srv import *
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import JointState

rospy.init_node('fwd_graspit', anonymous=True)

rospy.loginfo('waiting for GraspIt interface')
rospy.wait_for_service('/graspit/clearWorld')

first_frame = True
setDOF = rospy.ServiceProxy('/graspit/forceRobotDof', ForceRobotDOF) # TODO: maybe use setting for this?

rospy.loginfo('listening to hand pose topic')
def frame_cb(data):
    global first_frame
    if first_frame or data.header.seq == 0:
        first_frame = False
        rospy.loginfo('setting up environment')
        rospy.ServiceProxy('/graspit/clearWorld', ClearWorld)()
        rospy.ServiceProxy('/graspit/importRobot', ImportRobot)(
            'HumanHand20DOF',
            Pose(
                Point(0.0, 0.0, 0.0), # position
                Quaternion(0.0, 0.0, 0.0, 1.0) # orientation (quaternion)
            )
        )
    # rospy.loginfo('Received hand pose:', data.position)
    setDOF(0, data.position)
rospy.Subscriber('/hand_capture/joints', JointState, frame_cb)

rospy.spin()