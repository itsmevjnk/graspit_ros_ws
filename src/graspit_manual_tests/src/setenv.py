#!/usr/bin/env python3

import rospy
import rospkg
import yaml

from tf.transformations import quaternion_from_euler

from graspit_interface.srv import *
from geometry_msgs.msg import Pose, Point, Quaternion

GRASPIT_NODE = '/graspit'

def make_pose(obj):
    position = obj['position']
    orientation = obj['orientation']; quaternion = quaternion_from_euler(orientation[0], orientation[1], orientation[2], 'sxyz').tolist()
    rospy.loginfo(f'position: {position}, quaternion: {quaternion}')
    return Pose(
        Point(position[0], position[1], position[2]),
        Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3])
    )

if __name__ == "__main__":
    rospy.init_node('graspit_setenv', anonymous=True)

    with open(rospkg.RosPack().get_path('graspit_manual_tests') + '/world.yml', 'r') as f:
        world = yaml.load(f, yaml.SafeLoader)

    rospy.loginfo('waiting until graspit_interface is available')
    rospy.wait_for_service('/graspit/clearWorld')

    rospy.loginfo('clearing world')
    rospy.ServiceProxy(GRASPIT_NODE + '/clearWorld', ClearWorld)()

    rospy.loginfo('configuring object')
    rospy.ServiceProxy(GRASPIT_NODE + '/importGraspableBody', ImportGraspableBody)(
        world['object']['name'],
        make_pose(world['object'])
    )

    rospy.loginfo('configuring robot')
    rospy.ServiceProxy(GRASPIT_NODE + '/importRobot', ImportRobot)(
        world['robot']['name'],
        make_pose(world['robot'])
    )