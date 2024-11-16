#!/usr/bin/env python3

import rospy
import yaml
from datetime import datetime

from tf.transformations import euler_from_quaternion

from graspit_interface.srv import *
from geometry_msgs.msg import Pose, Point, Quaternion

GRASPIT_NODE = '/graspit'

def extract_pose(pose):
    position = pose.position; quaternion = pose.orientation
    return {
        'position': [position.x, position.y, position.z],
        'orientation': list(euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w], 'sxyz'))
    }

if __name__ == "__main__":
    rospy.init_node('graspit_dump', anonymous=True)

    rospy.loginfo('waiting until graspit_interface is available')
    rospy.wait_for_service('/graspit/clearWorld')

    world = dict()
    
    
    rospy.loginfo('getting object pose')
    get_body = rospy.ServiceProxy(GRASPIT_NODE + '/getBody', GetBody) # will be reused
    world['object'] = extract_pose(get_body(0).body.pose)

    rospy.loginfo('getting robot pose')
    robot = rospy.ServiceProxy(GRASPIT_NODE + '/getRobot', GetRobot)(0)
    world['robot'] = extract_pose(robot.robot.pose)
    world['robot']['dof'] = robot.robot.dofs
    
    world['robot']['bodies'] = []
    i = 1
    while True:
        body = get_body(i)
        if body.result != 0: break
        rospy.loginfo(f'getting robot body {i} pose')
        world['robot']['bodies'].append(extract_pose(body.body.pose))
        i += 1

    quality = rospy.ServiceProxy(GRASPIT_NODE + '/computeQuality', ComputeQuality)(0)
    world['robot']['quality'] = {
        'result': quality.result,
        'volume': quality.volume,
        'epsilon': quality.epsilon
    }
    rospy.loginfo(f'grasp quality metrics ({quality.result}): v={quality.volume}, e={quality.epsilon}')

    fname = datetime.now().strftime('%Y%m%d_%H%M%S.yml')
    rospy.loginfo(f'saving world as {fname} (object and robot names to be specified)')
    with open(fname, 'w') as f:
        yaml.safe_dump(world, f)