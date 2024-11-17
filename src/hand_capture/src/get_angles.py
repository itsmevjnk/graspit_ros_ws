#!/usr/bin/env python3

import cv2
import mediapipe as mp
import numpy as np
import time
import math

import rospy
import rospkg

from cv_bridge import CvBridge, CvBridgeError

from std_msgs.msg import Header
from sensor_msgs.msg import JointState, Image

rospy.init_node('get_angles', anonymous=True)

basepath = rospkg.RosPack().get_path('hand_capture')

import matplotlib.pyplot as plt

# cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

def make_vect(lm):
    # print(lm)
    return np.array([lm.x, lm.y, lm.z])

# https://stackoverflow.com/a/59204638
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def normalise_vect(a, axis = -1, order = 2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return (a / np.expand_dims(l2, axis)).flatten()

# build hand coordinate frame
def build_frame(hand, left = True):
    wrist = hand.landmark[0]; wristCV = make_vect(wrist) # CV = coordinate vector
    pinkyMCP = hand.landmark[17]; pinkyCV = make_vect(pinkyMCP); pinkyVect = pinkyCV - wristCV
    indexMCP = hand.landmark[5]; indexCV = make_vect(indexMCP) ; indexVect = indexCV - wristCV
    
    if left: zVect = np.cross(indexVect, pinkyVect) # Z vector originating from wrist, positive = palm side
    else: zVect = np.cross(pinkyVect, indexVect)
    zVect_norm = normalise_vect(zVect)
    xVect = normalise_vect(pinkyVect) + normalise_vect(indexVect); xVect_norm = normalise_vect(xVect) # X vector is angle bisector of pinky and index knuckle vectors
    yVect = yVect_norm = np.cross(zVect_norm, xVect_norm) # Y vector is determined from Z and X

    return np.matrix(np.column_stack((np.row_stack([xVect_norm, yVect_norm, zVect_norm, wristCV]), [0, 0, 0, 1]))).T

std_norm = np.load(basepath + '/palmnorm.npy') # standard (GraspIt) palm plane normal
rot_matrix = np.row_stack([
    np.column_stack([
        rotation_matrix_from_vectors(np.array([0.,0.,1.]), std_norm), # rotation matrix aligning Z axis to palm plane normal (for rotation)
        np.zeros(3)
    ]),
    [0.,0.,0.,1.]
]) # expand into 4x4 transformation matrix

def get_xy(width, height, pt):
    if pt.ndim > 1: pt = np.array(pt).flatten()
    x = pt[0:1].item()
    y = pt[1:2].item()
    tup = (int(width * x), int(height * y))#; print(tup)
    return tup

def draw_vector(img, origin, vect, color = (0, 255, 0), length = 0.25, thickness = 2):
    # rospy.loginfo(f'{origin} -> {vect}')
    endpoint = origin + length * normalise_vect(vect[:3])
    height, width, _ = img.shape
    cv2.arrowedLine(img, get_xy(width, height, origin), get_xy(width, height, endpoint), color, thickness)

def get_frame_origin(frame):
    return frame[:3,3].flatten()

def draw_frame(img, frame, length = 0.25, thickness = 2, ptO = None):
    if ptO is None: ptO = get_frame_origin(frame)

    draw_vector(img, ptO, frame[:3,0].flatten(), (255, 0, 0), length, thickness)
    draw_vector(img, ptO, frame[:3,1].flatten(), (0, 255, 0), length, thickness)
    draw_vector(img, ptO, frame[:3,2].flatten(), (0, 0, 255), length, thickness)

def vect_angle(a, b):
    return np.arccos(a.dot(b) / (np.sqrt(a.dot(a)) * np.sqrt(b.dot(b))))

def rotateX(ang):
    s, c = np.sin(ang), np.cos(ang)
    return np.matrix([
        [1., 0., 0.],
        [0., c, -s],
        [0., s, c, 0.]
    ])

def rotateY(ang):
    s, c = np.sin(ang), np.cos(ang)
    return np.matrix([
        [c, 0., s],
        [0., 1., 0.],
        [-s, 0., c]
    ])

def rotateZ(ang):
    s, c = np.sin(ang), np.cos(ang)
    return np.matrix([
        [c, -s, 0.],
        [s, c, 0.],
        [0., 0., 1.]
    ])

def rotateX4(ang):
    s, c = np.sin(ang), np.cos(ang)
    return np.matrix([
        [1., 0., 0., 0.],
        [0., c, -s, 0.],
        [0., s, c, 0.,],
        [0., 0., 0., 1.]
    ])

def rotateY4(ang):
    s, c = np.sin(ang), np.cos(ang)
    return np.matrix([
        [c, 0., s, 0.],
        [0., 1., 0., 0.],
        [-s, 0., c, 0.],
        [0., 0., 0., 1.]
    ])

def rotateZ4(ang):
    s, c = np.sin(ang), np.cos(ang)
    return np.matrix([
        [c, -s, 0., 0.],
        [s, c, 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]
    ])

rospy.loginfo('setting up joint state publisher')
joints_pub = rospy.Publisher('/hand_capture/joints', JointState, queue_size=10)

joints_offset = np.append(np.zeros(16), np.radians([0.0, 0.0, 0.0, 0.0])) # TODO: thumb offsets

rospy.loginfo('creating annotated frame topic')
frame_pub = rospy.Publisher('/hand_capture/frame', Image, queue_size=10)

rospy.loginfo('subscribing to camera topic')
bridge = CvBridge()
seq = 0
# while True:
def frame_cb(data):
    img = bridge.imgmsg_to_cv2(data, 'rgb8')
    # _, img = cap.read()
    img = cv2.flip(img, 1)
    
    results = hands.process(img)
    handedness = results.multi_handedness
    # if handedness is not None: print(handedness)
    
    if results.multi_hand_landmarks:
        for iHand, hand in enumerate(results.multi_hand_landmarks):
            left = handedness[iHand].classification[0].label == 'Left'
            if left: continue # TODO

            handFrame = build_frame(hand, left) # build hand frame
            wrist = make_vect(hand.landmark[0])
            landmarksHF = landmarksHF_rot = np.matmul(np.linalg.inv(handFrame), np.array([np.append(make_vect(hand.landmark[i]) - wrist, [1.]) for i in range(21)]).T) # landmarks in hand frame
            # landmarksHF_rot = np.matmul(rot_matrix, landmarksHF) # rotate to align palm planes
            # print(landmarksHF_rot)

            def calculate_joint_angles(mcpIndex):
                # MCP-PIP
                mcp_pip = np.asarray(landmarksHF_rot[0:3,mcpIndex+1] - landmarksHF_rot[0:3,mcpIndex]).reshape(-1)
                # print(mcp_pip)
                mcpAbduct = np.arctan(mcp_pip[1] / mcp_pip[0]) # y / x
                mcpFlex = np.arctan(mcp_pip[2] / np.sqrt(mcp_pip[0] ** 2 + mcp_pip[1] ** 2)) # z / sqrt(x^2+y^2)

                # PIP-DIP
                pip_dip = np.asarray(landmarksHF_rot[0:3,mcpIndex+2] - landmarksHF_rot[0:3,mcpIndex+1]).reshape(-1)
                pip = vect_angle(pip_dip, mcp_pip)

                # DIP-tip
                dip_tip = np.asarray(landmarksHF_rot[0:3,mcpIndex+3] - landmarksHF_rot[0:3,mcpIndex+2]).reshape(-1)
                dip = vect_angle(dip_tip, pip_dip)

                return [mcpAbduct, mcpFlex, pip, dip]
            
            joints = np.array([calculate_joint_angles(i) for i in [5, 9, 13, 17]]).flatten() # joint positions

            # calculate thumb
            cmcIndex = 1
            cmc_wrist = -np.asarray(landmarksHF_rot[0:3,cmcIndex]).reshape(-1) # CMC-wrist
            cmc_mcp = np.asarray(landmarksHF_rot[0:3,cmcIndex+1] - landmarksHF_rot[0:3,cmcIndex]).reshape(-1) # CMC-MCP
            thumb_norm = np.cross(cmc_wrist, cmc_mcp) # thumb plane normal
            cmc_flex = vect_angle(thumb_norm, np.array([0., 0., 1.]))

            # cmc_mcp = np.asarray(landmarksHF_rot[0:3,cmcIndex+1] - landmarksHF_rot[0:3,cmcIndex]).reshape(-1) # CMC-MCP
            # cmc_mcp[0] = 0 # X = 0 (project onto YZ plane)
            # cmc_flex = vect_angle(cmc_mcp, np.array([0., -1., 0.]))

            thumbFrame_tf = np.matmul(handFrame, np.matmul(rotateX4(np.degrees(180) - cmc_flex), np.linalg.inv(handFrame))) # transformation matrix from hand frame to thumb frame
            
            # thumbFrame = np.matmul(handFrame, rotateX4(np.degrees(180) - cmc_flex)) # HF * Rx * HF^-1 * HF (Thm 1.5)
            # draw_frame(img, thumbFrame, 0.15, ptO = make_vect(hand.landmark[cmcIndex]))
            
            #cmcPoint = make_vect(hand.landmark[cmcIndex])
            #draw_vector(img, cmcPoint, cmc_mcp, (255, 255, 0))
            # draw_vector(img, cmcPoint, zVect, (0, 255, 255))
            #draw_vector(img, cmcPoint, np.array(np.matmul(handFrame, np.append(thumb_norm, [1.]))).flatten(), (255, 0, 255))            

            joints = np.append(joints, [cmc_flex, 0.0, 0.0, 0.0]) # TODO: thumb

            joints = (joints + joints_offset).tolist()
            rospy.loginfo(f'Thumb: {" ".join([str(math.degrees(j)) for j in joints[-4:]])}')

            global seq
            joints_pub.publish(JointState(
                Header(seq, rospy.Time.now(), ''),
                '',
                joints,
                [], []
            ))
            seq += 1

            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

            draw_frame(img, handFrame)

    # cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    frame_pub.publish(bridge.cv2_to_imgmsg(img, 'rgb8'))    

rospy.Subscriber('/camera/image_raw', Image, frame_cb)

rospy.spin()