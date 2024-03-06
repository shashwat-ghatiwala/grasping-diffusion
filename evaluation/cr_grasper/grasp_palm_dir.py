"""

Carlyn C. Dougherty
ccd2134@columbia.edu
Robotics Research, Fall 2018

"""


#gripper
#finger hand

#if use GUI in config


#TODO: self collision issues - pubullet
#TODO: figure out another hand
#TODO: simulated annealing (which is better grasps)
#TODO: underactuation???? splaying too far back


########################################################################################################################


import pybullet as p
from math import pi, sqrt
import pybullet_data
from time import sleep, time
import astropy.coordinates
import random
from transforms3d import euler
from configparser import ConfigParser


"""#####################################################################################################################
                                    PYBULLET HOUSEKEEPING + GUI MAINTENANCE 
#####################################################################################################################"""


physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally

# This is to change the visualizer window settings
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, enable=0)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, enable=0)
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, enable=0)
# change init camera distance/location to view scene
p.resetDebugVisualizerCamera(cameraDistance=.7, cameraYaw=135, cameraPitch=-20, cameraTargetPosition=[0.0, 0.0, 0.0])



"""#####################################################################################################################
                                       GLOBAL VARIABLES  - from config file 
#####################################################################################################################"""


config = ConfigParser()
config.read('gripper_config.ini')

# loading
robot_path = config.get('file_paths', 'robot_path')
object_path = config.get('file_paths', 'object_path')
object_scale = config.getfloat('file_paths', 'object_scale')
#object_path = "lego.urdf"

# finding hand positions
# how far from the origin should the hand be at the start (just needs to be beyond the length of the object)
init_grasp_distance = config.getfloat('grasp_settings', 'init_grasp_distance')
speed_find_distance = config.getfloat('grasp_settings', 'speed_find_distance')
# how far from touching do you want the palm to be when attempting grips
grasp_distance_margin = config.getfloat('grasp_settings', 'grasp_distance_margin')

# grasping
# target velocity for joints when grasping
max_grasp_force = config.getfloat('grasp_settings', 'max_grasp_force')
# max force allowed
target_grasp_velocity = config.getfloat('grasp_settings', 'target_grasp_velocity')
# how long given to find a grasp
grasp_time_limit = config.getfloat('grasp_settings', 'grasp_time_limit')
# which joints in the hand to use - specified w/ num in the ObjectURDFs
active_grasp_joints = [int(j.strip()) for j in config.get('grasp_settings', 'active_grasp_joints').split(',')]


#use GUI?
use_gui = config.getboolean('gui_settings', 'use_gui')


"""#####################################################################################################################
                                                    UTILIES
#####################################################################################################################"""


def rand_coord():
    """
    Gets random polar coordinate by generating 2 random #s between -pi/2 and pi/2 to serve as theta and phi
    """
    rand_theta = random.uniform(-pi / 2, pi / 2)
    rand_phi = random.uniform(-pi / 2, pi / 2)
    return rand_theta, rand_phi


"""#####################################################################################################################
                                           ObjectURDFs/OBJECT MANAGEMENT
#####################################################################################################################"""


def reset_hand(rID=None, rPos=(0, 0, -init_grasp_distance), rOr=(0, 0, 0, 1), fixed=True):
    """
    loads a new instance of the hand into the scene - this disrupts all current physics, so "resets" the scene
    if you send in the "handID", it will delete the old instance before populating a new one
    TODO: give the hand some sensitivity for grasping - force sensors for fingertips?
    """
    if rID is not None:
        p.removeBody(rID)
    rID = p.loadURDF(robot_path, basePosition=rPos, baseOrientation=rOr, useFixedBase=fixed, globalScaling=1)

    #p.addUserDebugText("whatevs", [0,0,0.1],textColorRGB=[1,0,0],textSize=1.5,parentObjectUniqueId=hID, parentLinkIndex=1)
    p.addUserDebugLine([0,0,0],[.2,0,0],[1,0,0],parentObjectUniqueId=rID, parentLinkIndex=-1, lineWidth = 500)
    p.addUserDebugLine([0,0,0],[0,.2,0],[0,1,0],parentObjectUniqueId=rID, parentLinkIndex=-1, lineWidth = 500)
    p.addUserDebugLine([0,0,0],[0,0,.2],[0,0,1],parentObjectUniqueId=rID, parentLinkIndex=-1, lineWidth = 500)

    return rID


def reset_ob(oID=None, oPos=(0, 0, 0), fixed=True):
    """
    if you send in the "obID", it will delete the old instance before populating a new one
    orientation here is ignored because the hand orientation changes, so the object doesnt also have to rotate
    """
    if oID is not None:
        p.removeBody(oID)
    oID = p.loadURDF(object_path, oPos, globalScaling=object_scale, useFixedBase=fixed)
    return oID


"""#####################################################################################################################
                                            HAND ORIENTATION + LOCATION
#####################################################################################################################"""


def hand_dist(oID, rID, pos, oren):
    """
    actually does tthe movement to have hand touch object to judge distance

    returns position of the hand when it touches the object
    """
    # print("reset hand for non-fixed base")
    reset_hand(rID, rPos=pos, rOr=oren, fixed=False)

    relax(rID)  # want fingers splayed to get distance
    neg_pos = [-pos[0]*speed_find_distance, -pos[1]*speed_find_distance, -pos[2]*speed_find_distance]
    has_contact = 0
    while not has_contact:  # while still distance between hand/object
        p.applyExternalForce(rID, 1, neg_pos, pos, p.WORLD_FRAME)  # move hand toward object
        p.stepSimulation()
        contact_points = p.getContactPoints(rID, oID)  # get contact between cube and hand
        has_contact = len(contact_points)  # any contact stops the loop
    t_pos, t_oren = p.getBasePositionAndOrientation(rID)
    p.removeBody(rID)  # clean up
    return t_pos  # only need the position of the object


def adjust_point_dist(theta_rad, phi_rad, rID, oID, carts, quat):
    """
    move the hand w/fingers splayed until it touches the object
    should touch in center/palm - this should be the best for an initial grasp

    returns set of position coordinates representing how far from the object the hand should be (touching + a margin)
    """

    touching_pos = hand_dist(oID, rID, carts, quat)
    t_dist = sqrt(touching_pos[0] ** 2 + touching_pos[1] ** 2 + touching_pos[2] ** 2)
    # add a small margin to the contact point to allow for legal grasps
    m_dist = t_dist + grasp_distance_margin

    carts = astropy.coordinates.spherical_to_cartesian(m_dist, theta_rad, phi_rad)
    # associated coords have it facing away from the object - move to other side
    flip_carts = (-carts[0], -carts[1], -carts[2])

    return flip_carts


def get_given_point(dist, theta_rad, phi_rad, rID, oID):
    """
    get the coords and the quat for the hand based on distance from origin and angles

    returns a single (position, orientation) pair

    For the transform3d euler to quat: (seems like their z is our x, their y is our y, their x is our z)
    #Rotate about the current z-axis by ϕ. Then, rotate about the new y-axis by θ
    """
    carts = astropy.coordinates.spherical_to_cartesian(dist, theta_rad, phi_rad)
    # associated coords have it facing away from the object - move to other side
    neg_carts = (-carts[0], -carts[1], -carts[2])
    # the pi in the z brings it to face "up"
    quat = euler.euler2quat(phi_rad + pi, pi / 2 - theta_rad, pi, axes='sxyz')
    # move the hand w/fingers splayed until it touches the object, that is the dist to try for a grip
    close_carts = adjust_point_dist(theta_rad, phi_rad, rID, oID, neg_carts, quat)

    return (close_carts, quat)


def circle_set(rID, oID, n=20, theta=(pi/4), phi=pi):
    """
    move the hand around the object in a reasonable way
    returns an array of (position, orientation) pairs
    """
    increment = 2 * pi / n
    set = []
    for i in range(0, (n + 1)):
        # TODO: dist here needs to be programmatic
        set.append(get_given_point(dist=init_grasp_distance, theta_rad=-theta, phi_rad=(-phi) + increment * i, rID=rID, oID=oID))

    return set


def get_rand_point(dist):
    """
    get a random point dist away from the origin and facing the object

    returns a single (position, orientation) pair

    For the transform3d euler to quat: (seems like their z is our x, their y is our y, their x is our z)
    #Rotate about the current z-axis by ϕ. Then, rotate about the new y-axis by θ
    """
    theta_rad, phi_rad = rand_coord()
    carts = astropy.coordinates.spherical_to_cartesian(dist, theta_rad, phi_rad)
    neg_carts = (-carts[0], -carts[1], -carts[2])
    quat = euler.euler2quat(phi_rad + pi, pi / 2 - theta_rad, pi, axes='sxyz')  # the pi in the z brings it to face "up"

    return (neg_carts, quat)


def rand_set(dist=init_grasp_distance, n=10):
    """
    get n pairs for the hand randomly distributed dist away from the origin
    returns an array of (position, orientation) pairs
    """

    print("Getting set of random points")
    set = []
    for i in range(n):
        set.append(get_rand_point(dist))

    return set


def test_points(dist=init_grasp_distance):
    """
    some non-programmatic hand points ones that actually work
    returns an array of (position, orientation) pairs
    """
    test = [([dist, 0, 0], [0, -1, 0, 1]), ([-dist, 0, 0], [0, 1, 0, 1]), ([0, dist, 0], [1, 0, 0, 1]),
            ([0, -dist, 0], [-1, 0, 0, 1]), ([0, 0, -dist], [0, 0, 0, 1])]
    return test


"""#####################################################################################################################
                                            GRIPPER FUNCTIONS/MOVEMENT
#####################################################################################################################"""


def grasp(handId):
    """
    closes the gripper uniformly + attempts to find a grasp
    this is based on time + not contact points because contact points could just be a finger poking the object
    relies on grip_joints - specified by user/config file which joints should close
    TODO: make grasp mirror barret hand irl = that means make the base joint close before tip joint in hand
    """
    print("finding grasp")
    finish_time = time() + grasp_time_limit
    while time() < finish_time:
        p.stepSimulation()
        for joint in active_grasp_joints:
            p.setJointMotorControl2(bodyUniqueId=handId, jointIndex=joint, controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=max_grasp_force, force=target_grasp_velocity)


def relax(handID):
    """
    return all joints to neutral/furthest extended, based on urdf specification
    """
    print("relaxing hand")
    joint = 0
    num = p.getNumJoints(handID)
    while joint < num:
        p.resetJointState(bodyUniqueId=handID, jointIndex=joint, targetValue=0.0)
        joint = joint + 1


"""#####################################################################################################################
                                        POSITION/ORIENTATION DATA - GRAP MEMORY 
#####################################################################################################################"""


# TODO: edit this to more closely align with other standards

class Grasp:

    def __init__(self, position, orientation, joints):
        self.position = position
        self.orientation = orientation
        self.joints = joints

    def __repr__(self):
        return "Position: " + str(self.position) + " , Orientation: " + str(self.orientation) + " , Joints: " + str(
            self.joints) + " "

    def __str__(self):
        return "Position: " + str(self.position) + " , Orientation: " + str(self.orientation) + " , Joints: " + str(
            self.joints) + " "


def get_robot_config(handID):
    pos, oren = p.getBasePositionAndOrientation(handID)
    joints = {}
    num = p.getNumJoints(handID)
    for joint in range(0, num):
        joints[joint] = p.getJointState(handID, joint)
    return Grasp(pos, oren, joints)


"""#####################################################################################################################
                                            GRASP EVALUATION FUNCTIONS
#####################################################################################################################"""


def check_grip(cubeID, handID):
    # TODO: make direction of motion consistant (up and away from origin?)
    # TODO: modify to take in hand and cube position/orientation + load into environment w/gravity before shaking
    """
    check grip by adding in gravity
    """
    print("checking strength of current grip")
    mag = 1
    pos, oren = p.getBasePositionAndOrientation(handID)
    # pos, oren = p.getBasePositionAndOrientation(cubeID)
    time_limit = .5
    finish_time = time() + time_limit
    p.addUserDebugText("Grav Check!", [-.07, .07, .07], textColorRGB=[0, 0, 1], textSize=1)
    while time() < finish_time:
        p.stepSimulation()
        # add in "gravity"
        p.applyExternalForce(cubeID, linkIndex=-1, forceObj=[0, 0, -mag], posObj=pos, flags=p.WORLD_FRAME)
    contact = p.getContactPoints(cubeID, handID)  # see if hand is still holding obj after gravity is applied
    print("contacts", contact)
    if len(contact) > 0:
        p.removeAllUserDebugItems()
        p.addUserDebugText("Grav Check Passed!", [-.07, .07, .07], textColorRGB=[0, 1, 0], textSize=1)
        sleep(.3)
        print("Good Grip to Add")
        return get_robot_config(handID)

    else:
        p.removeAllUserDebugItems()
        p.addUserDebugText("Grav Check Failed!", [-.07, .07, .07], textColorRGB=[1, 0, 0], textSize=1)
        sleep(.3)
        return None


# TODO: other grasp measurement metrics - simulated annealing, etc


"""#####################################################################################################################
                                        MAIN MAIN MAIN MAIN MAIN MAIN
#####################################################################################################################"""
print("grasp!")
#handID = reset_hand()
cubeID = reset_ob()

#hand_set = circle_set(handID, cubeID)
hand_set = test_points()
print(hand_set)

handID = reset_hand()
cubeID = reset_ob(cubeID, [0, 0, 0])

good_grips = []

pos = 0
for each in hand_set:
    print("position #: ", pos)

    sleep(.2)
    relax(handID)
    p.multiplyTransforms([0,0,0], [0, .707, 0, -.707])
    p.resetBasePositionAndOrientation(handID, each[0], each[1])
    p.addUserDebugLine([0,0,0],[.2,0,0],[1,0,0],parentObjectUniqueId=handID, parentLinkIndex=-1, lineWidth = 500)
    p.addUserDebugLine([0,0,0],[0,.2,0],[0,1,0],parentObjectUniqueId=handID, parentLinkIndex=-1, lineWidth = 500)
    p.addUserDebugLine([0,0,0],[0,0,.2],[0,0,1],parentObjectUniqueId=handID, parentLinkIndex=-1, lineWidth = 500)
    cubeID = reset_ob(cubeID, [0, 0, 0], fixed=False)
    grasp(handID)
    sleep(.01)
    good_grips.append(check_grip(cubeID, handID))
    p.removeAllUserDebugItems()
    pos += 1

print("Num Good Grips: ", len(good_grips))
print("Grips:")
for grip in good_grips:
    print(grip)

"""#####################################################################################################################
                                       USEFUL INFORMATION, MAYBE 
########################################################################################################################


Num joints:  11
joint # 3
(3, b'bh_j32_joint', 0, 7, 6, 1, 100.0, 1.0, 0.0, 2.44, 30.0, 2.0, b'bh_finger_32_link', (0.0, 0.0, -1.0), (-0.0040000006556510925, 0.0, 0.033900000154972076), (-0.7071080610451548, 3.121199146877591e-17, -3.12121044560798e-17, 0.7071055013256238), 2)
joint # 6
(6, b'bh_j12_joint', 0, 10, 9, 1, 100.0, 1.0, 0.0, 2.44, 30.0, 2.0, b'bh_finger_12_link', (0.0, 0.0, -1.0), (-0.0040000006556510925, 0.0, 0.033900000154972076), (-0.7071080610451548, 0.0, 0.0, 0.7071055013256238), 5)
joint # 9
(9, b'bh_j22_joint', 0, 13, 12, 1, 100.0, 1.0, 0.0, 2.44, 30.0, 2.0, b'bh_finger_22_link', (0.0, 0.0, -1.0), (-0.0040000006556510925, 0.0, 0.033900000154972076), (-0.7071080610451548, 3.121199146877591e-17, -3.12121044560798e-17, 0.7071055013256238), 8)


joint lower limit: 0.0
joint upper limit: 2.44

(0, b'hand_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'hand_base_link', (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), -1)
(1, b'bh_base_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'bh_base_link', (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0)
(2, b'bh_j31_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'bh_finger_31_link', (0.0, 0.0, 0.0), (0.0, 0.0, 0.05040000006556511), (0.0, 0.0, 0.7071067966408574, 0.7071067657322373), 1)
(3, b'bh_j32_joint', 0, 7, 6, 1, 100.0, 1.0, 0.0, 2.44, 30.0, 2.0, b'bh_finger_32_link', (0.0, 0.0, -1.0), (-0.008000001311302185, 0.0, 0.06780000030994415), (-0.7071080610451548, 3.121199146877591e-17, -3.12121044560798e-17, 0.7071055013256238), 2)
(4, b'bh_j33_joint', 0, 8, 7, 1, 100.0, 1.0, 0.0, 0.84, 30.0, 2.0, b'bh_finger_33_link', (0.0, 0.0, -1.0), (-0.1398719996213913, 0.006000000052154064, 0.0), (0.0, 0.0, 0.0, 1.0), 3)
(5, b'bh_j11_joint', 0, 9, 8, 1, 100.0, 1.0, 0.0, 3.1416, 30.0, 2.0, b'bh_finger_11_link', (0.0, 0.0, -1.0), (-0.05000000074505806, 0.0, 0.05040000006556511), (0.0, 0.0, -0.7071080610451548, 0.7071055013256238), 1)
(6, b'bh_j12_joint', 0, 10, 9, 1, 100.0, 1.0, 0.0, 2.44, 30.0, 2.0, b'bh_finger_12_link', (0.0, 0.0, -1.0), (-0.008000001311302185, 0.0, 0.06780000030994415), (-0.7071080610451548, 0.0, 0.0, 0.7071055013256238), 5)
(7, b'bh_j13_joint', 0, 11, 10, 1, 100.0, 1.0, 0.0, 0.84, 30.0, 2.0, b'bh_finger_13_link', (0.0, 0.0, -1.0), (-0.1398719996213913, 0.006000000052154064, 0.0), (0.0, 0.0, 0.0, 1.0), 6)
(8, b'bh_j21_joint', 0, 12, 11, 1, 100.0, 1.0, 0.0, 3.1416, 30.0, 2.0, b'bh_finger_21_link', (0.0, 0.0, 1.0), (0.05000000074505806, 0.0, 0.05040000006556511), (0.0, 0.0, -0.7071080610451548, 0.7071055013256238), 1)
(9, b'bh_j22_joint', 0, 13, 12, 1, 100.0, 1.0, 0.0, 2.44, 30.0, 2.0, b'bh_finger_22_link', (0.0, 0.0, -1.0), (-0.008000001311302185, 0.0, 0.06780000030994415), (-0.7071080610451548, 3.121199146877591e-17, -3.12121044560798e-17, 0.7071055013256238), 8)
(10, b'bh_j23_joint', 0, 14, 13, 1, 100.0, 1.0, 0.0, 0.84, 30.0, 2.0, b'bh_finger_23_link', (0.0, 0.0, -1.0), (-0.1398719996213913, 0.006000000052154064, 0.0), (0.0, 0.0, 0.0, 1.0), 9)


points = points_on_circumference(center=(0, 0), r=.15, n=5)


for each in points:
    p.resetBasePositionAndOrientation(handID, [each[0], each[1], 0], [0, 0, 0, 1])

planeId = p.loadURDF("plane.urdf", [0, 0, -0.5]) #the ground!


"""
