import pybullet as p
import pybullet_data
from time import sleep, time
from mathutils import Vector #https://github.com/majimboo/py-mathutils

#physicsClient = p.connect(p.DIRECT)  #p.DIRECT for non-graphical version
physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally

# This is to change the visualizer window settings
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, enable=0)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, enable=0)
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, enable=0)
# change init camera distance/location to view scene
p.resetDebugVisualizerCamera(cameraDistance=.75, cameraYaw=45, cameraPitch=-20, cameraTargetPosition=[0.0, 0.0, 0.0])


#rID = p.loadURDF("/Users/carlyndougherty/PycharmProjects/cr_grasper/RobotURDFs/gripper_description/pr2_gripper.urdf", useFixedBase=True)
rID = p.loadURDF("/Users/carlyndougherty/PycharmProjects/cr_grasper/RobotURDFs/finger_description/urdf/RH8D.urdf", globalScaling=1, useFixedBase=True)
p.addUserDebugLine([0, 0, 0], [.2, 0, 0], [1, 0, 0], parentObjectUniqueId=rID, parentLinkIndex=-1, lineWidth=500)
p.addUserDebugLine([0, 0, 0], [0, .2, 0], [0, 1, 0], parentObjectUniqueId=rID, parentLinkIndex=-1, lineWidth=500)
p.addUserDebugLine([0, 0, 0], [0, 0, .2], [0, 0, 1], parentObjectUniqueId=rID, parentLinkIndex=-1, lineWidth=500)
#rID = p.loadSDF("/Users/carlyndougherty/PycharmProjects/cr_grasper/pybullet_examples/bullet3/data/gripper/wsg50_one_motor_gripper.sdf")[0]


jointNum = p.getNumJoints(rID)

for joint in range(0,jointNum):
    print(p.getJointInfo(rID, joint))
init = Vector((0,.04,-.21))
active_grasp_joints = [4,5,7,8,10,11,13,18]
#active_grasp_joints = [4,6,9,12,16,19]
#active_grasp_joints = [4,6,9,12,16,19,5,8,11,14,18]
def grasp(handId):
    """
    closes the gripper uniformly + attempts to find a grasp
    this is based on time + not contact points because contact points could just be a finger poking the object
    relies on grip_joints - specified by user/config file which joints should close
    TODO: make grasp mirror barret hand irl = that means make the base joint close before tip joint in hand
    """
    print("finding grasp")
    grasp_time_limit = 1
    finish_time = time() + grasp_time_limit
    while time() < finish_time:
        p.stepSimulation()
        for joint in active_grasp_joints:
            p.setJointMotorControl2(bodyUniqueId=handId, jointIndex=joint, controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=.4, force=5)
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



while True:
    #p.resetBasePositionAndOrientation(rID, [-.4,0,0], [0, .707, 0, -.707])
    p.resetBasePositionAndOrientation(rID, init, [0, 0, 0, 1])
    relax(rID)
    cID = p.loadURDF("/Users/carlyndougherty/PycharmProjects/cr_grasper/pybullet_examples/bullet3/data/cube_small.urdf",
                     globalScaling=.45)
    #p.resetBasePositionAndOrientation(rID, [init[0], init[1], init[2]], [0, .707, 0, -.707])
    #([-dist, 0, 0], [0, 1, 0, 1])

    grasp(rID)
    while True:
        p.stepSimulation()


"""

gripper:
(0, b'left_gripper_joint', 0, 7, 6, 1, 0.0, 0.0, -2.0, 2.548, 1000.0, 0.5, b'left_gripper', (0.0, 0.0, 1.0), (0.20000000298023224, 0.009999999776482582, 0.0), (0.0, 0.0, 0.0, 1.0), -1)
(1, b'left_tip_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'left_tip', (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0)
(2, b'right_gripper_joint', 0, 8, 7, 1, 0.0, 0.0, -2.0, 2.0, 1000.0, 0.5, b'right_gripper', (0.0, 0.0, -1.0), (0.20000000298023224, -0.009999999776482582, 0.0), (0.0, 0.0, 0.0, 1.0), -1)
(3, b'right_tip_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'right_tip', (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 2)



rh8d:
base_link(0, b'base_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'base', (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), -1)
(1, b'forearm__base', 0, 7, 6, 1, 0.1, 0.0, -1.57079632679, 1.57079632679, 0.5, 1.0, b'forearm', (0.0, -1.0, 2.220446049250313e-16), (2.3248499928740785e-05, -5.52515011804644e-05, -0.0018176099401898682), (0.7071067966408574, 0.0, 0.0, 0.7071067657322373), 0)
(2, b'palm axis__forearm', 0, 8, 7, 1, 0.1, 0.0, -0.785398163397, 0.785398163397, 0.5, 1.0, b'palm axis', (-2.220446049250313e-16, 0.0, -1.0), (-2.3043300245717546e-07, -0.0994626060128212, -0.01242292020469904), (4.371139000186237e-08, 0.9999999999999981, 4.371139000186237e-08, -1.9106856158949176e-15), 1)
(3, b'palm__palm axis', 0, 9, 8, 1, 0.1, 0.0, -0.785398163397, 0.785398163397, 0.5, 1.0, b'palm', (-1.0, 2.220446049250313e-16, 0.0), (-0.026499999687075615, -0.009025000035762787, 0.0), (0.0, 0.0, 0.0, 1.0), 2)
(4, b'Rproximal__palm', 0, 10, 9, 1, 0.1, 0.0, 0.0, 1.4835298642, 0.5, 1.0, b'Rproximal', (-1.0000001192092896, -8.462200185022084e-08, 1.4422505500988336e-07), (0.022196771577000618, -0.038704995065927505, 0.0025688399327918887), (0.7930019658754083, 0.06156963395891232, -0.002688090651835728, 0.6060939171948113), 3)
(5, b'Rmiddle__Rproximal', 0, 11, 10, 1, 0.1, 0.0, 0.0, 1.4835298642, 0.5, 1.0, b'Rmiddle', (-1.0000001192092896, 1.7276738617511e-07, 2.3154271389103087e-07), (0.005062655080109835, -0.00017841116641648114, -0.014479481615126133), (0.04361968755853659, 6.0394149958666276e-09, -1.383243913373079e-07, 0.9990482084750848), 4)
(6, b'Rtip__Rmiddle', 0, 12, 11, 1, 0.1, 0.0, 0.0, 1.4835298642, 0.5, 1.0, b'Rtip', (-1.0000001192092896, -1.9437865717009117e-07, 1.0272773920405598e-07), (0.005062737502157688, -0.00017847789384006774, -0.014479444362223148), (0.04361929677534002, -6.387485024985695e-08, 1.7610796237058589e-07, 0.9990482255370807), 5)
(7, b'Mproximal__palm', 0, 13, 12, 1, 0.1, 0.0, 0.0, 1.4835298642, 0.5, 1.0, b'Mproximal', (-1.0000003576278687, -5.712353754461219e-07, 2.2211921191228612e-07), (-0.002552090212702751, -0.0403130017220974, 0.0018457099795341492), (0.7925982181247307, -0.03215928432673143, -0.02946876489973575, 0.6081820754941554), 3)
(8, b'Mmiddle__Mproximal', 0, 14, 13, 1, 0.1, 0.0, 0.0, 1.4835298642, 0.5, 1.0, b'Mmiddle', (-1.0000003576278687, -2.492787132268859e-07, -7.013297675939612e-08), (0.005062680225819349, -0.000178611403555351, -0.014479018747806549), (0.04361924467094592, -1.2809186962501998e-07, -1.64139826818337e-07, 0.9990482278119975), 7)
(9, b'Mtip__Mmiddle', 0, 15, 14, 1, 0.1, 0.0, 0.0, 1.4835298642, 0.5, 1.0, b'Mtip', (-1.0000003576278687, -8.762479097867981e-08, 1.532650912849931e-07), (0.005062691867351532, -0.00017828853313517357, -0.014479816891252995), (0.043619486584317246, 1.1894046008216775e-07, -8.256309925728865e-08, 0.999048217249848), 8)
(10, b'Pproximal__palm', 0, 16, 15, 1, 0.1, 0.0, 0.0, 1.4835298642, 0.5, 1.0, b'Pproximal', (-0.9999997019767761, -2.6381343332104734e-07, -1.779326339601539e-07), (0.046860167756676674, -0.029467996209859848, 0.00494262995198369), (0.791951313945375, 0.12267105323858558, -0.005355990858946725, 0.5981105603477078), 3)
(11, b'Pmiddle__Pproximal', 0, 17, 16, 1, 0.1, 0.0, 0.0, 1.4835298642, 0.5, 1.0, b'Pmiddle', (-0.9999997019767761, -2.6981936684933316e-07, -3.089447488946462e-07), (0.0050628213876109385, -0.00017847168783191592, -0.014478928409516811), (0.043619456810370674, -5.310948932704238e-08, -2.3188140801496735e-09, 0.9990482185498182), 10)
(12, b'Ptip__Pmiddle', 0, 18, 17, 1, 0.1, 0.0, 0.0, 1.4835298642, 0.5, 1.0, b'Ptip', (-0.9999997019767761, -1.1446730496800228e-07, -5.484685061674099e-07), (0.005062717944383621, -0.00017845426918938756, -0.01447964645922184), (0.04361910696637803, -1.1126679310952315e-07, -5.8902314869320834e-08, 0.9990482338242933), 11)
(13, b'palm__thumb base', 0, 19, 18, 1, 0.1, 0.0, 0.0, 1.57079632679, 0.5, 1.0, b'thumb base', (7.477392927057736e-08, -1.0000003576278687, -1.6957208970325155e-07), (-0.035564529709517956, 0.02871800120919943, 0.010119629558175802), (-0.6294324723053143, -0.35806419278806756, 0.6673237940537868, 0.17402227024236983), 3)
(14, b'Tproximal__thumb base', 0, 20, 19, 1, 0.1, 0.0, 0.0, 1.57079632679, 0.5, 1.0, b'Tproximal', (-1.0000001192092896, 4.0430174408356834e-07, -1.0396789207334223e-07), (0.005062170183578019, -0.008211906999349594, 0.010910095617873594), (0.9752529984320588, -1.577589615193237e-07, 3.576427308219542e-08, 0.2210918113573012), 13)
(15, b'Tmiddle__Tproximal', 0, 21, 20, 1, 0.1, 0.0, 0.0, 1.57079632679, 0.5, 1.0, b'Tmiddle', (-1.0000001192092896, -1.255591683957391e-07, 1.1845047964698097e-07), (0.005091341886739542, 0.00010905539733130354, -0.012410018593072891), (0.0047676489745258075, 1.1054343319804662e-07, 2.648928962599829e-07, 0.9999886346970016), 14)
(16, b'Ttip__Tmiddle', 0, 22, 21, 1, 0.1, 0.0, 0.0, 1.57079632679, 0.5, 1.0, b'Ttip', (-1.0000001192092896, -1.255591683957391e-07, 1.1845047964698097e-07), (0.005091457162052393, 0.00010963625146587219, -0.012409763410687447), (0.0, 0.0, 0.0, 1.0), 15)
(17, b'Iproximal__palm', 0, 23, 22, 1, 0.1, 0.0, 0.0, 1.4835298642, 0.5, 1.0, b'Iproximal', (-0.9999998211860657, -3.110646389359317e-07, -1.9257539918271505e-07), (-0.02760872943326831, -0.039222996681928635, 0.0045974100939929485), (0.7918965584395677, -0.09355363545821303, -0.026788086314377756, 0.6028515210694367), 3)
(18, b'Imiddle__Iproximal', 0, 24, 23, 1, 0.1, 0.0, 0.0, 1.4835298642, 0.5, 1.0, b'Imiddle', (-0.9999998211860657, 9.534426226309733e-08, 4.368456245629204e-07), (0.005062692798674107, -0.00017866119742393494, -0.014479033648967743), (0.043619267001321606, 3.191158268278808e-07, -2.0833853152961528e-07, 0.9990482268369842), 17)
(19, b'Itip__Imiddle', 0, 25, 24, 1, 0.1, 0.0, 0.0, 1.4835298642, 0.5, 1.0, b'Itip', (-0.9999998211860657, -4.999270686312229e-07, 2.564417513895023e-07), (0.005062660668045282, -0.00017852871678769588, -0.0144792590290308), (0.04361942331464743, -8.129227160317894e-08, 2.822320449717363e-07, 0.9990482200122331), 18)
"""

