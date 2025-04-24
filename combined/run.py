import pybullet as p
import numpy as np
from camera import Camera
import cv2
import time

IMG_SIDE = 300
IMG_HALF = IMG_SIDE / 2
camera = Camera(imgSize=[IMG_SIDE, IMG_SIDE])

Z0 = 0.3  # camera height
dt = 1 / 240
coef = 0.5
maxTime = 10
logTime = np.arange(0.0, maxTime, dt)

physicsClient = p.connect(p.GUI, options="--background_color_red=1 --background_color_blue=1 --background_color_green=1")
p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=-90, cameraPitch=-89.999, cameraTargetPosition=[0.5, 0.5, 0.6])
p.setGravity(0, 0, -10)

boxId = p.loadURDF("combined/robot.urdf.xml", useFixedBase=True)

jointIndices = []
for i in range(p.getNumJoints(boxId)):
    jointInfo = p.getJointInfo(boxId, i)
    if jointInfo[2] == p.JOINT_REVOLUTE:
        jointIndices.append(i)

eefLinkIdx = p.getNumJoints(boxId) - 1

c = p.loadURDF("combined/aruco.urdf", (0.5, 0.5, 0.0), useFixedBase=True)
x = p.loadTexture("combined/aruco_cube.png")
p.changeVisualShape(c, -1, textureUniqueId=x)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

def computeInterMatrix(Z, sd0):
    L = np.zeros((8, 3))
    for idx in range(4):
        x = sd0[2 * idx, 0]
        y = sd0[2 * idx + 1, 0]
        L[2 * idx] = np.array([-1 / Z, 0, y])
        L[2 * idx + 1] = np.array([0, -1 / Z, -x])
    return L

def updateCamPos(cam):
    linkState = p.getLinkState(boxId, linkIndex=eefLinkIdx)
    xyz = linkState[0]
    quat = linkState[1]
    rotMat = p.getMatrixFromQuaternion(quat)
    rotMat = np.reshape(np.array(rotMat), (3, 3))
    camera.set_new_position(xyz, rotMat)

def move_to_cartesian(xyz, rpy):
    quat = p.getQuaternionFromEuler(rpy)
    ik_solution = p.calculateInverseKinematics(boxId, eefLinkIdx, xyz, quat)
    joint_angles = [ik_solution[i] for i in range(3)]
    p.setJointMotorControlArray(bodyIndex=boxId, jointIndices=jointIndices, targetPositions=joint_angles, controlMode=p.POSITION_CONTROL)
    for _ in range(100):
        p.stepSimulation()

# Начальное положение над маркером
move_to_cartesian([0.47, 0.47, 0.3], [0.0, 0.0, 1.66])

updateCamPos(camera)
img = camera.get_frame()
corners, markerIds, rejectedCandidates = detector.detectMarkers(img)

if len(corners) == 0:
    print("No markers detected! Exiting.")
    p.disconnect()
    exit()

sd0 = np.reshape(np.array(corners[0][0]), (8, 1))
sd0 = np.array([(s - IMG_HALF) / IMG_HALF for s in sd0])

# Смещенное начальное положение
move_to_cartesian([0.45, 0.45, 0.3], [0.0, 0.0, 0.05])

camCount = 0
w = np.zeros((3, 1))

for t in logTime[1:]:
    p.stepSimulation()
    camCount += 1
    if camCount == 5:
        camCount = 0
        updateCamPos(camera)
        img = camera.get_frame()
        corners, markerIds, rejectedCandidates = detector.detectMarkers(img)

        if len(corners) == 0:
            continue

        s0 = np.reshape(np.array(corners[0][0]), (8, 1))
        s0 = np.array([(ss - IMG_HALF) / IMG_HALF for ss in s0])
        L0 = computeInterMatrix(Z0, s0)
        L0T = np.linalg.pinv(L0)
        e = s0 - sd0
        w = -coef * L0T @ e

    jStates = p.getJointStates(boxId, jointIndices=jointIndices)
    jPos = [state[0] for state in jStates]
    (linJac, angJac) = p.calculateJacobian(boxId, eefLinkIdx, [0, 0, 0], jPos, [0] * len(jointIndices), [0] * len(jointIndices))

    J = np.block([
        [np.array(linJac)[:2, :2], np.zeros((2, 1))],
        [np.array(angJac)[2, :]]
    ])
    dq = (np.linalg.pinv(J) @ w).flatten()
    dq = dq[[1, 0, 2]]
    dq[2] = -dq[2]

    p.setJointMotorControlArray(boxId, jointIndices, targetVelocities=dq, controlMode=p.VELOCITY_CONTROL)

p.disconnect()