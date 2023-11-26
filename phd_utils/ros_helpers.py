import numpy as np
from geometry_msgs.msg import Quaternion, Transform, Vector3, Pose
from tf.transformations import *
from phd_utils.nuitrack import joints_idx

kinect_joints = ["K4ABT_JOINT_PELVIS", "K4ABT_JOINT_SPINE_NAVEL", "K4ABT_JOINT_SPINE_CHEST", "K4ABT_JOINT_NECK", "K4ABT_JOINT_CLAVICLE_LEFT", "K4ABT_JOINT_SHOULDER_LEFT", "K4ABT_JOINT_ELBOW_LEFT", "K4ABT_JOINT_WRIST_LEFT", "K4ABT_JOINT_HAND_LEFT", "K4ABT_JOINT_HANDTIP_LEFT", "K4ABT_JOINT_THUMB_LEFT", "K4ABT_JOINT_CLAVICLE_RIGHT", "K4ABT_JOINT_SHOULDER_RIGHT", "K4ABT_JOINT_ELBOW_RIGHT", "K4ABT_JOINT_WRIST_RIGHT", "K4ABT_JOINT_HAND_RIGHT", "K4ABT_JOINT_HANDTIP_RIGHT", "K4ABT_JOINT_THUMB_RIGHT", "K4ABT_JOINT_HIP_LEFT", "K4ABT_JOINT_KNEE_LEFT", "K4ABT_JOINT_ANKLE_LEFT", "K4ABT_JOINT_FOOT_LEFT", "K4ABT_JOINT_HIP_RIGHT", "K4ABT_JOINT_KNEE_RIGHT", "K4ABT_JOINT_ANKLE_RIGHT", "K4ABT_JOINT_FOOT_RIGHT", "K4ABT_JOINT_HEAD", "K4ABT_JOINT_NOSE", "K4ABT_JOINT_EYE_LEFT", "K4ABT_JOINT_EAR_LEFT", "K4ABT_JOINT_EYE_RIGHT", "K4ABT_JOINT_EAR_RIGHT", "K4ABT_JOINT_COUNT"]
kinect_joints_idx = {kinect_joints[i]:i for i in range(len(kinect_joints))}

def mat2ROS(T:np.ndarray)->tuple:
	if T.shape == (4,4):
		return Vector3(*T[:3,3].tolist()), Quaternion(*quaternion_from_matrix(T))
	elif T.shape == (3,):
		return Vector3(*T.tolist()), Quaternion(0,0,0,1)

def ROS2mat(msg :(Transform or Pose))->np.ndarray:
	if isinstance(msg, Transform):
		return TF2mat(msg)
	elif isinstance(msg, Pose):
		return Pose2mat(msg)

def mat2TF(T:np.ndarray)->Transform:
	assert(T.shape == (4,4) or T.shape == (3,))
	return Transform(*mat2ROS(T))

def mat2Pose(T:np.ndarray)->Transform:
	assert(T.shape == (4,4) or T.shape == (3,))
	return Pose(*mat2ROS(T))

def TF2vec(transform:Transform)->np.ndarray:
	return np.array([transform.translation.x, transform.translation.y, transform.translation.z])

def TF2mat(transform:Transform)->np.ndarray:
	T = quaternion_matrix([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w])
	T[:3,3] = TF2vec(transform)
	return T

def Pose2vec(pose:Pose)->np.ndarray:
	return np.array([pose.position.x, pose.position.y, pose.position.z])

def Pose2mat(pose:Pose)->np.ndarray:
	T = quaternion_matrix([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
	T[:3,3] = Pose2vec(pose)
	return T