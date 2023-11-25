import torch
import torch.nn.functional as F
from torch.nn.functional import grid_sample, affine_grid

import numpy as np

from phd_utils.transformations import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def downsample_trajs(train_data, downsample_len):
	theta = torch.Tensor(np.array([[[1,0,0.], [0,1,0]]])).to(device).repeat(train_data[0].shape[1],1,1)
	num_trajs = len(train_data)
	for i in range(num_trajs):
		old_shape=train_data[i].shape
		train_data[i] = train_data[i].transpose(1,2,0) # 4, 3, seq_len
		train_data[i] = torch.Tensor(train_data[i]).to(device).unsqueeze(2) # 4, 3, 1 seq_len
		train_data[i] = torch.concat([train_data[i], torch.zeros_like(train_data[i])], dim=2) # 4, 3, 2 seq_len
		
		grid = affine_grid(theta, torch.Size([old_shape[1], old_shape[2], 2, int(downsample_len*old_shape[0])]), align_corners=True)
		train_data[i] = grid_sample(train_data[i].type(torch.float32), grid, align_corners=True) # 4, 3, 2 downsample_len
		train_data[i] = train_data[i][:, :, 0].cpu().detach().numpy() # 4, 3, downsample_len
		train_data[i] = train_data[i].transpose(2,0,1) # downsample_len, 4, 3
	return train_data

joints = ["waist", "neck", "head", "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist"]
joints_dic = {joints[i]:i for i in range(len(joints))}

def angle(a,b):
	dot = np.dot(a,b)
	cos = dot/(np.linalg.norm(a)*np.linalg.norm(b))
	if np.allclose(cos, 1):
		cos = 1
	elif np.allclose(cos, -1):
		cos = -1
	return np.arccos(cos)

def projectToPlane(plane, vec):
	return (vec - plane)*np.dot(plane,vec)

def rotation_normalization(skeleton):
	leftShoulder = skeleton[joints_dic["left_shoulder"]]
	rightShoulder = skeleton[joints_dic["right_shoulder"]]
	waist = skeleton[joints_dic["waist"]]
	
	xAxisHelper = waist - rightShoulder
	yAxis = leftShoulder - rightShoulder # right to left
	xAxis = np.cross(xAxisHelper, yAxis) # out of the human(like an arrow in the back)
	zAxis = np.cross(xAxis, yAxis) # like spine, but straight
	
	xAxis /= np.linalg.norm(xAxis)
	yAxis /= np.linalg.norm(yAxis)
	zAxis /= np.linalg.norm(zAxis)

	return np.array([[xAxis[0], xAxis[1], xAxis[2]],
					 [yAxis[0], yAxis[1], yAxis[2]],
					 [zAxis[0], zAxis[1], zAxis[2]]])

def joint_angle_extraction(skeleton): # Based on the Pepper Robot URDF, with the limits
	# Recreating arm with upper and under arm
	rightUpperArm = skeleton[1] - skeleton[0]
	rightUnderArm = skeleton[2] - skeleton[1]


	rightElbowAngle = np.clip(angle(rightUpperArm, rightUnderArm), 0.0087, 1.562)
	
	rightYaw = np.clip(np.arcsin(min(rightUpperArm[1],-0.0087)/np.linalg.norm(rightUpperArm)), -1.562, -0.0087)
	
	rightPitch = np.arctan2(max(rightUpperArm[0],0), rightUpperArm[2])
	rightPitch -= np.pi/2 # Needed for pepper frame
	
	# Recreating under Arm Position with known Angles(without roll)
	rightRotationAroundY = euler_matrix(0, rightPitch, 0,)[:3,:3]
	rightRotationAroundX = euler_matrix(0, 0, rightYaw)[:3,:3]
	rightElbowRotation = euler_matrix(0, 0, rightElbowAngle)[:3,:3]

	rightUnderArmInZeroPos = np.array([np.linalg.norm(rightUnderArm), 0, 0.])
	rightUnderArmWithoutRoll = np.dot(rightRotationAroundY,np.dot(rightRotationAroundX,np.dot(rightElbowRotation,rightUnderArmInZeroPos)))

	# Calculating the angle betwenn actual under arm position and the one calculated without roll
	rightRoll = angle(rightUnderArmWithoutRoll, rightUnderArm)

	return np.array([rightPitch, rightYaw, rightRoll, rightElbowAngle]).astype(np.float32)


def window_concat(traj_data, window_length, robot=None, input_dim=None):
	window_trajs = []
	for i in range(len(traj_data)):
		trajs_concat = []
		traj_shape = traj_data[i].shape
		dim = traj_shape[-1]
		if robot is None and input_dim is None:
			input_dim = dim//2
			# input_dim = int(2*dim/3)
		elif input_dim is None:
			if robot=='pepper':
				input_dim = dim-4
			elif robot=='yumi':
				input_dim = dim-7
		idx = np.array([np.arange(i,i+window_length) for i in range(traj_shape[0] + 1 - 2*window_length)])
		trajs_concat.append(traj_data[i][:,:input_dim][idx].reshape((traj_shape[0] + 1 - 2*window_length, window_length*input_dim)))
		idx = np.array([np.arange(i,i+window_length) for i in range(window_length, traj_shape[0] + 1 - window_length)])
		trajs_concat.append(traj_data[i][:,input_dim:][idx].reshape((traj_shape[0] + 1 - 2*window_length, window_length*(dim-input_dim))))
		# elif robot=='pepper':
		# 	idx = np.array([np.arange(i,i+window_length) for i in range(traj_shape[0] + 1 - 2*window_length)])
		# 	trajs_concat.append(traj_data[i][:,:dim-4][idx].reshape((traj_shape[0] + 1 - 2*window_length, window_length*(dim-4))))
		# 	idx = np.array([np.arange(i,i+window_length) for i in range(window_length, traj_shape[0] + 1 - window_length)])
		# 	trajs_concat.append(traj_data[i][:,-4:][idx].reshape((traj_shape[0] + 1 - 2*window_length, window_length*4)))
		# elif robot=='yumi':
		# 	idx = np.array([np.arange(i,i+window_length) for i in range(traj_shape[0] + 1 - 2*window_length)])
		# 	trajs_concat.append(traj_data[i][:,:dim-7][idx].reshape((traj_shape[0] + 1 - 2*window_length, window_length*(dim-7))))
		# 	idx = np.array([np.arange(i,i+window_length) for i in range(window_length, traj_shape[0] + 1 - window_length)])
		# 	trajs_concat.append(traj_data[i][:,-7:][idx].reshape((traj_shape[0] + 1 - 2*window_length, window_length*7)))

		trajs_concat = np.concatenate(trajs_concat,axis=-1)
		window_trajs.append(trajs_concat)
	return window_trajs
