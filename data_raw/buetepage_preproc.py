import numpy as np
import os
import argparse

from human_robot_interaction_data.read_hh_hr_data import *

parser = argparse.ArgumentParser(description='Data preprocessing for trajectories of Buetepage et al. (2020).')
parser.add_argument('--src', type=str, default='./human_robot_interaction_data', metavar='SRC',
					help='Path where https://github.com/souljaboy764/human_robot_interaction_data is extracted to read csv files (default: ./human_robot_interaction_data).')
parser.add_argument('--dst', type=str, default='./data/', metavar='DST',
					help='Path to save the processed trajectories to (default: ./data).')
parser.add_argument('--robot', action='store_true',
					help='Whether to use the Yumi robot data or not. If set, preproccessing of HRI data is done else of HHI (default: False).')
args = parser.parse_args()

if not os.path.exists(args.dst):
	os.mkdir(args.dst)

train_data, train_labels, test_data, test_labels = [], [], [] ,[]
action_onehot = np.eye(5)
actions = ['hand_wave', 'hand_shake', 'rocket', 'parachute']

for a in range(len(actions)):
	action = actions[a]
	trajectories, traj_labels = [], []
	idx_list = np.array([joints_dic[joint] for joint in ['RightArm', 'RightForeArm', 'RightHand']])
	trials = ['1'] if args.robot else ['1', '2']
	for trial in trials:
		if args.robot:
			data_p1, data_r2 = read_hri_data(action, os.path.join(args.src, 'hr'))
			segments_file_h = os.path.join(args.src, 'hr', 'segmentation', action+'_p1.npy')
			segments_file_r = os.path.join(args.src, 'hr', 'segmentation', action+'_r2.npy')
			segments = np.load(segments_file_h)
			segments_r = np.load(segments_file_r)

		else:
			data_file_p1 = os.path.join(args.src, 'hh','p1',action+'_s1_'+trial+'.csv')
			data_file_p2 = os.path.join(args.src, 'hh','p2',action+'_s2_'+trial+'.csv')
			data_p2, _, _, _ = read_data(data_file_p2)
			data_p2[..., [0,1,2]]  = data_p2[..., [2,0,1]]
			data_p2[..., 1] *= -1
			segment_file = os.path.join(args.src, 'hh', 'segmentation', action+'_'+trial+'.npy')
			segments = np.load(segment_file)
		
			data_p1, _, _, _ = read_data(data_file_p1)
		data_p1[..., [0,1,2]]  = data_p1[..., [2,0,1]]
		data_p1[..., 1] *= -1

		for i in range(len(segments)):
			s = segments[i]
			traj1 = data_p1[s[0]:s[1], idx_list] # seq_len, N, 3
			traj1 = traj1 - traj1[0,0]
			# the indices where no movement occurs at the end are annotated as "not active". (Sec. 4.3.1 of the paper)
			notactive_idx = np.where(np.sqrt(np.power(np.diff(traj1, axis=0),2).sum((2))).mean(1) > 1e-3)[0]
			labels = np.zeros((traj1.shape[0],5))
			labels[:] = action_onehot[a]
			if len(notactive_idx) > 0:
				labels[notactive_idx[-1]:] = action_onehot[-1]
			if args.robot:
				s_r = segments_r[i]
				traj2 = data_r2[s_r[0]:s_r[1]] # seq_len, 7
				traj = np.concatenate([traj1.reshape(-1, len(idx_list)*3), traj2], axis=-1) # seq_len, 3*N + 7
				
			else:
				traj2 = data_p2[s[0]:s[1], idx_list] # seq_len, N, 3
				traj2 = traj2 - traj2[0,0]
				traj = np.concatenate([traj1.reshape(-1, len(idx_list)*3), traj2.reshape(-1, len(idx_list)*3)], axis=-1) # seq_len, 6*N + 6*N
		
			trajectories.append(traj)
			traj_labels.append(labels)
			
	# the first 80% are for training and the last 20% are for testing (Sec. 4.3.2)
	split_idx = int(0.8*len(trajectories))
	train_data += trajectories[:split_idx]
	test_data += trajectories[split_idx:]
	train_labels += traj_labels[:split_idx]
	test_labels += traj_labels[split_idx:]
	print(len(train_data), len(test_data))

train_data = np.array(train_data, dtype=object)
test_data = np.array(test_data, dtype=object)
train_labels = np.array(train_labels, dtype=object)
test_labels = np.array(test_labels, dtype=object)

np.savez_compressed(os.path.join(args.dst, 'traj_data.npz'), train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels)