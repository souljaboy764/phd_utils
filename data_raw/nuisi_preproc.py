import numpy as np
import os
import argparse

from nuisi_dataset.read_nuisi import *

parser = argparse.ArgumentParser(description='Data preprocessing for trajectories of the NuiSI dataset')
parser.add_argument('--src', type=str, default='./nuisi_dataset', metavar='SRC',
					help='Path where https://github.com/souljaboy764/nuisi_dataset is extracted to read csv files (default: ./nuisi_dataset).')
parser.add_argument('--dst', type=str, default='./data/', metavar='DST',
					help='Path to save the processed trajectories to (default: ./data).')
args = parser.parse_args()

if not os.path.exists(args.dst):
	os.mkdir(args.dst)

train_data = []
test_data = []
train_labels = []
test_labels =[]
for a in range(len(actions)):
	index_p1, times_p1, num_joints_p1, data_p1 = readfile(os.path.join(args.src, 'data', f'{actions[a]}_p1.txt'))
	index_p2, times_p2, num_joints_p2, data_p2 = readfile(os.path.join(args.src, 'data', f'{actions[a]}_p2.txt'))

	data_p1[:,:,[-3,-2,-1]] = data_p1[:,:,[-1,-3,-2]]*0.001
	data_p1[:,:,-2] *= -1

	data_p2[:,:,[-3,-2,-1]] = data_p2[:,:,[-1,-3,-2]]*0.001
	data_p2[:,:,-2] *= -1

	trajs_a = []
	for s in trajectory_idx[actions[a]]:
		T1 = rotation_normalization(data_p1[s[0], :, -3:])
		T2 = rotation_normalization(data_p2[s[0], :, -3:])
		p1 = []
		p2 = []
		for i in range(s[0], s[1]):
			p1.append(T1[:3,:3].dot(data_p1[i, rarm_idx, -3:].T).T + T1[:3,3])
			p2.append(T2[:3,:3].dot(data_p2[i, rarm_idx, -3:].T).T + T2[:3,3])

		p1 = np.array(p1)
		p2 = np.array(p2)

		seq_len = s[1] - s[0]
		trajs_a.append(np.concatenate([p1.reshape((seq_len, -1)), p2.reshape((seq_len, -1))], axis=-1))


	train_split = int(0.8*len(trajs_a))
	train_data += trajs_a[:train_split]
	test_data += trajs_a[train_split:]
	train_labels += (np.ones(train_split)*a).tolist()
	test_labels += (np.ones(len(trajs_a)-train_split)*a).tolist()
	print(len(train_data), len(test_data))

train_data = np.array(train_data, dtype=object)
test_data = np.array(test_data, dtype=object)

np.savez_compressed(os.path.join(args.dst,'traj_data.npz'), train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels)
