import numpy as np
import os
import argparse
import glob
import csv

parser = argparse.ArgumentParser(description='Data preprocessing for trajectories of Buetepage et al. (2020).')
parser.add_argument('--src', type=str, default='.', metavar='SRC',
					help='Path where https://zenodo.org/records/7767535#.ZB2-43bMLIU is extracted and unzipped (parent folder of "Bimanual Handovers Dataset") to read csv files (default: .).')
parser.add_argument('--dst', type=str, default='./data/', metavar='DST',
					help='Path to save the processed trajectories to (default: ./data).')
parser.add_argument('--robot', action='store_true',
					help='Whether to preprocess the data for the Kobo robot or not. (default: False).')
args = parser.parse_args()

if not os.path.exists(args.dst):
	os.mkdir(args.dst)

joints = ['Hip', 'Ab', 'Chest', 'Neck', 'Head', 'LShoulder', 'LUArm', 'LFArm', 'LHand', 'RShoulder', 'RUArm', 'RFArm', 'RHand']
joints_dic = {joints[i]:i for i in range(len(joints))}
object_key = 'Rigid Body:object:Position'
label_idx = {'reach':0, 'transfer':1, 'retreat':2}

def preproc_files_list(files):
	p1_trajs = []
	p2_trajs = []
	object_trajs = []
	labels = []
	for f_csv in files:
		p1_name = os.path.basename(f_csv).split('_')[0]
		p2_name = os.path.basename(f_csv).split('_')[1]
		p1_body_keys = [f'Bone:{p1_name}:{joint}:Position' for joint in joints]
		p2_body_keys = [f'Bone:{p2_name}:{joint}:Position' for joint in joints]
		p1_traj = []
		p2_traj = []
		object_traj = []
		label = []
		reader = csv.DictReader(open(f_csv))
		try:
			for row in reader:
				p1_traj.append([])
				p2_traj.append([])
				for k in p1_body_keys:
					p1_traj[-1].append([-float(row[k+':X']), float(row[k+':Z']), float(row[k+':Y'])])
				for k in p2_body_keys:
					p2_traj[-1].append([-float(row[k+':X']), float(row[k+':Z']), float(row[k+':Y'])])
				object_traj.append([-float(row[object_key+':X']), float(row[object_key+':Z']), float(row[object_key+':Y'])])
				label.append(label_idx[row['Giver']])
			p1_traj = np.array(p1_traj, dtype=np.float32)
			p2_traj = np.array(p2_traj, dtype=np.float32)
			object_traj = np.array(object_traj, dtype=np.float32)

			if p1_traj[0, joints_dic["LHand"], 0] > p2_traj[0, joints_dic["RHand"], 0]:
				p1_traj[:, :, 0] *= -1
				p2_traj[:, :, 0] *= -1
				object_traj[:, 0] *= -1

				p1_traj[:, :, 1] *= -1
				p2_traj[:, :, 1] *= -1
				object_traj[:, 1] *= -1

			origin = p1_traj[0:1, joints_dic["Hip"]].copy()
			for t in range(p1_traj.shape[0]):
				p1_traj[t] = p1_traj[t] - origin
				p2_traj[t] = p2_traj[t] - origin
			object_traj = object_traj - origin
			if args.robot:
				offset = np.array([1,0,0.3])
				for t in range(p1_traj.shape[0]):
					p1_traj[t] = p1_traj[t] + offset
					p2_traj[t] = p2_traj[t]
				object_traj = object_traj + offset

				p1_traj[:, :, 0] *= 0.5
				object_traj[:, 0] *= 0.5

				p2_traj[:, :, 0] += 0.1
				p2_traj[:, :, 2] *= 1.2
				p2_traj[:, :, 2] += 0.1

			p1_trajs.append(p1_traj)
			p2_trajs.append(p2_traj)
			object_trajs.append(object_traj)
			labels.append(np.array(label, dtype=int))
		except Exception as e:
			# print(f'Error encountered: {e.__str__()}\nSkipping file {f_csv}')
			continue
	p1_trajs = np.array(p1_trajs, dtype=object)
	p2_trajs = np.array(p2_trajs, dtype=object)
	object_trajs = np.array(object_trajs, dtype=object)
	labels = np.array(labels, dtype=object)
	print(p1_trajs.shape, p2_trajs.shape, object_trajs.shape, labels.shape)
	return p1_trajs, p2_trajs, object_trajs, labels


train_dirs = ['P07_P08', 'P09_P10', 'P11_P12', 'P13_P14', 'P15_P16', 'P17_P18', 'P19_P20', 'P21_P22', 'P23_P24', 'P25_P26']
test_dirs = ['P27_P28', 'P29_P30']

train_files = []
for d in train_dirs:
	files = glob.glob(os.path.join(args.src, 'Bimanual Handovers Dataset', f'{d}', 'OptiTrack_Global_Frame','*double*.csv'))
	files.sort()
	train_files += files
for d in train_dirs:
	files = glob.glob(os.path.join(args.src, 'Bimanual Handovers Dataset', f'{d}', 'OptiTrack_Global_Frame','*single*.csv'))
	files.sort()
	train_files += files
train_data = preproc_files_list(train_files)

test_files = []
for d in test_dirs:
	files = glob.glob(os.path.join(args.src, 'Bimanual Handovers Dataset', f'{d}', 'OptiTrack_Global_Frame','*double*.csv'))
	files.sort()
	test_files += files
for d in test_dirs:
	files = glob.glob(os.path.join(args.src, 'Bimanual Handovers Dataset', f'{d}', 'OptiTrack_Global_Frame','*single*.csv'))
	files.sort()
	test_files += files
test_data = preproc_files_list(test_files)

np.savez_compressed(os.path.join(args.dst,'traj_data.npz'), train_data=train_data, test_data=test_data, joints=joints)