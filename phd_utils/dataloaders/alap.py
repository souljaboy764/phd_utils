import numpy as np
import os
from torch.utils.data import Dataset

## In all cases, only Robot-to-Human Handovers are considered where the Human (Observed) Agent is the Receiver and the Robot (Controlled) Agent is the Giver.

# Dataset class for the Human-Human Interaction data in https://zenodo.org/record/7767535#.ZB2-43bMLIU
class HHDataset(Dataset):
	def __init__(self, train=True, window_length=5, downsample=1):
		with np.load(os.path.join(os.path.dirname(__file__),'..','..','data_preproc','alap','traj_data.npz'), allow_pickle=True) as data:
			if train:
				p1_trajs, p2_trajs, _, _ = data['train_data']
				self.actidx = np.array([[0,105], [105, 168]])
			else:
				p1_trajs, p2_trajs, _, _ = data['test_data']
				self.actidx = np.array([[0,12], [12, 24]])
			joints = data['joints']
		
		joints_dic = {joints[i]:i for i in range(len(joints))}
		joints_idx = [joints_dic[i] for i in ['LUArm', 'LFArm', 'LHand', 'RUArm', 'RFArm', 'RHand']]
		self.input_data = []
		self.output_data = []
		self.traj_data = []
		for i in range(len(p1_trajs)):
			p1_pos = p1_trajs[i][::4, joints_idx]
			p1_pos = p1_pos.reshape((p1_pos.shape[0], 3*len(joints_idx)))
			p2_pos = p2_trajs[i][::4, joints_idx]
			p2_pos = p2_pos.reshape((p2_pos.shape[0], 3*len(joints_idx)))

			p1_vel = np.diff(p1_pos, axis=0, prepend=p1_pos[0:1])
			p2_vel = np.diff(p2_pos, axis=0, prepend=p2_pos[0:1])

			p1_traj = np.hstack([p1_pos, p1_vel])
			p2_traj = np.hstack([p2_pos, p2_vel])
			
			self.input_data.append(p2_traj)
			self.output_data.append(p1_traj)

			self.traj_data.append(np.concatenate([self.input_data[-1], self.output_data[-1]], axis=-1))

		self.input_data = np.array(self.input_data, dtype=object)
		self.output_data = np.array(self.output_data, dtype=object)
		self.traj_data = np.array(self.traj_data, dtype=object)
		self.input_dims = self.input_data[0].shape[-1]
		self.output_dims = self.output_data[0].shape[-1]

		self.len = len(self.input_data)
		
		# For mild_hri
		self.labels = np.zeros(self.len)
		for idx in range(len(self.actidx)):
			self.labels[self.actidx[idx][0]:self.actidx[idx][1]] = idx

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return np.hstack([self.input_data[index], self.output_data[index]]).astype(np.float32), self.labels[index].astype(np.int32)

# A temporal window over the inputs of HHDataset
class HHWindowDataset(Dataset):
	def __init__(self, train=True, window_length=5, downsample=1):
		with np.load(os.path.join(os.path.dirname(__file__),'..','..','data_preproc','alap','traj_data.npz'), allow_pickle=True) as data:
			if train:
				p1_trajs, p2_trajs, _, _ = data['train_data']
				self.actidx = np.array([[0,105], [105, 168]])
			else:
				p1_trajs, p2_trajs, _, _ = data['test_data']
				self.actidx = np.array([[0,12], [12, 24]])
			joints = data['joints']
		
		joints_dic = {joints[i]:i for i in range(len(joints))}
		joints_idx = [joints_dic[i] for i in ['LUArm', 'LFArm', 'LHand', 'RUArm', 'RFArm', 'RHand']]
		self.input_data = []
		self.output_data = []
		self.traj_data = []
		for i in range(len(p1_trajs)):
			p1_pos = p1_trajs[i][::4, joints_idx]
			p1_pos = p1_pos.reshape((p1_pos.shape[0], 3*len(joints_idx)))
			p2_pos = p2_trajs[i][::4, joints_idx]
			p2_pos = p2_pos.reshape((p2_pos.shape[0], 3*len(joints_idx)))

			p1_vel = np.diff(p1_pos, axis=0, prepend=p1_pos[0:1])
			p2_vel = np.diff(p2_pos, axis=0, prepend=p2_pos[0:1])

			p1_traj = np.hstack([p1_pos, p1_vel])
			p2_traj = np.hstack([p2_pos, p2_vel])
			
			input_dim = output_dim = p2_traj.shape[-1]
			seq_len = p2_traj.shape[0]
			idx = np.array([np.arange(i,i+window_length) for i in range(seq_len + 1 - 2*window_length)])
			self.input_data.append(p2_traj[idx].reshape((seq_len + 1 - 2*window_length, window_length*input_dim)))
			idx = np.array([np.arange(i,i+window_length) for i in range(window_length, seq_len + 1 - window_length)])
			self.output_data.append(p1_traj[idx].reshape((seq_len + 1 - 2*window_length, window_length*output_dim)))

			self.traj_data.append(np.concatenate([self.input_data[-1], self.output_data[-1]], axis=-1))

		self.input_data = np.array(self.input_data, dtype=object)
		self.output_data = np.array(self.output_data, dtype=object)
		self.traj_data = np.array(self.traj_data, dtype=object)
		self.input_dims = self.input_data[0].shape[-1]
		self.output_dims = self.output_data[0].shape[-1]

		self.len = len(self.input_data)
		
		# For mild_hri
		self.labels = np.zeros(self.len)
		for idx in range(len(self.actidx)):
			self.labels[self.actidx[idx][0]:self.actidx[idx][1]] = idx

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return np.hstack([self.input_data[index], self.output_data[index]]).astype(np.float32), self.labels[index].astype(np.int32)

# Handover Dataset class scaled for the Kobo robot
class KoboWindowDataset(Dataset):
	def __init__(self, train=True, window_length=5, downsample=1):
		with np.load(os.path.join(os.path.dirname(__file__),'..','..','data_preproc','alap_kobo','traj_data.npz'), allow_pickle=True) as data:
			if train:
				p1_trajs, p2_trajs, _, _ = data['train_data']
				self.actidx = np.array([[0,105], [105, 168]])
			else:
				p1_trajs, p2_trajs, _, _ = data['test_data']
				self.actidx = np.array([[0,12], [12, 24]])
			joints = data['joints']
		
		joints_dic = {joints[i]:i for i in range(len(joints))}
		p1_joints_idx = [joints_dic[i] for i in ['LHand', 'RHand']]
		# p2_joints_idx = [joints_dic[i] for i in ['LUArm', 'LFArm', 'LHand', 'RUArm', 'RFArm', 'RHand']]
		p2_joints_idx = [joints_dic[i] for i in ['LHand', 'RHand']]
		self.input_data = []
		self.output_data = []
		self.traj_data = []
		for i in range(len(p1_trajs)):
			p1_pos = p1_trajs[i][::4, p1_joints_idx]
			p1_pos = p1_pos.reshape((p1_pos.shape[0], 3*len(p1_joints_idx)))
			p2_pos = p2_trajs[i][::4, p2_joints_idx]
			p2_pos = p2_pos.reshape((p2_pos.shape[0], 3*len(p2_joints_idx)))

			p1_traj = p1_pos

			p2_vel = np.diff(p2_pos, axis=0, prepend=p2_pos[0:1])
			p2_traj = np.hstack([p2_pos, p2_vel])
			
			input_dim = output_dim = p2_traj.shape[-1]
			seq_len = p1_traj.shape[0]
			idx = np.array([np.arange(i,i+window_length) for i in range(seq_len + 1 - 2*window_length)])
			self.input_data.append(p2_traj[idx].reshape((seq_len + 1 - 2*window_length, window_length*input_dim)))
			idx = np.array([np.arange(i,i+window_length) for i in range(window_length, seq_len + 1 - window_length)])
			self.output_data.append(p1_traj[idx].reshape((seq_len + 1 - 2*window_length, window_length*output_dim)))

			self.traj_data.append(np.concatenate([self.input_data[-1], self.output_data[-1]], axis=-1))

		self.input_data = np.array(self.input_data, dtype=object)
		self.output_data = np.array(self.output_data, dtype=object)
		self.traj_data = np.array(self.traj_data, dtype=object)
		self.input_dims = self.input_data[0].shape[-1]
		self.output_dims = self.output_data[0].shape[-1]

		self.len = len(self.input_data)
		
		# For mild_hri
		self.labels = np.zeros(self.len)
		for idx in range(len(self.actidx)):
			self.labels[self.actidx[idx][0]:self.actidx[idx][1]] = idx

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return np.hstack([self.input_data[index], self.output_data[index]]).astype(np.float32), self.labels[index].astype(np.int32)
