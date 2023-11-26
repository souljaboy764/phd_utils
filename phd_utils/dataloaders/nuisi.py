from torch.utils.data import Dataset
import numpy as np
import os

from phd_utils.data import *

# Dataset class for the Human-Human Interaction data in https://github.com/souljaboy764/nuisi_dataset/
class HHDataset(Dataset):
	def __init__(self, train=True, downsample=None): # downsample only needed for compatibility
		with np.load(os.path.join(os.path.dirname(__file__),'..','..','data_preproc','nuisi','traj_data.npz'), allow_pickle=True) as data:
			if train:
				traj_data = data['train_data']
				self.actidx = np.array([[0, 9], [9, 17], [17, 26], [26, 33]])

			else:
				traj_data = data['test_data']
				self.actidx = np.array([[0, 3], [3, 6], [6, 9], [9, 11]])
			
			self.traj_data = []
			for i in range(len(traj_data)):
				seq_len, dims = traj_data[i].shape
				traj_1 = traj_data[i][:, :dims//2]
				traj_2 = traj_data[i][:, dims//2:]
				# traj_1 = self.traj_data[i][:, dims//2-3:dims//2]
				# traj_2 = self.traj_data[i][:, -3:]

				vel_1 = np.diff(traj_1, axis=0, prepend=traj_1[0:1,:])
				vel_2 = np.diff(traj_2, axis=0, prepend=traj_2[0:1,:])

				self.traj_data.append(np.concatenate([traj_1, vel_1, traj_2, vel_2], axis=-1))
				# self.traj_data.append(np.concatenate([traj_2, vel_2, traj_1, vel_1], axis=-1))
			
			self.len = len(self.traj_data)
			self.labels = np.zeros(self.len)
			for idx in range(len(self.actidx)):
				self.labels[self.actidx[idx][0]:self.actidx[idx][1]] = idx

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return self.traj_data[index].astype(np.float32), self.labels[index].astype(np.int32)

# Dataset class wrapping the HHDataset class for using a temporal window of observations
class HHWindowDataset(Dataset):
	def __init__(self, train=True, window_length=5, downsample=None): # downsample only needed for compatibility
		dataset = HHDataset(train)
		self.actidx = dataset.actidx
		self.traj_data = window_concat(dataset.traj_data, window_length)
		self.len = len(self.traj_data)
		
		# For mild_hri
		self.labels = np.zeros(self.len)
		for idx in range(len(self.actidx)):
			self.labels[self.actidx[idx][0]:self.actidx[idx][1]] = idx

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return self.traj_data[index].astype(np.float32), self.labels[index].astype(np.int32)

# Dataset class wrapping the HHDataset class for extracting Pepper's joint angles from the human skeletons
class PepperDataset(HHDataset):
	def __init__(self, train=True, downsample=None):
		super().__init__(train, downsample)
		
		for i in range(len(self.traj_data)):
			seq_len, dims = self.traj_data[i].shape
			traj_r = []

			for frame in self.traj_data[i][:, dims//2:3*dims//4].reshape((seq_len, 3, 3)):
				joints = joint_angle_extraction(frame)
				traj_r.append(joints)

			traj_r = np.array(traj_r) # seq_len, 4

			self.traj_data[i] = np.concatenate([self.traj_data[i][:, :dims//2], traj_r], axis=-1) # seq_len, dims//2 + 4
			# self.traj_data[i] = np.concatenate([self.traj_data[i][:, dims//4-3:dims//4], self.traj_data[i][:, dims//2-3:dims//2], traj_r], axis=-1) # seq_len, dims//2 + 4

# Dataset class wrapping the HHWindowDataset class and abstracting the PepperDataset class for using a temporal window of observations
class PepperWindowDataset(HHWindowDataset):
	def __init__(self, train=True, window_length=5, downsample = 1):
		self._dataset = PepperDataset(train, downsample)
		self.actidx = self._dataset.actidx
		self.traj_data = window_concat(self._dataset.traj_data, window_length, 'pepper')
		self.len = len(self.traj_data)
		
		# For mild_hri
		self.labels = np.zeros(self.len)
		for idx in range(len(self.actidx)):
			self.labels[self.actidx[idx][0]:self.actidx[idx][1]] = idx
