from __future__ import print_function, division
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

#################################################
	# Energy consumption dataset class
#################################################
class Energy_consumption(Dataset):
	def __init__(self, csv_file):
		self.energy_consumption_data = np.array(pd.read_csv(csv_file))

	def get_data(self, index):			# index = 2 for voice rehearsal and index = 3 for cockery programme
		data = self.energy_consumption_data[:, index].astype(np.float32)
		return data

# index = 3
# csv_file = '../dataset/energy_consumption_dataset.csv'
# dataset = Energy_consumption(csv_file)
# data = dataset.get_data(index)
# print(data)
