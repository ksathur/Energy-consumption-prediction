import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sys

################################################
	# Importing functions from other directory
################################################
sys.path.append('utils')
from dataset_class import *
from utils import *
from model import *

#################################################
	# Data loading
#################################################
index = 2
csv_file = 'dataset/energy_consumption_dataset.csv'
dataset = Energy_consumption(csv_file)
data = dataset.get_data(index)

#################################################
	# Data Visualization
#################################################
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size

plt.title('Voice Rehearsal Energy Consumption(Actual)')
plt.ylabel('Energy Consumption (kWh)')
plt.xlabel('Day')
plt.xticks(np.arange(0, len(data)+1, 5))
plt.grid(True)
plt.plot(np.arange(len(data)), data)
# plt.autoscale(axis='x',tight=True)
# plt.plot(data)
plt.savefig('dumps/accuracy/voice_rehearsal_actual.png')
plt.show()

#################################################
	# Data Preprocessing
#################################################
test_data_size = 7
train_data = data[:-test_data_size]
test_data = data[-test_data_size:]
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
train_window = 28

train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

################################################
	# Model and training parameters
################################################
model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 150

################################################
	# Training
################################################
for i in range(epochs):
	for seq, labels in train_inout_seq:
		optimizer.zero_grad()
		model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
						torch.zeros(1, 1, model.hidden_layer_size))

		y_pred = model(seq)

		single_loss = loss_function(y_pred, labels)
		single_loss.backward()
		optimizer.step()

	print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

################################################
	# Testing
################################################
fut_pred = 7
test_inputs = train_data_normalized[-train_window:].tolist()

model.eval()

for i in range(fut_pred):
	seq = torch.FloatTensor(test_inputs[-train_window:])
	with torch.no_grad():
		model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
						torch.zeros(1, 1, model.hidden_layer_size))
		test_inputs.append(model(seq).item())
# print(test_inputs[fut_pred:])
actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))
# print(actual_predictions)

################################################
	# Plotting outputs
################################################
x = np.arange(133, 140, 1)

plt.title('Voice Rehearsal Energy Consumption')
plt.ylabel('Energy Consumption (kWh)')
plt.xlabel('Day')
plt.grid(True)
# plt.autoscale(axis='x', tight=True)
plt.plot(data, label = 'Actual')
plt.plot(x, actual_predictions, label = 'Predicted')
plt.gca().legend()
plt.savefig('dumps/accuracy/voice_rehearsal_actual_and_predicted.png')
plt.show()

plt.title('Voice Rehearsal Energy Consumption')
plt.ylabel('Energy Consumption (kWh)')
plt.xlabel('Day')
plt.grid(True)
# plt.autoscale(axis='x', tight=True)
plt.plot(x, data[-fut_pred:], label = 'Actual')
plt.plot(x, actual_predictions, label = 'Predicted')
plt.gca().legend()
plt.savefig('dumps/accuracy/voice_rehearsal_actual_and_predicted_zoom.png')
plt.show()

################################################
	# Dumping files
################################################
file_name = 'dumps/array/voice_rehearsal_inputs_and_outputs.npz'
actual_data = data
predicted_data =  np.asarray(data[:-fut_pred].tolist() + [actual_predictions[i][0] for i in range(len(actual_predictions))])
np.savez(file_name, actual_data, predicted_data)
