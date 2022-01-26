import numpy as np
import matplotlib.pyplot as plt

voice_rehearsal_inputs_and_outputs = np.load('dumps/array/voice_rehearsal_inputs_and_outputs.npz')
cockery_inputs_and_outputs = np.load('dumps/array/cockery_inputs_and_outputs.npz')
################################################
	# Daily predictions and plotting
################################################
x = np.arange(133, 140, 1)

plt.title('Daily Energy Consumption')
plt.ylabel('Energy Consumption (kWh)')
plt.xlabel('Day')
plt.grid(True)
# plt.autoscale(axis='x', tight=True)
plt.plot((voice_rehearsal_inputs_and_outputs['arr_0'] + cockery_inputs_and_outputs['arr_0']), label = 'Actual')
plt.plot(x, (voice_rehearsal_inputs_and_outputs['arr_1'] + cockery_inputs_and_outputs['arr_1'])[-7:], label = 'Predicted')
plt.gca().legend()
plt.savefig('dumps/accuracy/daily_actual_and_predicted.png')
plt.show()

plt.title('Daily Energy Consumption')
plt.ylabel('Energy Consumption (kWh)')
plt.xlabel('Day')
plt.grid(True)
# plt.autoscale(axis='x', tight=True)
plt.plot(x, (voice_rehearsal_inputs_and_outputs['arr_0'] + cockery_inputs_and_outputs['arr_0'])[-7:], label = 'Actual')
plt.plot(x, (voice_rehearsal_inputs_and_outputs['arr_1'] + cockery_inputs_and_outputs['arr_1'])[-7:], label = 'Predicted')
plt.gca().legend()
plt.savefig('dumps/accuracy/daily_actual_and_predicted_zoom.png')
plt.show()

################################################
	# Weekly predictions
################################################
print('Next week total power consumption: ', np.sum((voice_rehearsal_inputs_and_outputs['arr_1'] + cockery_inputs_and_outputs['arr_1'])[-7:]))


