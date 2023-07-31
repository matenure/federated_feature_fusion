import numpy as np
test_ratio = 0.2
np.random.seed(123)

# data = np.load('./metr-la.npz')
data = np.load('./pems-bay.npz')
num_sample = data['x'].shape[0]
shuffled_index = np.random.permutation(num_sample)
cut_point1 = int(num_sample*(1-test_ratio))
cut_point2 = int(num_sample * (1 - 0.1 - test_ratio))
test_index = shuffled_index[cut_point1:]
val_index = shuffled_index[cut_point2:cut_point1]
train_index = shuffled_index[:cut_point2]
# np.savez('./metr-la_indexed.npz', x = data['x'], y=data['y'], train_index=train_index, val_index=val_index, test_index=test_index)
np.savez('./pems-bay_indexed.npz', x = data['x'], y=data['y'], train_index=train_index, val_index=val_index, test_index=test_index)
