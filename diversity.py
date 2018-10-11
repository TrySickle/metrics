import dataset_loading
import numpy as np
import functions

vr_data = dataset_loading.read_timeseries_data_only_train('VR_data')
print(vr_data.train.images, np.shape(vr_data.train.images))
print(vr_data.train.images[0].flatten(), np.shape(vr_data.train.images[0].flatten()))