import pandas as pd
import numpy as np

test = pd.read_csv('corrected_hips_detectron/testExportV2_world_correctedhips.csv')
test = test.to_numpy()
test = test.reshape(test.shape[0],test.shape[1]//3,3)
np.save('corrected_hips_detectron/testExportV2_world_correctedhips.npy',test,allow_pickle=True)
