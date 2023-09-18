# -*- coding: utf-8 -*-
"""
The videopose3d output was converted in order to adjust to camera space of paula.
"""

#   -k mpcoco --viz-subject EUDs1Front.mp4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import common.loss as loss
from common.arguments import parse_args
args = parse_args()
vid = args.viz_subject


fb3d_np = np.load('inference/'+vid[:-4]+'/' +args.keypoints +  '_predictions/'+vid[:-4]+'_pred3d_world.npy')
dict_keypoints = ['hips', 'R_HIP', 'R_KNEE', 'R_FOOT', 'L_HIP', 'L_KNEE', 'L_FOOT', 'SPINE','THORAX', 'NOSE', 'HEAD', 'leftshoulder', 'leftelbow', 'leftwrist',
                      'rightshoulder', 'rightelbow', 'rightwrist']
jxyz = []
for j in dict_keypoints:
    jxyz.extend([j+'X',j+'Y',j+'Z'])

fb3d = fb3d_np.reshape(fb3d_np.shape[0],fb3d_np.shape[1]*3)
fb3d = pd.DataFrame(fb3d, columns = jxyz)  ## in meters (m)

world_space = pd.read_csv('inference/'+vid[:-4]+'/'+vid[:-4]+'_world.csv')
world_space = world_space.div(100) # from cm to meters


N = 0
epoch_loss_3d_pos_procrustes = 0
epoch_loss_3d_pos = 0

body_keypoints_selected = ['leftshoulder','rightshoulder','leftelbow','rightelbow','leftwrist','rightwrist']# '11':'lefthip','12':'righthip',}
body_keypoints_selected_xyz = []
for b in body_keypoints_selected:
  body_keypoints_selected_xyz.append(b+'X')
  body_keypoints_selected_xyz.append(b+'Y')
  body_keypoints_selected_xyz.append(b+'Z')

inputs_3d_ = world_space[body_keypoints_selected_xyz].to_numpy()
inputs_3d_ = inputs_3d_.reshape(len(world_space.index),len(body_keypoints_selected),3)
inputs_3d = torch.from_numpy(inputs_3d_)
inputs_3d = inputs_3d.reshape(1,inputs_3d.shape[0],inputs_3d.shape[1],inputs_3d.shape[2])

predicted_3d = fb3d[body_keypoints_selected_xyz].to_numpy()
predicted_3d = predicted_3d.reshape(len(fb3d.index),len(body_keypoints_selected),3)
predicted_3d_pos  = torch.from_numpy(predicted_3d)
predicted_3d_pos = predicted_3d_pos.reshape(1,predicted_3d.shape[0],predicted_3d.shape[1],predicted_3d.shape[2])

error = loss.mpjpe(predicted_3d_pos, inputs_3d).cpu().numpy()
e1 = round(error *1000,2)
print ('mpjpe',e1, 'mm')

inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
e2 = loss.p_mpjpe(predicted_3d_pos, inputs)*1000
print ('p_mpjpe', round(e2,2),'mm')

dict_keypoints_error = pd.DataFrame()
dict_keypoints_xyz_error = pd.DataFrame()

for c in body_keypoints_selected:
  for frame in range (len(world_space.index)):
    # calculate Manhatan distance for each joint
    error_x = np.abs(world_space.loc[frame,c+'X'] - fb3d.loc[frame,c+'X'])
    error_y = np.abs(world_space.loc[frame,c+'Y'] - fb3d.loc[frame,c+'Y'])
    error_z = np.abs(world_space.loc[frame,c+'Z'] - fb3d.loc[frame,c+'Z'])
    dict_keypoints_xyz_error.loc[frame,c+'X'] = error_x
    dict_keypoints_xyz_error.loc[frame,c+'Y'] = error_y
    dict_keypoints_xyz_error.loc[frame,c+'Z'] = error_z

    # calculate Euclidian distance for each joint
    dict_keypoints_error.loc[frame, c] = np.sqrt(error_x**2+error_y**2+error_z**2)
dict_keypoints_error = dict_keypoints_error*1000 # meters to milimeters   
dict_keypoints_xyz_error = dict_keypoints_xyz_error*1000  # meters to milimeters


mean_error_kps_xyz = dict_keypoints_xyz_error.mean(axis=0)
print ('mean_error_kps_xyz:\n',mean_error_kps_xyz)

mean_error_kps = dict_keypoints_error.mean(axis=0)
print ('mean_error_kps:\n',mean_error_kps)

mean_error_frame = dict_keypoints_error.mean(axis=1)
print ('\nmean_error_frame:\n',mean_error_frame)

total_mean_error = (mean_error_frame).mean()
print ('\ntotal_mean_error:\n',total_mean_error)


## save results ##
dict_keypoints_xyz_error.to_csv('inference/'+vid[:-4]+'/' +args.keypoints +  '_predictions/'+vid[:-4]+'_error3d_xyz.csv', index=False)
dict_keypoints_error.to_csv('inference/'+vid[:-4]+'/' +args.keypoints +  '_predictions/'+vid[:-4]+'_error3d.csv',index=False)

