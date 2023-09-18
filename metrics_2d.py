import pandas as pd
import numpy as np
from common.arguments import parse_args
args = parse_args()
vid = args.viz_subject

world_space = pd.read_csv('inference/'+vid[:-4]+'/'+vid[:-4]+'_Rasterspace.csv')
keypoints = np.load('data/' + args.keypoints + '_'+vid+'.npz', allow_pickle=True)
keypoints = keypoints['positions_2d'].item()
keypoints = keypoints[vid]['custom'][0]
#keypoints = np.load('data/' + args.keypoints + '_h36m.npy', allow_pickle=True)


jxy = []


if args.keypoints in ['detectron', 'mpcoco']:
  dict_keypoints = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'leftshoulder',
                       'rightshoulder', 'leftelbow', 'rightelbow', 'leftwrist', 'rightwrist', 'left_hip',
                       'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
else:  # h3.6m
    dict_keypoints = ['hips', 'R_HIP', 'R_KNEE', 'R_FOOT', 'L_HIP', 'L_KNEE', 'L_FOOT', 'SPINE', 'THORAX', 'NOSE',
                      'HEAD', 'leftshoulder', 'leftelbow', 'leftwrist',
                      'rightshoulder', 'rightelbow', 'rightwrist']

for j in dict_keypoints:
    jxy.extend([j+'X',j+'Y'])
perd_2d = keypoints.reshape(keypoints.shape[0], keypoints.shape[1] * 2)
perd_2d = pd.DataFrame(perd_2d, columns = jxy)

body_keypoints_selected = ['leftshoulder','rightshoulder','leftelbow','rightelbow','leftwrist','rightwrist']# '11':'lefthip','12':'righthip',}
body_keypoints_selected_xy = []
for b in body_keypoints_selected:
  body_keypoints_selected_xy.append(b+'X')
  body_keypoints_selected_xy.append(b+'Y')

dict_keypoints_error = pd.DataFrame()
dict_keypoints_xy_error = pd.DataFrame()

for c in body_keypoints_selected:
  for frame in range (len(world_space.index)):
    # calculate Manhatan distance for each joint
    error_x = np.abs(world_space.loc[frame,c+'X'] - perd_2d.loc[frame, c + 'X'])
    error_y = np.abs(world_space.loc[frame,c+'Y'] - perd_2d.loc[frame, c + 'Y'])
    dict_keypoints_xy_error.loc[frame,c+'X'] = error_x
    dict_keypoints_xy_error.loc[frame,c+'Y'] = error_y

    # calculate Euclidian distance for each joint
    dict_keypoints_error.loc[frame, c] = np.sqrt(error_x**2+error_y**2)

dict_keypoints_xy_error.to_csv('inference/'+vid[:-4]+'/' +args.keypoints +  '_predictions/'+vid[:-4]+'_error2d_xy.csv', index=False)
dict_keypoints_error.to_csv('inference/'+vid[:-4]+'/' +args.keypoints +  '_predictions/'+vid[:-4]+'_error2d.csv',index=False)
