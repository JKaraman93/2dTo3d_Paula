## 10.5.23 ##
# take the .npy file with 2d predictions 17kps from openpose
# and convert in npz format to give it as input to Videopose

import numpy as np
from common.arguments import parse_args
args = parse_args()
video_name = args.viz_subject


vid_width = 600
vid_height = 800
detector = 'openpose'

kps = np.load('../inference/'+video_name[:-4]+'/'+ detector + '.npy')
#kps[:,8,1] += 35


'''if detector == 'mpcoco':
	metadata = {
		'layout_name': 'coco',
			'num_joints': 17,
		'keypoints_symmetry': [	[1, 3, 5, 7, 9, 11, 13, 15],[2, 4, 6, 8, 10, 12, 14, 16],],
		'video_metadata': {video_name: {'w': vid_width, 'h': vid_height}}}
else:'''
metadata = {
		'layout_name': 'h36m',
			'num_joints': 17,
			'keypoints_symmetry': [[4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16], ],
			'video_metadata': {video_name: {'w': vid_width, 'h': vid_height}}}

positions_2d = {
	video_name:
			{
			 'custom':[kps]}}

np.savez_compressed('../inference/'+video_name[:-4]+'/'+ detector + '.npz', positions_2d=positions_2d, metadata=metadata)
