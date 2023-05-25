testExportV2.csv  # original 3d world coordinates
testExportV2Front_Camspace.csv # 3d camera coords after transformation from world
testExportV2Front_pred3d_cam.npy # output in camera space
testExportV2Front_pred3d_world.npy  # output in world space
testExportV2Front_Rasterspace.csv #2d raster coords in pixel
testExportV2_world.csv  # 3d world coordinates after modifications according to task
testExportV2_world.npy # same as above in np array
testExportV2Front_error3d.csv  # error per joint per frame
testExportV2Front_error3d_xyz.csv  # error per component of joint per frame


mp3d : predictions of mediapipe
		: corrected 3d GT L/R/C hip corrdinate Z from 113.9mm -> 120mm
mediapipe : predictions of Videopose with mediapipe 2d input in h36m format
openpose : predictions of Videopose with openpose  2d input (thorax_y +35) and  (R/L shoulder_x +-5) in h36m format
mp coco : predictions of Videopose with mediapipe 2d input in coco format
		: corrected 3d GT L/R/C hip corrdinate Z from 113.9mm -> 120mm
detectron : predictions of Videopose with detectron 2d input in coco format
		: corrected 3d GT L/R/C hip corrdinate Z from 113.9mm -> 120mm


## Full Body ##

h36m joints : ['HIP', 'R_HIP', 'R_KNEE', 'R_FOOT', 'L_ HIP', 'L_KNEE', 'L_FOOT', 'SPINE','THORAX',
 'NOSE', 'HEAD', 'L_SHOULDER', 'L_ELBOW', 'L_WRIST','R_SHOULDER', 'R_ELBOW', 'R_WRIST']
 
detectron kps : ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
 'right_shoulder',   'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip','right_hip', 
 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']


### Custom dataset  ###
data  (.npz)
	metadata (dict)
		{'layout_name': 'coco',
		'num_joints': 17,
		'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15],	[2, 4, 6, 8, 10, 12, 14, 16]],
		'video_metadata': {'testExportV2FullFront.mp4': {'w': 600, 'h': 800}}}
	positions_2d (dict)
		{'testExportV2FullFront.mp4':
			{'custom': [array([[[298.1093  , 101.12254 ],  [313.10107 ,  86.89055 ],    [283.11758 ,  86.89055 ],   ...,   [253.1341  , 542.3145  ],}} # list with array (61,17,2)
			
			
### H36m ###
'data/data_3d_h36m.npz' : 'positions_3d' (dict 7)

dataset
self._cam (dict 10)
	'S1' (list 4)  #subject
		0 (dict 11)  #camera
			'orientation' np.array 4
			'translation' np.array 3 
			res_w
			res_h
			azimuth np.array () =70
			intrinsic np.array 9 (whitch contains focal length(2), center (2), radial_distortion (3), tangential_distortion(2)
					
		1 (dict 11) 
		...



data (dict 7)
	'S1' dict 30  #subject
		Directions 1 np.array (1383,32,3) #actions
		...
		
(using the above)
self._data (dict 7 )
	'S1' dict 30  #subject
		'Directions 1' (dict 2)
			positions np.array (1383,32,3)
			cameras (list 4)   # like self._cam for each subject
		
	
	
