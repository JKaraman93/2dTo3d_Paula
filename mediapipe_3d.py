import matplotlib.pyplot as plt

import functions
import mediapipe as mp
import cv2
import pandas as pd
import numpy as np
from common.arguments import parse_args
import numpy as np
from common.render_animation_mp3d import render_animation
import os
import shutil


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

args = parse_args()
video_name = args.viz_subject
try:
    os.mkdir('inference/'+video_name[:-4]+'/'+'mp3d_predictions')
except:
    print ('Folder already exists!')

ldmarks=[]
select_marks = [24,26,28,23,25,27,0,11,13,15,12,14,16]

vidcap = cv2.VideoCapture('inference/'+video_name[:-4]+'/'+video_name)  #testExportV2FullFront

fr_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
fr_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fr_size = (fr_width, fr_height)
#out = cv2.VideoWriter('testExportV2FullFront_mediapipe.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 30, fr_size)

print('Total frames:', int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)))
print('FPS :', int(vidcap.get(cv2.CAP_PROP_FPS)))
# ret = vidcap.set(cv2.CAP_PROP_FRAME_WIDTH,320)

ldmarks_all = []
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False,
                  model_complexity=2, ) as pose:
    while vidcap.isOpened():
        ldmarks = []
        ret, frame = vidcap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image_hight, image_width, _ = image.shape
        for s in select_marks:
            ldmrk = results.pose_world_landmarks.landmark[s]
            ldmarks.append([ldmrk.x ,ldmrk.y ,ldmrk.z])

        nose = ldmarks[6]
        lhip= ldmarks[3]
        rhip = ldmarks[0]
        rsh = ldmarks[10]
        lsh = ldmarks[7]

        hip, spine, thorax, headbase, headtop = functions.artficial_joints3d (nose, lhip, rhip, rsh, lsh)

        ldmarks.insert(0, hip)
        ldmarks.insert(7, spine)
        #ldmarks.insert(4, thorax)
        ldmarks.insert(8, headbase)
        ldmarks.insert(10, headtop)
        ldmarks_all.append(ldmarks)


        # Plot pose world landmarks.


pred_mediapipe = np.array(ldmarks_all)
ground_truth_world = np.float32(np.load('inference/'+video_name[:-4]+'/'+video_name[:-4]+'_world_correctedhips.npy'))/100 #cm to meters
pred_mediapipe[:,:,:] = pred_mediapipe * (-1)
temp = np.copy(pred_mediapipe[:,:,2])
pred_mediapipe[:,:,2] = pred_mediapipe[:,:,1]
pred_mediapipe[:,:,1] = temp
trajectory = ground_truth_world[:, :1]
pred_mediapipe += trajectory


np.save('inference/'+video_name[:-4]+'/'+'mp3d_predictions/'+video_name[:-4] +'_pred3d_world.npy', pred_mediapipe, allow_pickle=True)


keypoints_metadata = {
    'layout_name': 'coco',
    'num_joints': 17,
    'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16], ],
    'video_metadata': {video_name: {'w': fr_width, 'h': fr_height}}}

input_keypoints = np.load('inference/'+video_name[:-4]+'/mpcoco.npy',allow_pickle=True)

anim_output = {}

'''pred_mediapipe = np.insert(pred_mediapipe, 2, pred_mediapipe[:, 1, :], axis=1)
pred_mediapipe = np.insert(pred_mediapipe, 3, pred_mediapipe[:, 1, :], axis=1)
pred_mediapipe = np.insert(pred_mediapipe, 5, pred_mediapipe[:, 4, :], axis=1)
pred_mediapipe = np.insert(pred_mediapipe, 6, pred_mediapipe[:, 4, :], axis=1)'''

anim_output['Reconstruction'] = pred_mediapipe
anim_output['Ground truth'] = ground_truth_world


render_animation(input_keypoints, keypoints_metadata, anim_output, None, 3000, 70, 'inference/'+video_name[:-4]+'/'+'mp3d_predictions/output.mp4', viewport=(600, 800),
                 limit=-1, downsample=1, size=6,  input_video_path='inference/'+video_name[:-4]+'/'+video_name,  input_video_skip=0,)
shutil.copyfile('inference/'+video_name[:-4]+'/'+'mp3d_predictions/output.mp4',
                'static/output_' + video_name[:-4] + '_mp3d.mp4')
