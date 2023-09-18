## create npz file for 2d detections
import mediapipe as mp
import cv2
import numpy as np
from common.arguments import parse_args

args = parse_args()
video_name = args.viz_subject

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

select_marks= ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
                             'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip',
                             'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']        # coco format

vidcap = cv2.VideoCapture('inference/'+video_name[:-4]+'/'+video_name)  #testExportV2FullFront

fr_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
fr_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fr_size = (fr_width, fr_height)
#out = cv2.VideoWriter('rgb_mediapipe.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 30, fr_size)

print('Total frames:', int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)))
print('FPS :', int(vidcap.get(cv2.CAP_PROP_FPS)))
# ret = vidcap.set(cv2.CAP_PROP_FRAME_WIDTH,320)

count = -1
timestamp_list = []
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
            s = s.upper()  #coco
            ldmrk = results.pose_landmarks.landmark[getattr(mp_pose.PoseLandmark,s)]#coco
            landx = ldmrk.x * image_width
            landy = ldmrk.y * image_hight
           #if s in ['LEFT_HIP', 'RIGHT_HIP']:
            #    landy -= 20
            ldmarks.append([landx,landy])
        ldmarks_all.append(ldmarks)
pred_mediapipe = np.array(ldmarks_all)

metadata = {
		'layout_name': 'coco',
			'num_joints': 17,
		'keypoints_symmetry': [	[1, 3, 5, 7, 9, 11, 13, 15],[2, 4, 6, 8, 10, 12, 14, 16],],
		'video_metadata': {video_name: {'w': fr_width, 'h': fr_height}}}


positions_2d = {
	video_name:
			{
			 'custom':[pred_mediapipe]}}

np.savez_compressed('inference/'+video_name[:-4]+'/mpcoco', positions_2d=positions_2d, metadata=metadata)
np.save('inference/'+video_name[:-4]+'/mpcoco.npy', pred_mediapipe, allow_pickle=True)



