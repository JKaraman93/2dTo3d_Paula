
## Task
> 3D pose estimation after 2D keypoints detection on signing videos.



## Models
|Approach | 2d Detector| 2d keypoints format(input) | 3d Detector  | 3d keypoints format(output) |
| :-------------: | :-------------: |:-------------:| :-----:|:-----:|
| *openpose*  | openpose[^1]  | H3.6m | Videopose3D(pretrained_h36m_cpn.bin) | H3.6m |
| *detectron* | detectron | COCO[^2]  | Videopose3D(pretrained_h36m_detectron_coco.bin) | H3.6m |
| *mpcoco* | mediapipe | COCO  | Videopose3D(pretrained_h36m_detectron_coco.bin) | H3.6m |
| *mp3d* | mediapipe | COCO | mediapipe | H3.6m [^3]|

[^1]: Midhip and Spine were artificially generated using information from adjacent keypoints.
The thorax was lowered to the height of the shoulders, with a movement of -35 pixels.
[^2]: In COCO approaches, the predicted skeleton is shifted by 6cm along the z-axis(height).
[^3]: Midhip, Spine, Thorax and Headtop were artificially generated using information from adjacent keypoints.

## Keypoints Format
* 17 keypoints
* 
<img src="img/fullbody_coco.png" width="300" height="400"><img src="img/fullbody_h36m.png" width="300" height="400">

## Results 


## Demo Instructions
Run on Google Colab :
https://colab.research.google.com/drive/1s7ASDuQjFfxZzCKvrpeWO1LBhn26Nd-u?usp=share_link



