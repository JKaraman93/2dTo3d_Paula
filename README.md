
## Task
> 3D pose estimation after 2D keypoints detection on sign language videos.


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
 
<img src="img/fullbody_coco.png" width="300" height="400"><img src="img/fullbody_h36m.png" width="300" height="400">

## Results 
| Videos |EUDs1| | | |EUDs2| | ||EUDs3| | | |EUDs4| | ||EUDs5| | ||
| :---: |  :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
|Errors |MPJPE (mm)|x|y|z|MPJPE (mm)|x|y|z|MPJPE (mm)|x|y|z|MPJPE (mm)|x|y|z|MPJPE (mm)|x|y|z|
|*openpose*|91|59|37|45|78|53|33|33|101|57|39|61|80|51|25|47|96|58|48|40|
|*mp3d*|104|34|73|48|82|31|42|47|101|34|69|42|87|32|54|43|95|30|62|45|
|*detectron*|73|30|38|44|81|33|35|56|78|32|43|43|75|33|32|50|83|33|47|49|
|*mpcoco*|70|29|42|35|72|31|38|41|75|31|46|35|72|33|38|38|72|32|45|35|

## Demo Instructions
Run on Google Colab :
https://colab.research.google.com/drive/1s7ASDuQjFfxZzCKvrpeWO1LBhn26Nd-u?usp=share_link



