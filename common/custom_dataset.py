# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import copy
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
from common.camera import normalize_screen_coordinates, image_coordinates
from common.h36m_dataset import h36m_skeleton
       

custom_camera_params = {
    'id': None,
    'res_w': None, # Pulled from metadata
    'res_h': None, # Pulled from metadata
    
    # Dummy camera parameters (taken from Human3.6M), only for visualization purposes
    'azimuth': 70, # Only used for visualization
    #'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
    #'orientation': [0.        , 0.        , 0.67226634, 0.74030937],  #paula's cam2world quaternion
    'orientation':[-0.0090482 , -0.00665045,  0.65560794,  0.75501796], # full body paula

    #'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
    #'translation': [  0., 633., 230.], #paula's cam2world translation
    'translation': [ -27., 1200.,  280.] # full body paula

}

class CustomDataset(MocapDataset):
    def __init__(self, detections_path, remove_static_joints=True):
        super().__init__(fps=None, skeleton=h36m_skeleton)        
        
        # Load serialized dataset
        data = np.load(detections_path, allow_pickle=True)
        resolutions = data['metadata'].item()['video_metadata']
        
        self._cameras = {}
        self._data = {}
        for video_name, res in resolutions.items():
            cam = {}
            cam.update(custom_camera_params)
            cam['orientation'] = np.array(cam['orientation'], dtype='float32')
            cam['translation'] = np.array(cam['translation'], dtype='float32')
            cam['translation'] = cam['translation']/1000 # mm to meters
            
            cam['id'] = video_name
            cam['res_w'] = res['w']
            cam['res_h'] = res['h']
            
            self._cameras[video_name] = [cam]
        
            self._data[video_name] = {
                'custom': {
                    'cameras': cam
                }
            }
                
        if remove_static_joints:
            # Bring the skeleton to 17 joints instead of the original 32
            self.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
            #self.remove_joints([2, 3, 4,  5,  7,  8,  9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])  # compose skeleton without foot JK edit

            # Rewire shoulders to the correct parents
            self._skeleton._parents[11] = 8
            self._skeleton._parents[14] = 8

            #self._skeleton._parents[7] = 4  # compose skeleton without foot JK edit
            #self._skeleton._parents[10] = 4 # compose skeleton without foot JK edit
            
    def supports_semi_supervised(self):
        return False
   