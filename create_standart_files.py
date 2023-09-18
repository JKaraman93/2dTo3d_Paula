import os
import shutil
import pandas as pd
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

view = 'Front'
selected_columns = ['murX','murY',	'murZ',	'rightfemurX',	'rightfemurY',	'rightfemurZ',	'murX',	'murY',
'murZ',	'murX',	'murY',	'murZ',	'leftfemurX',	'leftfemurY',	'leftfemurZ',	'murX',
'murY',	'murZ',	'murX',	'murY',	'murZ',	'upperspineX',	'upperspineY',	'upperspineZ',
'neckX',	'neckY',	'neckZ',	'sphnosetipX',	'sphnosetipY',	'sphnosetipZ',	'sphheadtopX',
'sphheadtopY',	'sphheadtopZ',	'leftshoulderX',	'leftshoulderY',	'leftshoulderZ',
'leftelbowX',	'leftelbowY',	'leftelbowZ',	'leftwristX',	'leftwristY',	'leftwristZ',
'rightshoulderX',	'rightshoulderY',	'rightshoulderZ',	'rightelbowX',	'rightelbowY',
'rightelbowZ',	'rightwristX',	'rightwristY',	'rightwristZ']


def worldTocamToraster_csv(csv_path):
    video_path = csv_path[:-3] + 'mp4'
    vcap = cv2.VideoCapture(video_path)
    if vcap.isOpened():
        nx = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        ny = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        success, image = vcap.read()

    fov = 10  ##vertical fov ?????
    n = 10
    f = 10000
    start = np.array([-27, 1200, 280])  # camera position - frontview
    # start = np.array([743, -69, 230])   # camera position - rightview
    # start = np.array([-743, -69, 230])   # camera position - left view
    to = np.array([-0.45, 0.31, 110])  # look_at point
    up_vec = np.array([0.0038895, -0.0926873, 0.995688])

    t = n * np.tan(np.deg2rad(fov / 2.))  # convert from degrees to rad     #top
    r = (nx / ny) * t  # right
    b = -t  # bottom
    l = -r  # left
    height = t - b
    width = r - l
    gaze = np.subtract(to, start)
    w = np.divide(-gaze, np.linalg.norm(gaze))
    u = np.cross(up_vec, w)
    u = np.divide(u, np.linalg.norm(u))
    v = np.cross(w, u)
    np.append(u, 0)
    np.append(v, 0)
    np.append(w, 0)
    np.append(start, 1)
    Mcam2world = np.column_stack((np.append(u, 0), np.append(v, 0), np.append(w, 0), np.append(start, 1)))
    # print ('Mcam2world: \n',Mcam2world)
    Mworld2cam = np.linalg.inv(Mcam2world)
    # print ('Mworld2cam: \n',Mcam2world)

    Mper = np.array(
        [[2 * n / (r - l), 0, 0, 0], [0, 2 * n / (t - b), 0, 0], [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
         [0, 0, -1, 0]])

    Mvp = np.array([[nx / 2, 0, 0, (nx - 1) / 2], [0, ny / 2, 0, (ny - 1) / 2], [0, 0, 1, 0], [0, 0, 0, 1]])

    r = R.from_matrix(Mcam2world[:3, :3])
    Rot = r.as_quat()
    Rot = np.append(Rot[3], Rot[:3])
    t = Mcam2world[:, -1][:-1]
    print(video_path, Rot, t)

    df_org = pd.read_csv(csv_path)
    df = df_org.iloc[:, 2:]
    df_cam = df.copy(deep=True)
    df_raster = df.copy(deep=True)

    dfr = df_org.iloc[:, 2:]  # raster space
    dfc = df_org.iloc[:, 2:]  # camera space

    for r in df.index:
        for c in range(0, len(df.columns), 3):
            Xcor = dfr.iloc[r, c]
            Ycor = dfr.iloc[r, c + 1]
            Zcor = dfr.iloc[r, c + 2]
            Pworld = np.array([Xcor, Ycor, Zcor, 1])  # homogeneous cords of selected joint
            # print (Pworld)
            Pcam = np.dot(Mworld2cam, Pworld)
            dfc.iloc[r, c] = Pcam[0]
            dfc.iloc[r, c + 1] = Pcam[1]
            dfc.iloc[r, c + 2] = Pcam[2]
            Praster = np.dot(Mvp, np.dot(Mper, Pcam))
            Praster = Praster / Praster[-1]  # from homogeneous to cartesian coordinates
            dfr.iloc[r, c] = Praster[0]
            dfr.iloc[r, c + 1] = ny - Praster[1]
            dfr.iloc[r, c + 2] = Praster[2]
    df_cam[dfc.columns] = dfc  # data transformed in camera space
    df_raster[dfr.columns] = dfr  # data transformed in raster space (pixel)
    df_raster = df_raster.drop([s for s in df_raster.columns if s.endswith('Z')], axis=1)

    df_cam.to_csv(video_path[:-4] + '_Camspace.csv', index=False)
    df_raster.to_csv(video_path[:-4] + '_Rasterspace.csv', index=False)

for file in os.listdir("inference/videos"):
    if file.endswith(".csv"):
        if file.endswith(".csv"):
            try :
                path = os.path.join('inference',file[:-4]+view)
                os.mkdir(path)
                print ('Directory is created.')
                shutil.copy(os.path.join('inference/videos',file[:-4]+view+'.mp4'),os.path.join('inference',file[:-4]+view,file[:-4]+view+'.mp4'))
                shutil.copy(os.path.join('inference/videos',file),os.path.join('inference',file[:-4]+view,file[:-4]+view+'.csv'))

            except:
                print('The directory already exists!')
            finally:
                df = pd.read_csv(os.path.join('inference', file[:-4] + view, file[:-4] + view + '.csv'))
                df.insert(0,'murX',(df['rightfemurX']+df['leftfemurX'])/2)
                df.insert(1,'murY',(df['rightfemurY']+df['leftfemurY'])/2)
                df.insert(2,'murZ',(df['rightfemurZ']+df['leftfemurZ'])/2)
                df2 = df[selected_columns]
                df2.to_csv(os.path.join('inference', file[:-4] + view, file[:-4] + view + '_world.csv'), index=False)

                df_array = df2.to_numpy(dtype='float32')
                df_array = df_array.reshape(df_array.shape[0], df_array.shape[1] // 3, 3)
                np.save(os.path.join('inference', file[:-4] + view, file[:-4] + view + '_world'), df_array)
                worldTocamToraster_csv(os.path.join('inference', file[:-4] + view, file[:-4] + view + '.csv'))
                df_array[:,0:7,2] += 6.1  # corrected hips
                np.save(os.path.join('inference', file[:-4] + view, file[:-4] + view + '_world_correctedhips'), df_array)






