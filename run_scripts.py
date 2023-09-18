import subprocess
from common.arguments import parse_args
args = parse_args()

vids = ['EUDs3Front.mp4','EUDs4Front.mp4' ,'EUDs5Front.mp4']    #args.viz_subject
detector = 'mpcoco'
for vid in vids:
    #subprocess.run(['C:/Users/dimis/PycharmProjects/VideoPose/venv/Scripts/python.exe', 'mediapipe_2d.py','--viz-subject='+ vid])
    #subprocess.run(['C:/Users/dimis/PycharmProjects/VideoPose/venv/Scripts/python.exe', 'mediapipe_3d.py','--viz-subject='+ vid])


    subprocess.run(['C:/Users/dimis/PycharmProjects/VideoPose/venv/Scripts/python.exe','run_openpose.py',
    '-d=custom',
    '-k=' +detector,
    '-arc=3,3,3,3,3',
    '-c=checkpoint',
    '--render',
    '--viz-subject=' + vid,
    '--viz-action=custom',
    '--viz-camera=0',
    '--viz-video=inference/'+vid[:-4]+'/'+vid,
    '--viz-output=output.mp4',
    '--viz-export=' + vid[:-4] +'_pred3d_cam.npy',
    '--viz-size=6'])

    subprocess.run(['C:/Users/dimis/PycharmProjects/VideoPose/venv/Scripts/python.exe', 'metrics_worldspace.py','-k=' +detector,'--viz-subject='+ vid])

#subprocess.run(['C:/Users/dimis/PycharmProjects/VideoPose/venv/Scripts/python.exe', 'metrics_2d.py','-k=+detector','--viz-subject='+ vid])
#subprocess.run(['C:/Users/dimis/PycharmProjects/VideoPose/venv/Scripts/python.exe', 'metrics_worldspace.py','-k=' +detector,'--viz-subject='+ vid])
#subprocess.run(['C:/Users/dimis/PycharmProjects/VideoPose/venv/Scripts/python.exe', 'dash_total_2d.py','--viz-subject='+ vid])
#subprocess.run(['C:/Users/dimis/PycharmProjects/VideoPose/venv/Scripts/python.exe', 'dash_results_2d.py','-k=' +detector','--viz-subject='+ vid])
#subprocess.run(['C:/Users/dimis/PycharmProjects/VideoPose/venv/Scripts/python.exe', 'dash_results.py','-k=+detector','--viz-subject='+ vid])
#subprocess.run(['C:/Users/dimis/PycharmProjects/VideoPose/venv/Scripts/python.exe', 'dash_total.py','--viz-subject='+ vid])
