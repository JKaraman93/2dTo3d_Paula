

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import subprocess as sp
import pandas as pd


def get_resolution(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1, ) as pipe:  # shell=True was added by JK fow windows not
        for line in pipe.stdout:
            w, h = line.decode().strip().split(',')
            return int(w), int(h)


def get_fps(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            a, b = line.decode().strip().split('/')
            return int(a) / int(b)


def read_video(filename, skip=0, limit=-1):
    w, h = get_resolution(filename)

    command = ['ffmpeg',
               '-i', filename,
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-vsync', '0',
               '-vcodec', 'rawvideo', '-']

    i = 0
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        while True:
            data = pipe.stdout.read(w * h * 3)
            if not data:
                break
            i += 1
            if i > limit and limit != -1:
                continue
            if i > skip:
                yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))


def downsample_tensor(X, factor):
    length = X.shape[0] // factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)


def render_animation(keypoints, keypoints_metadata, poses, fps, bitrate, azim, output, viewport,
                     limit=-1, downsample=1, size=6, input_video_path=None, input_video_skip=0, ):
    # radius=rad):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    plt.ioff()
    fig = plt.figure(figsize=(size * (1 + len(poses)), size))
    # ax_in = fig.add_subplot(1, 1 + len(poses), 1)
    ax_in = fig.add_subplot(1, len(poses), 1)

    ax_in.get_xaxis().set_visible(False)  # jk
    ax_in.get_yaxis().set_visible(False)  # jk
    ax_in.set_axis_off()  # jk
    ax_in.set_title('Mediapipe(bl/red) vs GT (green)')

    vid = list(keypoints_metadata['video_metadata'].keys())[0]
    keypoints2D_label = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
                         'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip',
                         'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

    df1 = pd.read_csv('inference/' + vid[:-4] + '/' + vid[:-4] + '_Rasterspace.csv')
    keypoints_gt = df1.to_numpy(dtype='float32')
    keypoints_gt = keypoints_gt.reshape(keypoints_gt.shape[0], keypoints_gt.shape[1] // 2, 2)
    ax_3d = []
    lines_3d = []
    trajectories = []
    # azim=30

    radius = 1.7  # 0.4 #200 # 1.7
    for index, (title, data) in enumerate(poses.items()):
        if index == 0:  # i want to execute it once in onder to create only one 3dplot jk
            # ax = fig.add_subplot(1, 1 + len(poses), index+2, projection='3d')
            ax = fig.add_subplot(1, len(poses), index + 2, projection='3d')

            ax.view_init(elev=15., azim=azim)
            ax.set_xlabel('$X$')
            ax.set_ylabel('$Y$')
            ax.set_zlabel('$Z$')
            ax.set_xlim3d([-radius / 2, radius / 2])
            ax.set_zlim3d([0, 1.2 * radius])
            ax.set_ylim3d([-radius / 2, radius / 2])

            try:
                ax.set_aspect('equal')
            except NotImplementedError:
                ax.set_aspect('auto')
            # ax.set_xticklabels([])
            # ax.set_yticklabels([])
            # ax.set_zticklabels([])
            ax.dist = 7.5
            # ax.set_title(title) #, pad=35
            ax.set_title('GT (gr/yel) vs Reconstruction (r/b)')
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])  # list : 2 arrays / shape: (1621,2) hips x,y coords for each frame
    poses = list(poses.values())
    # fig.delaxes(ax) #jk

    # Decode video
    if input_video_path is None:
        # Black background
        all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
    else:
        # Load video using ffmpeg
        all_frames = []
        for f in read_video(input_video_path, skip=input_video_skip, limit=limit):
            all_frames.append(f)
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]

        keypoints = keypoints[input_video_skip:]  # todo remove
        for idx in range(len(poses)):
            poses[idx] = poses[idx][input_video_skip:]

        if fps is None:
            fps = get_fps(input_video_path)

    if downsample > 1:
        keypoints = downsample_tensor(keypoints, downsample)
        all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
        for idx in range(len(poses)):
            poses[idx] = downsample_tensor(poses[idx], downsample)
            trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
        fps /= downsample

    initialized = False
    image = None
    lines = []
    points = None
    annot_3d = [[], []]
    annot_2d = []
    numframe = None
    points_gt = None

    if limit < 1:
        limit = len(all_frames)
    else:
        limit = min(limit, len(all_frames))

    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]
    parents = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])
    def update_video(i):
        # print (i)
        nonlocal initialized, image, lines, points, annot_3d, annot_2d, numframe, points_gt

        # for n, ax in enumerate(ax_3d):
        # ax.clear() #jk add
        # ax.set_xlim3d([-radius/2 + trajectories[n][i, 0], radius/2 + trajectories[n][i, 0]])
        # ax.set_ylim3d([-radius/2 + trajectories[n][i, 1], radius/2 + trajectories[n][i, 1]])

        # Update 2D poses
        joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]
        colors_2d = np.full(keypoints.shape[1], 'black')
        colors_2d[joints_right_2d] = 'red'

        colors_2d_gt = np.full(keypoints_gt.shape[1], 'white')
        # colors_2d_gt[joints_right_2d] = 'yellow'
        zdir = 'z'  # jk

        if not initialized:
            image = ax_in.imshow(all_frames[i], aspect='equal')
            numframe = ax_in.text(0, 0, s='Frame: ' + str(i), fontstretch='ultra-condensed', fontsize=20,
                                  horizontalalignment='center', verticalalignment='center')

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
                    # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
                    lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                            [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color='pink'))

                col = 'red' if j in joints_right else 'black'
                col2 = 'yellow' if j in joints_right else 'green'  # jk edit  :  different colors for gt skeleton
                color = [col, col2]

                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    # lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                    lines_3d[n].append(ax_3d[0].plot([pos[j, 0], pos[j_parent, 0]],
                                                     # jk edit plot reconstruction and ground truth at the same plot
                                                     [pos[j, 1], pos[j_parent, 1]],
                                                     # [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))
                                                     [pos[j, 2], pos[j_parent, 2]], zdir='z', c=color[n]))  # jk edit
            for n, ax in enumerate(ax_3d):
                xs = pos[:, 0]
                ys = pos[:, 1]
                zs = pos[:, 2]
                for ind, (x, y, z) in enumerate(zip(xs, ys, zs)):
                    label = ''  # dict_keypoints[ind] + '(' + str(np.round(x, 2)) + ',' + str(np.round(y, 2)) + ',' + str(np.round(z, 2)) + ')'
                    annot_3d[n].append(ax_3d[0].text(x, y, z, label, zdir, fontstretch='ultra-condensed', fontsize=5.2,
                                                     horizontalalignment='center', verticalalignment='center'))

            points = ax_in.scatter(*keypoints[i].T, 10, color=colors_2d, edgecolors='white', zorder=10)
            points_gt = ax_in.scatter(*keypoints_gt[i].T, 10, color=colors_2d_gt, edgecolors='green', zorder=10)

            x2d = keypoints[i, :, 0]
            y2d = keypoints[i, :, 1]
            for ind, (x, y) in enumerate(zip(x2d, y2d)):
                label = keypoints2D_label[ind]
                annot_2d.append(ax_in.text(x, y, s=label, fontstretch='ultra-condensed', fontsize=10,
                                           horizontalalignment='center', verticalalignment='center'))

            initialized = True
        else:
            image.set_data(all_frames[i])

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
                    lines[j - 1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                             [keypoints[i, j, 1], keypoints[i, j_parent, 1]])

                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    xs = pos[:, 0]
                    ys = pos[:, 1]
                    zs = pos[:, 2]
                    for joint in range(xs.shape[0]):
                        annot_3d[n][joint].set(
                            text='')  # dict_keypoints[joint] + '(' + str(np.round(xs[joint], 2)) + ',' + str( np.round(ys[joint], 2)) + ',' + str(np.round(zs[joint], 2)) + ')')
                        annot_3d[n][joint].set_position_3d((xs[joint], ys[joint], zs[joint]), 'y')

                    lines_3d[n][j - 1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
                    lines_3d[n][j - 1][0].set_ydata(np.array([pos[j, 1], pos[j_parent, 1]]))
                    lines_3d[n][j - 1][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir='z')

            x2d = keypoints[i, :, 0]
            y2d = keypoints[i, :, 1]
            for joint in range(x2d.shape[0]):
                annot_2d[joint].set_position((x2d[joint], y2d[joint]))

            points.set_offsets(keypoints[i])
            points_gt.set_offsets(keypoints_gt[i])

            numframe.set_text('Frame: ' + str(i))

        print('{}/{}      '.format(i, limit), end='\r')

    fig.tight_layout()

    anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000 / fps,
                         repeat=False, )  # blit=True)
    if output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()