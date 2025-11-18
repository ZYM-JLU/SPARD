import os
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
import time



# def saveimgs(skeleton, poses_generator, t_hist, fix_0=True, azim=0.0, output=None, size=6,hist_col=('#3498db', '#e74c3c'), pred_col=('#8a11ed', '#3ce79a'), times = None,linewidth = 3):
#     """
#     TODO
#     Render an animation. The supported output modes are:
#      -- 'interactive': display an interactive figure
#                        (also works on notebooks if associated with %matplotlib inline)
#      -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
#      -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
#      -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
#     """
#     if times is None:
#         times = [0, 14, 29, 44, 59, 74]
#     poses,filename = next(poses_generator)
#     IDX = 0
#     plt.ioff()
#     ncol = len(times)
#     nrow = len(poses)
#     fig = plt.figure(figsize=(size * ncol, size * nrow))
#     ax_3d = []
#     lines_3d = []
#     radius = 1.7
#     for index in range(1,nrow * ncol + 1):
#         ax = fig.add_subplot(nrow, ncol, index, projection='3d')
#         ax.view_init(elev=15., azim=azim)
#         ax.set_xlim3d([-radius / 2, radius / 2])
#         ax.set_zlim3d([-radius / 2, radius / 2])
#         ax.set_ylim3d([-radius / 2, radius / 2])
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_zticklabels([])
#         ax.dist = 5.0
#         ax.set_axis_off()
#         ax.patch.set_alpha(0.0)
#         ax_3d.append(ax)
#         lines_3d.append([])
#         index += 1
#     fig.tight_layout()
#     # fig.suptitle(f"{IDX:03d}")
#     fig.subplots_adjust(wspace=-0.4, hspace=0)
#     hist_lcol, hist_rcol = hist_col
#     pred_lcol, pred_rcol = pred_col
#     parents = skeleton.parents()
#     keys = poses.keys()
#     os.makedirs(os.path.dirname(output), exist_ok=True)
#     for i,k in enumerate(keys):
#         # if k == "context" or k =="gt":
#         #     continue
#         pose = poses[k]
#         for j,t in enumerate(times):
#             current_ax = ax_3d[i * nrow + j]
#             # current_ax = ax_3d[j]
#             pos = pose[t]
#             for j, j_parent in enumerate(parents):
#                 if j_parent == -1:
#                     continue
#                 if t < t_hist:
#                     col = hist_rcol if j in skeleton.joints_right() else hist_lcol
#                 else:
#                     col = pred_rcol if j in skeleton.joints_right() else pred_lcol
#                 current_ax.plot([pos[j, 0], pos[j_parent, 0]],[pos[j, 1], pos[j_parent, 1]],[pos[j, 2], pos[j_parent, 2]], zdir='z', c=col,linewidth=linewidth)
#     plt.savefig(f'{output}/multi_obervation.pdf')
#     # plt.savefig(f'{output}/noise.pdf')

def saveimgs(skeleton, poses_generator, t_hist, fix_0=True, azim=0.0, output=None, size=6,hist_col=('#3498db', '#e74c3c'), pred_col=('#8a11ed', '#3ce79a'), times = None,linewidth = 3):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    if times is None:
        times = [0, 14, 29, 44, 59, 74]
    poses,id = next(poses_generator)
    IDX = 0
    plt.ioff()
    ncol = len(times)
    nrow = len(poses)
    fig = plt.figure(figsize=(size * ncol, size * nrow))
    ax_3d = []
    lines_3d = []
    radius = 1.7
    for index in range(1,nrow * ncol + 1):
        ax = fig.add_subplot(nrow, ncol, index, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([-radius / 2, radius / 2])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 5.0
        ax.set_axis_off()
        ax.patch.set_alpha(0.0)
        ax_3d.append(ax)
        lines_3d.append([])
        index += 1
    fig.tight_layout()
    # fig.suptitle(f"{IDX:03d}")
    fig.subplots_adjust(wspace=-0.4, hspace=0)
    hist_lcol, hist_rcol = hist_col
    pred_lcol, pred_rcol = pred_col
    parents = skeleton.parents()
    keys = poses.keys()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    for i,k in enumerate(keys):
        pose = poses[k]
        for j,t in enumerate(times):
            current_ax = ax_3d[i * ncol + j]
            pos = pose[t]
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                if t < t_hist:
                    col = hist_rcol if j in skeleton.joints_right() else hist_lcol
                else:
                    col = pred_rcol if j in skeleton.joints_right() else pred_lcol
                current_ax.plot([pos[j, 0], pos[j_parent, 0]],[pos[j, 1], pos[j_parent, 1]],[pos[j, 2], pos[j_parent, 2]], zdir='z', c=col,linewidth=linewidth)
    plt.savefig(f'{output}/{id}.pdf')
    # plt.savefig(f'{output}/adjust.jpg')
    # plt.savefig(f'{output}/gt.pdf')
    # plt.savefig(f'{output}/gt.jpg')
    # plt.savefig(f'{output}/gt_padding.pdf')
    # plt.savefig(f'{output}/gt_padding.jpg')
    # plt.savefig(f'{output}/gt_observed.pdf')
    # plt.savefig(f'{output}/gt_observed.jpg')
    # plt.savefig(f'{output}/gt_predict.pdf')
    # plt.savefig(f'{output}/gt_predict.jpg')
    # plt.savefig(f'{output}/predict.pdf')
    # plt.savefig(f'{output}/predict.jpg')
    # plt.savefig(f'{output}/norefine_predict.pdf')
    # plt.savefig(f'{output}/norefine_predict.jpg')
    # plt.savefig(f'{output}/norefine_all_predict.pdf')
    # plt.savefig(f'{output}/norefine_all_predict.jpg')
    # plt.savefig(f'{output}/adjust.pdf')
    # plt.savefig(f'{output}/adjust.jpg')

def savesingleimgs(skeleton, poses_generator, t_hist, fix_0=True, azim=0.0, output=None, size=6,hist_col=('#3498db', '#e74c3c'), pred_col=('#8a11ed', '#3ce79a'), times = None,linewidth = 3):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    if times is None:
        times = [0, 14, 29, 44, 59, 74]
    poses,filename = next(poses_generator)
    IDX = 0
    plt.ioff()
    ncol = len(times)
    nrow = 1
    fig = plt.figure(figsize=(size * ncol, size * nrow))
    ax_3d = []
    lines_3d = []
    radius = 1.7
    for index in range(1,nrow * ncol + 1):
        ax = fig.add_subplot(nrow, ncol, index, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([-radius / 2, radius / 2])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 5.0
        ax.set_axis_off()
        ax.patch.set_alpha(0.0)
        ax_3d.append(ax)
        lines_3d.append([])
        index += 1
    fig.tight_layout()
    # fig.suptitle(f"{IDX:03d}")
    fig.subplots_adjust(wspace=-0.4, hspace=0)
    hist_lcol, hist_rcol = hist_col
    pred_lcol, pred_rcol = pred_col
    parents = skeleton.parents()
    keys = poses.keys()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    for i,k in enumerate(keys):
        if k == "context" or k =="gt":
            continue
        pose = poses[k]
        for j,t in enumerate(times):
            current_ax = ax_3d[j]
            pos = pose[t]
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                if t < t_hist:
                    col = hist_rcol if j in skeleton.joints_right() else hist_lcol
                else:
                    col = pred_rcol if j in skeleton.joints_right() else pred_lcol
                current_ax.plot([pos[j, 0], pos[j_parent, 0]],[pos[j, 1], pos[j_parent, 1]],[pos[j, 2], pos[j_parent, 2]], zdir='z', c=col,linewidth=linewidth)
    plt.savefig(f'{output}/obervation.pdf')
    # plt.savefig(f'{output}/noise.pdf')


def save_multi_imgs(skeleton, poses_generator, t_hist, fix_0=True, azim=0.0, output=None, size=6,hist_col=('#3498db', '#e74c3c'), pred_col=('#8a11ed', '#3ce79a'), times = [29, 44, 59, 74],linewidth = 3,opacity = 0.2,filename = "humanbbd_multi"):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    if times is None:
        times = [0, 14, 29, 44, 59, 74]
    poses,pose_id = next(poses_generator)
    IDX = 0
    plt.ioff()
    ncol = len(times)
    fig = plt.figure(figsize=(size * ncol, size * 1))
    ax_3d = []
    lines_3d = []
    radius = 1.7
    for index in range(1,1 * ncol + 1):
        ax = fig.add_subplot(1, ncol, index, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([-radius / 2, radius / 2])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 5.0
        ax.set_axis_off()
        ax.patch.set_alpha(0.0)
        ax_3d.append(ax)
        lines_3d.append([])
        index += 1
    fig.tight_layout()
    fig.subplots_adjust(wspace=-0.4, hspace=0)
    hist_lcol, hist_rcol = hist_col
    pred_lcol, pred_rcol = pred_col
    parents = skeleton.parents()
    keys = poses.keys()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    for time_index, t in enumerate(times):
        for k in keys:
            if "context" in k:
                continue
            pose = poses[k]
            pos = pose[t]
            current_ax = ax_3d[time_index]
            alpha = 1.0
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                if t < t_hist:
                    col = hist_rcol if j in skeleton.joints_right() else hist_lcol
                    alpha = 1.0
                else:
                    col = pred_rcol if j in skeleton.joints_right() else pred_lcol
                    alpha = opacity
                if "gt" in k:
                    col = hist_rcol if j in skeleton.joints_right() else hist_lcol
                    alpha = 1.0
                current_ax.plot([pos[j, 0], pos[j_parent, 0]],[pos[j, 1], pos[j_parent, 1]],[pos[j, 2], pos[j_parent, 2]], zdir='z', c=col,linewidth=linewidth,alpha = alpha)
    plt.savefig(f'{output}/{filename}_{pose_id}.pdf')

def save_switch_imgs(skeleton, poses_generator, t_hist, switch_start, fix_0=True, azim=0.0, output=None, size=6,hist_col=('#3498db', '#e74c3c'), pred_col=('#8a11ed', '#3ce79a'), trans_col = ('orange', 'black'), times = [29, 44, 59, 74],linewidth = 3,filename = "humanbbd_multi"):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    if times is None:
        times = [0, 14, 29, 44, 59, 74]
    poses,pose_id = next(poses_generator)
    plt.ioff()
    ncol = len(times)
    fig = plt.figure(figsize=(size * ncol, size * 1))
    ax_3d = []
    lines_3d = []
    radius = 1.7
    for index in range(1,1 * ncol + 1):
        ax = fig.add_subplot(1, ncol, index, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([-radius / 2, radius / 2])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 5.0
        ax.set_axis_off()
        ax.patch.set_alpha(0.0)
        ax_3d.append(ax)
        lines_3d.append([])
        index += 1
    fig.tight_layout()
    fig.subplots_adjust(wspace=-0.4, hspace=0)
    hist_lcol, hist_rcol = hist_col
    pred_lcol, pred_rcol = pred_col
    tran_lcol, tran_rcol = trans_col
    parents = skeleton.parents()
    keys = poses.keys()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    for time_index, t in enumerate(times):
        for k in keys:
            if "context" in k:
                continue
            pose = poses[k]
            pos = pose[t]
            current_ax = ax_3d[time_index]
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                if t < t_hist:
                    col = hist_rcol if j in skeleton.joints_right() else hist_lcol
                elif t >= switch_start:
                    col = tran_rcol if j in skeleton.joints_right() else tran_lcol
                else:
                    col = pred_rcol if j in skeleton.joints_right() else pred_lcol
                current_ax.plot([pos[j, 0], pos[j_parent, 0]],[pos[j, 1], pos[j_parent, 1]],[pos[j, 2], pos[j_parent, 2]], zdir='z', c=col,linewidth=linewidth,alpha = 1.0)
    plt.savefig(f'{output}/{filename}_{pose_id}.pdf')

def save_gt_imgs(skeleton, poses_generator,fix_0=True, azim=0.0, output=None, size=6,gt_col=('#3498db', '#e74c3c'), times = [0,7,14],linewidth = 3,filename = "humanbdb_gt",margin = 0.5):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    if times is None:
        times = [0, 14, 29, 44, 59, 74]
    poses,pose_id = next(poses_generator)
    plt.ioff()
    ncol = len(times)
    fig = plt.figure(figsize=(size * (ncol - margin), size * 1))
    ax_3d = []
    lines_3d = []
    radius = 1.7
    for index in range(1,1 * ncol + 1):
        ax = fig.add_subplot(1, ncol, index, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([-radius / 2, radius / 2])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 5.0
        ax.set_axis_off()
        ax.patch.set_alpha(0.0)
        ax_3d.append(ax)
        lines_3d.append([])
        index += 1
    fig.tight_layout()
    fig.subplots_adjust(wspace=-0.6, hspace=0)
    gt_lcol, gt_rcol = gt_col
    parents = skeleton.parents()
    keys = poses.keys()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    for time_index, t in enumerate(times):
        for k in keys:
            if "context" in k:
                continue
            pose = poses[k]
            pos = pose[t]
            current_ax = ax_3d[time_index]
            alpha = 1.0
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                col = gt_rcol if j in skeleton.joints_right() else gt_lcol
                alpha = 1.0
                current_ax.plot([pos[j, 0], pos[j_parent, 0]],[pos[j, 1], pos[j_parent, 1]],[pos[j, 2], pos[j_parent, 2]], zdir='z', c=col,linewidth=linewidth,alpha = alpha)
    plt.savefig(f'{output}/{filename}_{pose_id}.pdf')

def render_animation(skeleton, poses_generator, algos, t_hist, fix_0=True, azim=0.0, output=None, size=6, ncol=5,bitrate=3000):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """

    all_poses,filename = next(poses_generator)
    algo = algos[0] if len(algos) > 0 else next(iter(all_poses.keys()))
    t_total = next(iter(all_poses.values())).shape[0]
    poses = dict(filter(lambda x: x[0] in {'gt', 'context'} or algo == x[0].split('_')[0] or x[0].startswith('gt'),
                        all_poses.items()))
    plt.ioff()
    nrow = int(np.ceil(len(poses) / ncol))
    fig = plt.figure(figsize=(size * ncol, size * nrow))
    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(nrow, ncol, index + 1, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        # ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 5.0
        ax.set_title(title, y=1.2)
        ax.set_axis_off()
        ax.patch.set_alpha(0.0)
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    fig.tight_layout()
    fig.subplots_adjust(wspace=-0.4, hspace=0)
    poses = list(poses.values())

    anim = None
    initialized = False
    animating = True
    find = 0
    hist_lcol, hist_rcol = '#3498db', '#e74c3c'
    pred_lcol, pred_rcol = '#8a11ed', '#3ce79a'
    parents = skeleton.parents()
    def update_video(i):
        nonlocal initialized
        if i < t_hist:
            lcol, rcol = hist_lcol, hist_rcol
        else:
            lcol, rcol = pred_lcol, pred_rcol

        for n, ax in enumerate(ax_3d):
            if fix_0 and n == 0 and i >= t_hist:
                continue
            trajectories[n] = poses[n][:, 0, [0, 1, 2]]
            ax.set_xlim3d([-radius / 2 + trajectories[n][i, 0], radius / 2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius / 2 + trajectories[n][i, 1], radius / 2 + trajectories[n][i, 1]])
            ax.set_zlim3d([-radius / 2 + trajectories[n][i, 2], radius / 2 + trajectories[n][i, 2]])
            # ax.plot([0, 0.1],
            #         [0, 0],
            #         [0, 0], c='r')
            # ax.plot([0, 0],
            #         [0, 0.1],
            #         [0, 0], c='g')
            # ax.plot([0, 0],
            #         [0, 0],
            #         [0, 0.1], c='b')
        if not initialized:

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                col = rcol if j in skeleton.joints_right() else lcol
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))
                    # if n == 0:
                    # if j == 1:
                    #     for tx in ax.texts:
                    #         tx.remove()
                    #     for tx in ax.texts:
                    #         tx.remove()
                    # ax.text(pos[j, 0], pos[j, 1], pos[j, 2], f'{j}', None)
            initialized = True
        else:

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                col = rcol if j in skeleton.joints_right() else lcol
                for n, ax in enumerate(ax_3d):
                    if fix_0 and n == 0 and i >= t_hist:
                        continue
                    pos = poses[n][i]
                    lines_3d[n][j - 1][0].set_xdata([pos[j, 0], pos[j_parent, 0]])
                    lines_3d[n][j - 1][0].set_ydata([pos[j, 1], pos[j_parent, 1]])
                    lines_3d[n][j - 1][0].set_3d_properties([pos[j, 2], pos[j_parent, 2]], zdir='z')
                    lines_3d[n][j - 1][0].set_color(col)
                    # if j == 1:
                    #     for tx in ax.texts:
                    #         tx.remove()
                    #     for tx in ax.texts:
                    #         tx.remove()
                    # ax.text(pos[j, 0], pos[j, 1], pos[j, 2], f'{j}', None)

    def show_animation():
        nonlocal anim
        if anim is not None:
            anim.event_source.stop()
        anim = FuncAnimation(fig, update_video, frames=np.arange(0, poses[0].shape[0]), interval=0, repeat=True)
        plt.draw()

    def reload_poses():
        nonlocal poses
        poses = dict(filter(lambda x: x[0] in {'gt', 'context'} or algo == x[0].split('_')[0] or x[0].startswith('gt'),
                            all_poses.items()))
        for ax, title in zip(ax_3d, poses.keys()):
            ax.set_title(title, y=1.2)
        poses = list(poses.values())

    def save_figs():
        nonlocal algo, find
        old_algo = algo
        for algo in algos:
            reload_poses()
            update_video(t_total - 1)
            fig.savefig('out/%d_%s.png' % (find, algo), dpi=400, transparent=True)
        algo = old_algo
        find += 1

    def on_key(event):
        nonlocal algo, all_poses, animating, anim

        if event.key == 'd':
            all_poses = next(poses_generator)
            reload_poses()
            show_animation()
        elif event.key == 'c':
            save()
        elif event.key == ' ':
            if animating:
                anim.event_source.stop()
            else:
                anim.event_source.start()
            animating = not animating
        elif event.key == 'v':  # save images
            if anim is not None:
                anim.event_source.stop()
                anim = None
            save_figs()
        elif event.key.isdigit():
            algo = algos[int(event.key) - 1]
            reload_poses()
            show_animation()

    def save():
        nonlocal anim

        fps = 30
        anim = FuncAnimation(fig, update_video, frames=np.arange(0, poses[0].shape[0]), interval=1000 / fps,
                             repeat=False)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        if output.endswith('.mp4'):
            Writer = writers['ffmpeg']
            writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
            anim.save(output, writer=writer)
        elif output.endswith('.gif'):
            anim.save(output, dpi=80, writer='imagemagick')
        else:
            raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
        print(f'video saved to {output}!')

    fig.canvas.mpl_connect('key_press_event', on_key)
    show_animation()
    plt.show()

def switch_render_animation(skeleton, poses_generator, algos, t_hist, fix_0=True, azim=0.0, output=None, mode='switch', size=2,ncol=5,bitrate=3000, fix_index=None):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    if mode == 'switch':
        fix_0 = False
    if fix_index is not None:
        fix_list = [
            [1, 2, 3],  #
            [4, 5, 6],
            [7, 8, 9, 10],
            [11, 12, 13],
            [14, 15, 16],
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        ]
        fix_i = fix_list[fix_index]
        fix_col = 'darkblue'
    else:
        fix_i = None
    all_poses,filename = next(poses_generator)
    algo = algos[0] if len(algos) > 0 else next(iter(all_poses.keys()))
    t_total = next(iter(all_poses.values())).shape[0]
    poses = dict(filter(lambda x: x[0] in {'gt', 'context'} or algo == x[0].split('_')[0] or x[0].startswith('gt'),
                        all_poses.items()))
    plt.ioff()
    nrow = int(np.ceil(len(poses) / ncol))
    fig = plt.figure(figsize=(size * ncol, size * nrow))
    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(nrow, ncol, index + 1, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 5.0
        if mode == 'switch':
            if index == 0:
                ax.set_title('target', y=1.0, fontsize=12)
        if mode == 'pred' or 'fix' in mode or mode == 'control' or mode == 'zero_shot':
            if index == 0 or index == 1:
                ax.set_title(title, y=1.0, fontsize=12)
        ax.set_axis_off()
        ax.patch.set_alpha(0.0)
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    fig.tight_layout(h_pad=15, w_pad=15)
    fig.subplots_adjust(wspace=-0.4, hspace=0.4)
    poses = list(poses.values())

    anim = None
    initialized = False
    animating = True
    find = 0
    hist_lcol, hist_mcol, hist_rcol = '#3498db', '#e74c3c', '#e74c3c'
    pred_lcol, pred_mcol, pred_rcol = '#8a11ed', '#3ce79a', '#3ce79a'
    tran_lcol, tran_mcol, tran_rcol = 'orange', 'black', 'black'
    # hist_lcol, hist_mcol, hist_rcol = 'gray', 'black', 'red'
    # pred_lcol, pred_mcol, pred_rcol = 'purple', 'black', 'green'
    # tran_lcol, tran_mcol, tran_rcol = 'orange', 'black', 'blue'

    parents = skeleton.parents()

    def update_video(i):
        nonlocal initialized
        if mode == 'switch':
            if i < t_hist:
                lcol, mcol, rcol = hist_lcol, hist_mcol, hist_rcol
            elif i > 75:
                lcol, mcol, rcol = tran_lcol, tran_mcol, tran_rcol
            else:
                lcol, mcol, rcol = pred_lcol, pred_mcol, pred_rcol
        else:
            if i < t_hist:
                lcol, mcol, rcol = hist_lcol, hist_mcol, hist_rcol
            else:
                lcol, mcol, rcol = pred_lcol, pred_mcol, pred_rcol

        for n, ax in enumerate(ax_3d):
            if fix_0 and n == 0 and i >= t_hist:
                continue
            if fix_0 and n % ncol == 0 and i >= t_hist:
                continue
            trajectories[n] = poses[n][:, 0, [0, 1, 2]]
            ax.set_xlim3d([-radius / 2 + trajectories[n][i, 0], radius / 2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius / 2 + trajectories[n][i, 1], radius / 2 + trajectories[n][i, 1]])
            ax.set_zlim3d([-radius / 2 + trajectories[n][i, 2], radius / 2 + trajectories[n][i, 2]])
        if not initialized:

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if j in skeleton.joints_right():
                    col = rcol
                elif j in skeleton.joints_left():
                    col = lcol
                else:
                    col = mcol

                if fix_i is not None and j in fix_i:
                    col = fix_col

                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    # if j in fix_list[n // ncol] and ((n // ncol) * ncol < n < (n // ncol) * ncol + ncol):
                    #     col = fix_col

                    # if j in fix_list[n // ncol] and ((n // ncol) * ncol < n < ((n // ncol) + 1) * ncol):
                    #     lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                    #                                [pos[j, 1], pos[j_parent, 1]],
                    #                                [pos[j, 2], pos[j_parent, 2]], zdir='z', c=fix_col, linewidth=3.0))
                    # else:
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col, linewidth=3.0))
                    # if n == 0:
                    # if j == 1:
                    #     for tx in ax.texts:
                    #         tx.remove()
                    #     for tx in ax.texts:
                    #         tx.remove()
                    # ax.text(pos[j, 0], pos[j, 1], pos[j, 2], f'{j}', None)
            initialized = True
        else:

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if j in skeleton.joints_right():
                    col = rcol
                elif j in skeleton.joints_left():
                    col = lcol
                else:
                    col = mcol

                if fix_i is not None and j in fix_i:
                    col = fix_col

                for n, ax in enumerate(ax_3d):
                    if fix_0 and n == 0 and i >= t_hist:
                        continue
                    if fix_0 and n % ncol == 0 and i >= t_hist:
                        continue

                    # if j in fix_list[n // ncol] and ((n // ncol) * ncol < n < (n // ncol) * ncol + ncol):
                    #     col = fix_col

                    pos = poses[n][i]
                    x_array = np.array([pos[j, 0], pos[j_parent, 0]])
                    y_array = np.array([pos[j, 1], pos[j_parent, 1]])
                    z_array = np.array([pos[j, 2], pos[j_parent, 2]])
                    lines_3d[n][j - 1][0].set_data_3d(x_array, y_array, z_array)
                    # if j in fix_list[n // ncol] and ((n // ncol) * ncol < n < ((n // ncol) + 1) * ncol):
                    #     lines_3d[n][j - 1][0].set_color(fix_col)
                    # else:
                    lines_3d[n][j - 1][0].set_color(col)

                    # if j == 1:
                    #     for tx in ax.texts:
                    #         tx.remove()
                    #     for tx in ax.texts:
                    #         tx.remove()
                    # ax.text(pos[j, 0], pos[j, 1], pos[j,https://zjwsite.github.io/ 2], f'{j}', None)

    def show_animation():
        nonlocal anim
        if anim is not None:
            anim.event_source.stop()
        anim = FuncAnimation(fig, update_video, frames=np.arange(0, poses[0].shape[0]), interval=0, repeat=True)
        plt.draw()

    def reload_poses():
        nonlocal poses
        all_poses, filename = next(poses_generator)
        poses = dict(filter(lambda x: x[0] in {'gt', 'context'} or algo == x[0].split('_')[0] or x[0].startswith('gt'),
                            all_poses.items()))
        for ax, title in zip(ax_3d, poses.keys()):
            ax.set_title('target', y=1.0, fontsize=12)

        poses = list(poses.values())

    # def save_figs():
    #     nonlocal algo, find
    #     old_algo = algo
    #     for algo in algos:
    #         reload_poses()
    #         update_video(t_total - 1)
    #         fig.savefig('out/%d_%s.png' % (find, algo), dpi=400, transparent=True)
    #     algo = old_algo
    #     find += 1

    def save_figs():
        nonlocal algo, find
        old_algo = algo
        os.makedirs('out_svg', exist_ok=True)
        suffix = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')[:-3]
        os.makedirs('out_svg_' + suffix, exist_ok=True)
        for algo in algos:
            reload_poses()
            for i in range(0, t_total + 1, 10):
                if i == 0:
                    update_video(0)
                else:
                    update_video(i - 1)
                fig.savefig('out_svg_' + suffix + '/%d_%s_%d.svg' % (find, algo, i), transparent=True)
        algo = old_algo
        find += 1

    def on_key(event):
        nonlocal algo, all_poses, animating, anim

        if event.key == 'd':
            all_poses = next(poses_generator)
            reload_poses()
            show_animation()
        elif event.key == 'c':
            save()
        elif event.key == ' ':
            if animating:
                anim.event_source.stop()
            else:
                anim.event_source.start()
            animating = not animating
        elif event.key == 'v':  # save images
            if anim is not None:
                anim.event_source.stop()
                anim = None
            save_figs()
        elif event.key.isdigit():
            algo = algos[int(event.key) - 1]
            reload_poses()
            show_animation()

    def save():
        nonlocal anim

        fps = 50
        anim = FuncAnimation(fig, update_video, frames=np.arange(0, poses[0].shape[0]), interval=1000 / fps,
                             repeat=False)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        if output.endswith('.mp4'):
            Writer = writers['ffmpeg']
            writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
            anim.save(output, writer=writer)
        elif output.endswith('.gif'):
            anim.save(output, dpi=80, writer='pillow')
        else:
            raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
        print(f'video saved to {output}!')

    fig.canvas.mpl_connect('key_press_event', on_key)

    # save()
    show_animation()
    plt.show()
    plt.close()

    # save_figs()