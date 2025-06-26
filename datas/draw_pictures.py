#!/usr/bin/env python
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# encoding: utf-8
'''
@project : MSRGCN
@file    : draw_pictures.py
@author  : Droliven
@contact : droliven@163.com
@ide     : PyCharm
@time    : 2021-07-27 21:22
'''
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def draw_pic_single(mydata, I, J, LR, full_path):
    # 22, 3
    # I
    # J
    # LR

    # # ****************************
    # # 调整坐标，规范数据格式，：这里由于转换过来后本身应满足需求，不需要专门 revert_coordinate 或者交换坐标轴
    mydata = mydata[:, [0, 2, 1]]
    # # ****************************

    x = mydata[:, 0]
    y = mydata[:, 1]
    z = mydata[:, 2]

    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.grid(False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d([-1000, 1000])
    ax.set_ylim3d([-1000, 1000])
    ax.set_zlim3d([-1000, 1000])

    ax.scatter(x, y, z, c='b')

    # (250, 40, 40) #FA2828 红
    # (245, 125, 125) #F57D7D 粉
    # (11, 11, 11) #0B0B0B 黑色
    # (180, 180, 180) #B4B4B4 灰色

    # Make connection matrix
    for i in np.arange(len(I)):
        x, y, z = [np.array([mydata[I[i], j], mydata[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, c='#B4B4B4' if LR[i] else '#B4B4B4')

    plt.savefig(full_path)
    plt.close()

def draw_pic_single_2d(mydata, I, J, LR, full_path):
    x = mydata[:, 0]
    y = mydata[:, 1]

    plt.figure(figsize=(6, 6))

    plt.scatter(x, y, c='r')

    # (250, 40, 40) #FA2828 红
    # (245, 125, 125) #F57D7D 粉
    # (11, 11, 11) #0B0B0B 黑色
    # (180, 180, 180) #B4B4B4 灰色

    # Make connection matrix
    for i in np.arange(len(I)):
        x, y = [np.array([mydata[I[i], j], mydata[J[i], j]]) for j in range(2)]
        # ax.plot(x, y, z, lw=2, color='#FA2828' if LR[i] else '#F57D7D')
        # ax.plot(x, y, z, lw=2, color='#0B0B0B' if LR[i] else '#B4B4B4')
        plt.plot(x, y, lw=2, color='g' if LR[i] else 'b')

    plt.xlim((-800, 800))
    plt.ylim((-1500, 800))
    # 设置坐标轴名称
    plt.xlabel('x')
    plt.ylabel('y')
    # 设置坐标轴刻度
    my_x_ticks = np.arange(-1000, 1000, 200)
    my_y_ticks = np.arange(-1000, 1000, 200)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.grid(False)

    plt.savefig(full_path)
    plt.close(1)

def draw_pic_gt_pred(gt, pred, I, J, LR, full_path):
    # # ****************************
    # # 调整坐标，规范数据格式，：这里由于转换过来后本身应满足需求，不需要专门 revert_coordinate 或者交换坐标轴
    gt = gt[:, [0, 2, 1]]
    pred = pred[:, [0, 2, 1]]

    # # ****************************

    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d([-1000, 1000])
    ax.set_ylim3d([-1000, 1000])
    ax.set_zlim3d([-1000, 1000])

    ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c='k', linewidths=1)
    ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='r', linewidths=1)

    # (250, 40, 40) #FA2828 红
    # (245, 125, 125) #F57D7D 粉
    # (11, 11, 11) #0B0B0B 黑色
    # (180, 180, 180) #B4B4B4 灰色

    # Make connection matrix
    for i in np.arange(len(I)):
        x, y, z = [np.array([gt[I[i], j], gt[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=1, color='#0B0B0B' if LR[i] else '#B4B4B4')
    for i in np.arange(len(I)):
        x, y, z = [np.array([pred[I[i], j], pred[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, color='#FA2828' if LR[i] else '#F57D7D')

    plt.savefig(full_path)
    plt.close()

def draw_pic_gt_pred_2d(gt, pred, I, J, LR, full_path):

    plt.figure(figsize=(6, 6))

    plt.scatter(gt[:, 0], gt[:, 1], c='k', linewidths=1)
    plt.scatter(pred[:, 0], pred[:, 1], c='r', linewidths=1)

    # (250, 40, 40) #FA2828 红
    # (245, 125, 125) #F57D7D 粉
    # (11, 11, 11) #0B0B0B 黑色
    # (180, 180, 180) #B4B4B4 灰色

    # Make connection matrix
    for i in np.arange(len(I)):
        x, y = [np.array([gt[I[i], j], gt[J[i], j]]) for j in range(2)]
        plt.plot(x, y, lw=1, color='#0B0B0B' if LR[i] else '#B4B4B4')
    for i in np.arange(len(I)):
        x, y = [np.array([pred[I[i], j], pred[J[i], j]]) for j in range(2)]
        plt.plot(x, y, lw=2, color='#FA2828' if LR[i] else '#F57D7D')

    plt.xlim((-800, 800))
    plt.ylim((-1500, 800))
    # 设置坐标轴名称
    plt.xlabel('x')
    plt.ylabel('y')
    # 设置坐标轴刻度
    my_x_ticks = np.arange(-1000, 1000, 200)
    my_y_ticks = np.arange(-1000, 1000, 200)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.grid(False)

    plt.savefig(full_path)
    plt.close(1)


def plot_confidence_overlay(predicted_seq, ground_truth_seq, save_path=None):
    """Overlay predicted and ground truth skeletons with a joint-wise error heatmap.

    Args:
        predicted_seq (np.ndarray): Array of shape (T, J, 3) with predicted joint
            coordinates.
        ground_truth_seq (np.ndarray): Array of shape (T, J, 3) with ground
            truth joint coordinates.
        save_path (str, optional): Directory to save generated frames. If not
            provided, the frames are displayed using ``plt.show``.

    Each joint is colored from blue (low error) to red (high error).
    """

    assert predicted_seq.shape == ground_truth_seq.shape
    T, J, _ = predicted_seq.shape

    for t in range(T):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        pred = predicted_seq[t]
        gt = ground_truth_seq[t]
        errors = np.linalg.norm(pred - gt, axis=1)
        max_error = np.max(errors) + 1e-6

        for j in range(J):
            color = plt.cm.jet(errors[j] / max_error)
            ax.scatter(
                pred[j, 0],
                pred[j, 1],
                pred[j, 2],
                c=[color],
                s=60,
                label=f"Joint {j}" if t == 0 else "",
            )

        ax.set_title(f"Frame {t + 1} – Prediction Error Overlay")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=20, azim=-60)

        if save_path:
            plt.savefig(f"{save_path}/frame_{t:03d}.png")
        else:
            plt.show()
        plt.close(fig)


def animate_prediction_errors(
    predicted_seq,
    ground_truth_seq,
    save_video_path="output.mp4",
    fps=10,
):
    """Generate an animation of prediction errors as a video.

    Each joint of the predicted pose is colored based on the per-joint error
    relative to the ground truth pose. Blue indicates low error while red
    indicates high error.

    Args:
        predicted_seq (np.ndarray): Sequence of predicted poses of shape
            ``(T, J, 3)``.
        ground_truth_seq (np.ndarray): Sequence of ground truth poses with the
            same shape as ``predicted_seq``.
        save_video_path (str): Path of the exported ``.mp4`` file.
        fps (int): Frames per second for the output video.
    """

    assert predicted_seq.shape == ground_truth_seq.shape
    T, J, _ = predicted_seq.shape

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scat = ax.scatter([], [], [], c=[], s=60)

    def update(t):
        pred = predicted_seq[t]
        gt = ground_truth_seq[t]
        errors = np.linalg.norm(pred - gt, axis=1)
        max_error = np.max(errors) + 1e-6
        colors = [plt.cm.jet(e / max_error) for e in errors]

        scat._offsets3d = (pred[:, 0], pred[:, 1], pred[:, 2])
        scat.set_color(colors)
        ax.set_title(f"Frame {t + 1}")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        return scat,

    FuncAnimation(fig, update, frames=T, interval=1000 // fps)

    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)

    for t in range(T):
        update(t)
        plt.savefig(f"{temp_dir}/frame_{t:04d}.png")

    frame = cv2.imread(f"{temp_dir}/frame_0000.png")
    height, width, _ = frame.shape
    out = cv2.VideoWriter(
        save_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for t in range(T):
        frame = cv2.imread(f"{temp_dir}/frame_{t:04d}.png")
        out.write(frame)

    out.release()
    plt.close(fig)

    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    os.rmdir(temp_dir)


if __name__ == "__main__":
    import numpy as np

    data = np.random.randn(220, 220)

