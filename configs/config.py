from utils.logging import setup_logging
setup_logging()
import os
import getpass
import torch
import numpy as np
import argparse

class Config():
    def __init__(self, exp_name="h36m", input_n=10, output_n=10, dct_n=15,
                 device="cuda:0", num_works=0, test_manner="all",
                 train_batch_size=16, lr=2e-4, n_epoch=5000, data_dir=None):
        self.platform = getpass.getuser()
        assert exp_name in ["h36m", "cmu", "3dpw"]
        self.exp_name = exp_name

        self.p_dropout = 0.1
        self.train_batch_size = train_batch_size
        self.test_batch_size = 128
        self.lr = lr
        self.lr_decay = 0.98
        self.n_epoch = n_epoch
        self.leaky_c = 0.2

        self.test_manner = test_manner
        self.input_n = input_n
        self.output_n = output_n
        self.seq_len = input_n + output_n
        self.dct_n = dct_n
        if self.output_n == 25:
            self.frame_ids = [1, 3, 7, 9, 13, 24]
        elif self.output_n == 10:
            self.frame_ids = [1, 3, 7, 9]

        if exp_name == "h36m":

            self.origin_noden = 32
            self.final_out_noden = 22

            self.dim_used_3d = [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30]
            self.dim_repeat_22 = [9, 9, 14, 16, 19, 21]
            self.dim_repeat_32 = [16, 24, 20, 23, 28, 31]

            self.Index2212 = [[0], [1, 2, 3], [4], [5, 6, 7], [8, 9], [10, 11], [12], [13], [14, 15, 16], [17], [18], [19, 20, 21]]
            self.Index127 = [[0, 1], [2, 3], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11]]
            self.Index74 = [[0, 2], [1, 2], [3, 4], [5, 6]]

            self.I32_plot = np.array(
                [0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12, 16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27,
                 28,
                 27, 30])
            self.J32_plot = np.array(
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                 29,
                 30, 31])
            self.LR32_plot = np.array(
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

            self.I22_plot = np.array([8, 0, 1, 2, 8, 4, 5, 6, 8, 9, 10, 9, 12, 13, 14, 14, 9, 17, 18, 19, 19])
            self.J22_plot = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
            self.LR22_plot = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

            self.I12_plot = np.array([4, 0, 4, 2, 4, 4, 6, 7, 4, 9, 10])
            self.J12_plot = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11])
            self.LR12_plot = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0])

            self.I7_plot = np.array([2, 2, 2, 3, 2, 5])
            self.J7_plot = np.array([0, 1, 3, 4, 5, 6])
            self.LR7_plot = np.array([0, 1, 1, 1, 0, 0])

            self.I4_plot = np.array([0, 1])
            self.J4_plot = np.array([3, 2])
            self.LR4_plot = np.array([0, 1])
        elif exp_name == "cmu":

            self.origin_noden = 38
            self.final_out_noden = 25

            self.dim_used_3d = [3, 4, 5, 6, 9, 10, 11, 12, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 28, 30, 31, 32, 34, 35, 37]
            self.dim_repeat_22 = [9, 9, 9, 15, 15, 21, 21]
            self.dim_repeat_32 = [16, 20, 29, 24, 27, 33, 36]

            self.Index2212 = [[0], [1, 2, 3], [4], [5, 6, 7], [8, 9], [10, 11, 12], [13], [14, 15], [16, 17, 18], [19], [20, 21], [22, 23, 24]]  # 其实是 Index2512, 为了保持统一没改名
            self.Index127 = [[0, 1], [2, 3], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11]]
            self.Index74 = [[0, 2], [1, 2], [3, 4], [5, 6]]

            self.Index2510 = [[0], [1, 2, 3], [4], [5, 6, 7], [8, 9], [10, 11, 12], [14, 15], [16, 17, 18], [20, 21],
                         [22, 23, 24]]
            self.Index105 = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
            self.Index53 = [[2], [0, 3], [1, 4]]

            self.I32_plot = np.array(
                [0, 1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 0, 13, 14, 15, 16, 17, 18, 16, 20, 21, 22, 23, 24, 25, 23, 27,
                 16, 29, 30, 31, 32, 33, 34, 32, 36])
            self.J32_plot = np.array(
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                 29, 30, 31, 32, 33, 34, 35, 36, 37])
            self.LR32_plot = np.array(
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                 1, 1, 1])

            self.I22_plot = np.array([8, 0, 1, 2, 8, 4, 5, 6, 8, 9, 10, 11, 9, 13, 14, 15, 16, 15, 9, 19, 20, 21, 22, 21])
            self.J22_plot = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
            self.LR22_plot = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

            self.I12_plot = np.array([4, 0, 4, 2, 4, 4, 6, 7, 4, 9, 10])
            self.J12_plot = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11])
            self.LR12_plot = np.array([0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1])

            self.I7_plot = np.array([2, 2, 2, 3, 2, 5])
            self.J7_plot = np.array([0, 1, 3, 4, 5, 6])
            self.LR7_plot = np.array([0, 1, 0, 0, 1, 1])

            self.I4_plot = np.array([0, 1])
            self.J4_plot = np.array([2, 3])
            self.LR4_plot = np.array([0, 1])

        self.device = device
        self.num_works = num_works
        self.ckpt_dir = os.path.join("./ckpt/", exp_name, "short_term" if self.output_n==10 else "long_term")
        if not os.path.exists(os.path.join(self.ckpt_dir, "models")):
            os.makedirs(os.path.join(self.ckpt_dir, "models"))
        if not os.path.exists(os.path.join(self.ckpt_dir, "images")):
            os.makedirs(os.path.join(self.ckpt_dir, "images"))

        env_map = {
            "h36m": "H36M_DATA_DIR",
            "cmu": "CMU_DATA_DIR",
            "3dpw": "THREEDPW_DATA_DIR"
        }
        env_var = env_map.get(self.exp_name)

        supplied_dir = data_dir
        if not supplied_dir:
            maybe_args = globals().get("args")
            if maybe_args is not None and getattr(maybe_args, "data_dir", ""):
                supplied_dir = maybe_args.data_dir
        if not supplied_dir:
            supplied_dir = os.environ.get(env_var, "")

        if not supplied_dir:
            raise ValueError(
                f"Dataset directory not provided. Use --data_dir or set {env_var}"
            )

        self.base_data_dir = os.path.abspath(supplied_dir)


# ---------------------------------------------------------------------------
# command line arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="configuration")
parser.add_argument('--exp_name', type=str, default='cmu', help='h36m / cmu')
parser.add_argument('--input_n', type=int, default=10)
parser.add_argument('--output_n', type=int, default=25)
parser.add_argument('--dct_n', type=int, default=35)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--num_works', type=int, default=0)
parser.add_argument('--test_manner', type=str, default='all')
parser.add_argument('--debug_step', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=5000)
parser.add_argument('--learning_rate', type=float, default=2e-4)
parser.add_argument('--is_train', type=bool, default='', help='train mode')
parser.add_argument('--is_load', type=bool, default='', help='load checkpoint')
parser.add_argument('--model_path', type=str, default='', help='pretrained model')
parser.add_argument('--dct', type=bool, default=True)
parser.add_argument('--data_dir', type=str, default='', help='path to dataset directory')

args = parser.parse_args()



