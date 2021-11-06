import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torchaudio
import numpy as np
from einops import rearrange
from scipy import ndimage
from scipy import signal


# b t s
class PostProcessor():
    def __init__(self, 
                 num_frame_per_sec = 4,
                 median_kernel_len = 11):

        self.num_frame_per_sec = num_frame_per_sec
        self.median_kernel_len = median_kernel_len

    def report(self, ans_chunk): # t, s
        t, s = ans_chunk.shape
        ans_chunk = ndimage.median_filter(ans_chunk, size = (self.median_kernel_len, 1))
        ans_chunk = np.pad(ans_chunk, ((1, 1), (0, 0)))
        diff = ans_chunk[1:, :] - ans_chunk[:-1, :]
        start_t, start_cla = np.where(diff == 1)
        end_t, end_cla = np.where(diff == -1)
        estimation = []
        for i in range(3):
            start_t_cla, end_t_cla = np.sort(start_t[start_cla == i]), np.sort(end_t[end_cla == i])
            if len(start_t_cla) != 0:
                mids = np.round((start_t_cla + end_t_cla - 1) / 2 / self.num_frame_per_sec).astype(int)
                mids = list(mids)
            else:
                mids = [None]
            estimation.append(mids)
        return estimation

if __name__ == '__main__':
    out = np.array([[0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).transpose(-1, -2)
    postprocessor = PostProcessor()
    estimation = postprocessor.report(out)
    print(estimation)


