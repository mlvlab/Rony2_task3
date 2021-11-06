import os
import sys
import numpy as np
import julius

import torch
import torchaudio

sys.path.append('lib')

from task2.lib.marg_network import Estimator
from task2.lib.marg_postprocess import PostProcessor
from task2.lib.marg_utils import *

def task2_inference(data_path):

    ckpt = torch.load('task2/weights/marg_ckpt.pt')
    estimator = Estimator().cuda()
    estimator.load_state_dict(ckpt['net'])

    estimator.eval()
    estimator.reset_state()

    set_num = 5
    drone_num = 3
    answer_list = []
    for i in range(set_num):
        sub_answer_list = []
        for j in range(drone_num):
            pred_cla_full = []
            folder_name = 'set_0' + str(i+1)
            file_name = 'set0' + str(i+1) + '_drone0' + str(j+1) + '_ch1.wav'
            y, sr = torchaudio.load(os.path.join(data_path, folder_name, file_name))
            wav_full = julius.resample_frac(y, sr, 16000)[0]

            with torch.no_grad():
                for k in range(wav_full.shape[0] // (16000 * 10)):
                    wav = wav_full[i * 16000 * 10: (i + 1) * 16000 * 10]

                    wav = torch.tensor(wav).float().cuda()
                    wav = wav.view(1, 1, -1).repeat(1, 7, 1)
                    pred_cla = estimator(wav)
                    pred_cla_full.append(pred_cla)

                pred_cla_full = torch.cat(pred_cla_full, -2).squeeze()
                output = torch.sigmoid(pred_cla_full).detach().cpu().numpy()
                output_zeroone = (output > 0.5).astype(int)

            postprocessor = PostProcessor()
            estimation = postprocessor.report(output_zeroone)
            # print(estimation)                
            sub_answer_list.append(estimation)
        answer_list.append(sub_answer_list)
    out_str = answer_list_to_json(answer_list)                    

    return out_str
