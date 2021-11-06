import argparse
import pickle
import time
import os
import shutil
from pathlib import Path

from yaml.tokens import KeyToken

from tqdm import tqdm
import torch
import cv2
import numpy as np

from task3.lib.yolov5.utils.torch_utils import select_device, time_sync

# AlphaPose libraries
from task3.lib.alphapose.scripts.demo_track_api import SingleImageAlphaPose
from task3.lib.alphapose.alphapose.utils.config import update_config

class PoseEstimate:
    def __init__(self, pickle_path=None, device=None, max_frames=50, threshold=0.4):
        self.max_frames = max_frames
        self.threshold = threshold
        self.pickle_path = pickle_path
        self.device = device

    def cosine_distance(self, x, y, normalized=False):
        if not normalized:
            x = np.asarray(x) / np.linalg.norm(x)
            y = np.asarray(y) / np.linalg.norm(y)
        return 1. - np.dot(x, y.T)

    def check_movement(self):
        config_alphapose = "task3/lib/alphapose/configs/alphapose.yaml"

        # Initialize
        
        device = select_device(self.device)
        '''
        out = opt.output
        if os.path.exists(out):
            shutil.rmtree(out)
        os.makedirs(out)
        ### max_frames = opt.max_frames
        output_txt_path = out + '/output.txt'
        '''
        t0 = time.time()
        moving = 0
        stationary = 0

        # AlphaPose initialization
        args_p = update_config(config_alphapose)
        cfg_p = update_config(args_p.ALPHAPOSE.cfg)
        args_p.ALPHAPOSE.tracking = args_p.ALPHAPOSE.pose_track or args_p.ALPHAPOSE.pose_flow

        demo = SingleImageAlphaPose(args_p.ALPHAPOSE, cfg_p, device)

        # Run inference
        t0 = time.time()

        ### pickle_path = opt.source
        with open(self.pickle_path, 'rb') as pkl:
            videos = pickle.load(pkl)
        
        pose_list = {} # {(person_id): (img_num, pose)}
        person_list = []
        movement_id = {}
        pose_delta = {}
        max_pose_delta = []
        # txt = open(output_txt_path, 'w')

        for video in videos:
            for person_id, images in video.items(): 
                if person_id not in person_list:
                    person_list.append(person_id)
                t1 = time_sync()
                cnt = min(len(images), self.max_frames)
                '''
                id_path = str(Path(out) / Path(str(person_id)))
                if os.path.exists(id_path):
                    shutil.rmtree(id_path)
                os.makedirs(id_path)
                '''
                i = 0
                for img in tqdm(images):
                    if i > cnt:
                        break
                    img_brightness = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
                    avg_brightness = np.average(img_brightness)
                    if avg_brightness < 30:
                        continue
                    # save_path = str(id_path / Path(str(i))) + '.jpg'
                    track_ = [np.int64(0), np.int64(0), np.int64(img.shape[1]), np.int64(img.shape[0]), np.int64(person_id)]
                    tracklet = [np.array(track_)]
                    pose = demo.process('id: '+str(person_id), img, tracklet)

                    if pose['result']:
                        keypoints = pose['result'][0]['keypoints'].numpy()
                        scores = pose['result'][0]['kp_score'].numpy()
                        if np.array_equal(keypoints[5], keypoints[6]) or np.array_equal(keypoints[11], keypoints[12]): 
                            continue
                        score_low = False
                        for score in scores:
                            if score[0] <= 0.4:
                                score_low = True
                                break
                        if score_low == True:                            
                            continue
                        '''
                        txt.write(str(i) + '\n')
                        print(pose, file=txt)
                        print(img_brightness, file=txt)
                        print(avg_brightness, file=txt)
                        '''
                        pose_img = demo.vis(img, pose)
                        # cv2.imwrite(save_path, pose_img)

                        if person_id not in pose_list:
                            pose_list[person_id] = list()
                        pose_list[person_id].append((i, keypoints))
                        i = i + 1
                t2 = time_sync()
                print('ID %s Pose Estimation Done. (%.3fs)' % (str(person_id), t2 - t1))
        # txt.close()
        # varlist = []
        framelist_temp = {}
        for person_id, framelist in pose_list.items():
            for i in range(len(framelist)-1):
                (frame_num, pose_val) = framelist[i]
                (next_frame_num, next_pose_val) = framelist[i+1]
                pose_val_np_1 = np.transpose(pose_val)[0]
                pose_val_np_2 = np.transpose(pose_val)[1]
                next_pose_np_1 = np.transpose(next_pose_val)[0]
                next_pose_np_2 = np.transpose(next_pose_val)[1]
                delta_1 = self.cosine_distance(pose_val_np_1, next_pose_np_1)
                delta_2 = self.cosine_distance(pose_val_np_2, next_pose_np_2)
                delta = delta_1 + delta_2
                framelist_temp[frame_num] = delta
                # print(person_id, i, delta, delta_1, delta_2)

                if person_id not in pose_delta:
                    pose_delta[person_id] = list()
                pose_delta[person_id].append(delta)
            
            if person_id in pose_delta:
                max_delta = max(pose_delta[person_id])
                avg = sum(pose_delta[person_id]) / len(pose_delta[person_id])
                var = sum((x-avg)**2 for x in pose_delta[person_id]) / len(pose_delta[person_id])
                if var > 1e-05:
                    movement_id[person_id] = 0
                    continue
                # varlist.append((var,person_id))
                for f, d in framelist_temp.items():
                    if d == max_delta:
                        print(str(f)+' frame is max for '+str(d)+': '+str(person_id))
                # max_delta = sum(pose_delta[person_id])
                max_pose_delta.append((person_id, max_delta))
            
            max_pose_delta.sort(key=lambda x: x[1], reverse=True)
            # varlist.sort(key=lambda x: x[0], reverse=True)

        for (person_id, max_delta) in max_pose_delta:
            # print(person_id, max_delta)
            if max_delta >= self.threshold:
                # moving
                movement_id[person_id] = 1
                moving = moving + 1
            else:
                # not moving
                movement_id[person_id] = 0
                stationary = stationary + 1
        
        return moving, stationary, (moving+stationary)
        '''
        for i, n in movement_id.items():
            if n == 1:
                print('moving '+str(i))
            else:
                print('stationary '+str(i))
        '''

        #for (var, person_id) in varlist:
        #    print('var of ' + str(person_id)+': '+str(var))
        print('Movement Prediction Done. (%.3fs)' % (time.time() - t0))

'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--output', type=str, default='/home/ubuntu/juyeon/track_and_count/pose_estimation', help='output folder')  # output folder
    parser.add_argument('--source', type=str, default='/home/ubuntu/task3/references/output_video/fairmot/1/1.pickle', help='source pickle folder') # input folder
    parser.add_argument('--max_frames', type=int, default=20, help='maximum number of frames to use when detecting movement')
    parser.add_argument('--thres', type=float, default=0.009, help='threshold for predicting movement')
    opt = parser.parse_args()
    print(opt)

    check_movement()
'''

#  Nose         [0]
#  Left Eye     [1], Right Eye     [2],
#  Left Ear     [3], Right Ear     [4],
#  Left Shoulder[5], Right Shoulder[6],
#  Left Elbow   [7], Right Elbow   [8],
#  Left Wrist   [9], Right Wrist  [10], 
#  Left Hip    [11], Right Hip    [12],
#  Left Knee   [13], Right knee   [14],
#  Left Ankle  [15], Right Ankle  [16],
#  Neck        [17]