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
    def __init__(self, crop_imgs=None, new_id_list=None, device=None, max_frames=50, threshold=0.4, isReleaseMode=False):
        self.max_frames = max_frames
        self.threshold = threshold
        self.crop_imgs = crop_imgs
        self.new_id_list = new_id_list
        self.device = device
        self.release_mode = isReleaseMode

    def cosine_distance(self, x, y, normalized=False):
        if not normalized:
            x = np.asarray(x) / np.linalg.norm(x)
            y = np.asarray(y) / np.linalg.norm(y)
        return 1. - np.dot(x, y.T)

    def check_movement(self):
        config_alphapose = "task3/lib/alphapose/configs/alphapose.yaml"

        # Assertion
        assert(os.path.exists(config_alphapose))
        
        # Initialize
        device = select_device(self.device)
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
        videos = self.crop_imgs
        
        pose_list = {} # {(person_id): (img_num, pose)}
        person_list = []
        movement_id = {}
        pose_delta = {}
        max_pose_delta = []

        for video in videos:
            for person_id, images in video.items(): 
                if person_id not in person_list:
                    person_list.append(person_id)
                #t1 = time_sync()
                
                cnt = min(len(images), self.max_frames)
                
                i = 0             
                for img in tqdm(images):
                    if i > cnt:
                        break
                    img_brightness = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
                    avg_brightness = np.average(img_brightness)
                    if avg_brightness < 30:
                        continue

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
                        
                        pose_img = demo.vis(img, pose)

                        if person_id not in pose_list:
                            pose_list[person_id] = list()
                        pose_list[person_id].append((i, keypoints))
                        i = i + 1
                
                if person_id not in pose_list:
                    movement_id[person_id] = 0

                #t2 = time_sync()
                #print('ID %s Pose Estimation Done. (%.3fs)' % (str(person_id), t2 - t1))

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
                if self.release_mode == False:
                    for f, d in framelist_temp.items():
                        if d == max_delta:
                            print(str(f)+' frame is max for '+str(d)+': '+str(person_id))
                max_pose_delta.append((person_id, max_delta))
            else:
                movement_id[person_id] = 0 # delta 값을 계산하지 못하면 (판단하기에 충분한 프레임이 없으면) -> 움직이지 않는 것으로 판단

            max_pose_delta.sort(key=lambda x: x[1], reverse=True)

        for (person_id, max_delta) in max_pose_delta:
            if max_delta >= self.threshold:
                # moving
                movement_id[person_id] = 1
            else:
                # not moving
                movement_id[person_id] = 0
            
        
        new_people_num = len(self.new_id_list)
        new_movement_id = [0] * new_people_num
        for i in range(new_people_num):
            new_people = self.new_id_list[i]
            for ori_person in new_people:
                if movement_id[ori_person] == 1:
                    new_movement_id[i] = 1
                    break # 동영상 내 하나라도 움직이면 move 로 판단.
                else:
                    new_movement_id[i] = 0
        
        idx = 1
        for flag in new_movement_id:
            if flag == 1:
                moving = moving + 1
            else:
                stationary = stationary + 1
            idx = idx+1
        if self.release_mode == False:
            print('All Pose Estimation Done. (%.3fs)' % (time.time() - t0))

        return moving, stationary, (moving+stationary)