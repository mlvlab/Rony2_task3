import os
import shutil
from pathlib import Path
import math

from tqdm import tqdm
import torch
import cv2
import numpy as np

from .models.with_mobilenet import PoseEstimationWithMobileNet
from .modules.keypoints import extract_keypoints, group_keypoints
from .modules.load_state import load_state
from .modules.pose import Pose
from .val import normalize, pad_width


def cosine_distance(x, y, normalized=False):
    if not normalized:
        x = np.asarray(x) / np.linalg.norm(x)
        y = np.asarray(y) / np.linalg.norm(y)
    return 1. - np.dot(x, y.T)

def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, device,
            pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if device != 'cpu' and device != '':
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad

def check_movement(checkpoint_path, videos, new_id_list, move_id_list, device, release_mode, temp_dir):
    # Network Initializaton
    net = PoseEstimationWithMobileNet()    

    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_state(net, checkpoint)

    net = net.eval()
    if device != 'cpu' and device != '':   
        net = net.cuda()
    stride = 8
    height_size = 256 # network input layer height size
    upsample_ratio = 4

    # Initialization for movement prediction
    moving = 0
    stationary = 0
    num_keypoints = Pose.num_kpts
    max_frames = 50
    threshold = 4.0
    out = temp_dir + '/pose'
    #output_txt_path = out + '/output.txt'
    pose_list = {} # {(person_id): (img_num, pose)}
    person_list = []
    movement_id = {}
    pose_delta = {}
    max_pose_delta = []
    #txt = open(output_txt_path, 'w')

    for video in videos:
        for person_id, images in video.items(): 
            if person_id not in person_list:
                person_list.append(person_id)
            #t1 = time_synchronized()
            cnt = min(len(images), max_frames)
            id_path = str(Path(out) / Path(str(person_id)))

            if release_mode == False:
                if os.path.exists(id_path):
                    shutil.rmtree(id_path)
                os.makedirs(id_path)

            i = 0
            for img in tqdm(images):
                if i > cnt:
                    break
                img_brightness = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
                avg_brightness = np.average(img_brightness)
                if avg_brightness < 30:
                    continue
                save_path = str(id_path / Path(str(i))) + '.jpg'

                heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, device)
                total_keypoints_num = 0
                all_keypoints_by_type = []
                for kpt_idx in range(num_keypoints):  # 19th for bg
                    total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

                pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)

                for kpt_id in range(all_keypoints.shape[0]):
                    all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
                    all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
                current_poses = []
                
                for n in range(len(pose_entries)):
                    if len(pose_entries[n]) == 0:
                        continue
                    pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
                    for kpt_id in range(num_keypoints):
                        if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                            pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                            pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                    r_shoulder = pose_keypoints[2]
                    r_wrist = pose_keypoints[4]
                    l_shoulder = pose_keypoints[5]
                    l_wrist = pose_keypoints[7]
                    keypoints_r = [r_shoulder - r_wrist]
                    keypoints_l = [l_shoulder - l_wrist]

                    if (keypoints_r[0][0]==0 and keypoints_r[0][1]==0) or (keypoints_l[0][0]==0 and keypoints_l[0][1]==0):
                        continue

                    # TODO: 양쪽 어깨 등등 겹치면 안되는부분 처리?
                    '''
                    if release_mode == False:
                        pose = Pose(keypoints, pose_entries[n][18])
                        current_poses.append(pose)
                        for pose in current_poses:
                            pose.draw(img)
                        img = cv2.addWeighted(img, 0.6, img, 0.4, 0)
                        for pose in current_poses:
                            img = cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                                        (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
                        cv2.imwrite(save_path, img)
                    '''
                    if person_id not in pose_list:
                        pose_list[person_id] = list()
                    pose_list[person_id].append((i, [keypoints_r, keypoints_l]))
                    i = i + 1

            if person_id not in pose_list:
                movement_id[person_id] = 0

            #t2 = time_synchronized()
            #print('%s Done. (%.3fs)' % (str(person_id), t2 - t1))
    #txt.close()
    varlist = []
    framelist_temp = {}
    for person_id, framelist in pose_list.items():
        for i in range(len(framelist)-1):
            (frame_num, pose_both) = framelist[i]
            (next_frame_num, next_both) = framelist[i+1]
            pose_r = pose_both[0]
            pose_l = pose_both[1]
            next_pose_r = next_both[0]
            next_pose_l = next_both[1]

            cur_right_np_x = np.array(pose_r[0][0])
            cur_right_np_y = np.array(pose_r[0][1])
            cur_left_np_x = np.array(pose_l[0][0])
            cur_left_np_y = np.array(pose_l[0][1])
            next_right_np_x = np.array(next_pose_r[0][0])
            next_right_np_y = np.array(next_pose_r[0][1])
            next_left_np_x = np.array(next_pose_l[0][0])
            next_left_np_y = np.array(next_pose_l[0][1])
    
            cur_cosine_x = cosine_distance(cur_right_np_x, cur_left_np_x)
            cur_cosine_y = cosine_distance(cur_right_np_y, cur_left_np_y)
            next_cosine_x = cosine_distance(next_right_np_x, next_left_np_x)
            next_cosine_y = cosine_distance(next_right_np_y, next_left_np_y)

            delta_x = abs(cur_cosine_x - next_cosine_x)
            delta_y = abs(cur_cosine_y - next_cosine_y)
            delta = delta_x + delta_y
            if math.isnan(delta):
                continue
            framelist_temp[frame_num] = delta

            if person_id not in pose_delta:
                pose_delta[person_id] = list()
            pose_delta[person_id].append(delta)
        
        if person_id in pose_delta:
            max_delta = max(pose_delta[person_id])
            avg = sum(pose_delta[person_id]) / len(pose_delta[person_id])
            var = sum((x-avg)**2 for x in pose_delta[person_id]) / len(pose_delta[person_id])
            varlist.append((var,person_id))
            
            
            if var > 2.5:
                movement_id[person_id] = 0
                continue
            
            for f, d in framelist_temp.items():
                if d == max_delta and release_mode == False:
                    print(str(f)+' frame is max for '+str(d)+': '+str(person_id))
            # max_delta = sum(pose_delta[person_id])
            max_pose_delta.append((person_id, max_delta))
        
        if person_id not in pose_delta:
            movement_id[person_id] = 0

        max_pose_delta.sort(key=lambda x: x[1], reverse=True)
        varlist.sort(key=lambda x: x[0], reverse=True)

    for (person_id, max_delta) in max_pose_delta:
        if release_mode == False:
            print(person_id, max_delta)
        if max_delta >= threshold:
            # moving
            movement_id[person_id] = 1
        else:
            # not moving
            movement_id[person_id] = 0
    
    if release_mode == False:
        for i, n in movement_id.items():
            if n == 1:
                print('moving '+str(i))
            else:
                print('stationary '+str(i))

        for (var, person_id) in varlist:
            print('var of ' + str(person_id)+': '+str(var))

    new_people_num = len(new_id_list)
    new_movement_id = [0] * new_people_num
    for i in range(new_people_num):
        new_people = new_id_list[i]
        for ori_person in new_people:
            if ori_person in movement_id: 
                if movement_id[ori_person] == 1:
                    new_movement_id[i] = 1
                    break # 동영상 내 하나라도 움직이면 move 로 판단.
                elif ori_person in move_id_list:
                    new_movement_id[i] = 1
                    break # 동영상 간 존재여부 바뀌는데 동영상 index 관계에 따라 move 로 판단.
                else:
                    new_movement_id[i] = 0
 
    idx = 1
    for flag in new_movement_id:
        if flag == 1:
            moving = moving + 1
        else:
            stationary = stationary + 1
        idx = idx+1

    if release_mode == False:
        print('All Pose Estimation Done.')
    
    return moving, stationary, (moving+stationary)
