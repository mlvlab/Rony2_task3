# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Haoyi Zhu,Hao-Shu Fang
# -----------------------------------------------------

"""Script for single-image demo."""
import argparse
import torch
import os
import platform
import sys
import math
import time

import cv2
import numpy as np

from ..alphapose.utils.transforms import get_func_heatmap_to_coord
from ..alphapose.utils.pPose_nms import pose_nms
from ..alphapose.utils.presets import SimpleTransform
from ..alphapose.utils.transforms import flip, flip_heatmap
from ..alphapose.models import builder
from ..alphapose.utils.config import update_config
from ..detector.apis import get_detector
from ..alphapose.utils.vis import getTime

class DetectionLoader():
    def __init__(self, cfg, opt, device):
        self.cfg = cfg
        self.opt = opt
        self.device = device

        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE

        self._sigma = cfg.DATA_PRESET.SIGMA

        pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
        if cfg.DATA_PRESET.TYPE == 'simple':
            self.transformation = SimpleTransform(
                pose_dataset, scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0, sigma=self._sigma,
                train=False, add_dpg=False, gpu_device=self.device)

        self.pose = (None, None, None, None, None)

    def process(self, trackers, im0):
        '''
        Function that prepares YOLOv5 outputs in format suitable for AlphaPose
        '''
        ids = torch.zeros(len(trackers), 1)                                  # ID numbers
        scores = torch.ones(len(trackers), 1)                                # confidence scores
        boxes = torch.zeros(len(trackers), 4)                                # bounding boxes 
        inps = torch.zeros(len(trackers), 3, *self._input_size)              # 
        cropped_boxes = torch.zeros(len(trackers), 4)                        # cropped_boxes
    
        for i, d in enumerate(trackers):
    
            # Alpha pose: prepare data in required format and feed to pose estimator
            inps[i], cropped_box = self.transformation.test_transform(im0, d[:-1])
            cropped_boxes[i] = torch.FloatTensor(cropped_box)
    
            ids[i,0] = int(d[-1])
            boxes[i,:] = torch.from_numpy(d[:-1])
        self.pose = (inps, boxes, scores, ids, cropped_boxes)
        return self.pose
    

class DataWriter():
    def __init__(self, cfg, opt):
        self.cfg = cfg
        self.opt = opt

        self.eval_joints = list(range(cfg.DATA_PRESET.NUM_JOINTS))
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        self.item = (None, None, None, None, None, None, None)

    def start(self):
        # start to read pose estimation results
        return self.update()

    def update(self):
        norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE

        # get item
        (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name) = self.item
        if orig_img is None:
            return None
        # image channel RGB->BGR
        orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]
        self.orig_img = orig_img
        if boxes is None or len(boxes) == 0:
            return None
        else:
            # location prediction (n, kp, 2) | score prediction (n, kp, 1)
            assert hm_data.dim() == 4
            if hm_data.size()[1] == 136:
                self.eval_joints = [*range(0,136)]
            elif hm_data.size()[1] == 26:
                self.eval_joints = [*range(0,26)]
            pose_coords = []
            pose_scores = []

            for i in range(hm_data.shape[0]):
                bbox = cropped_boxes[i].tolist()
                pose_coord, pose_score = self.heatmap_to_coord(hm_data[i][self.eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type)
                pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
            preds_img = torch.cat(pose_coords)
            preds_scores = torch.cat(pose_scores)

            boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                pose_nms(boxes, scores, ids, preds_img, preds_scores, self.opt.min_box_area)

            _result = []
            for k in range(len(scores)):
                _result.append(
                    {
                        'keypoints':preds_img[k],
                        'kp_score':preds_scores[k],
                        'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                        'idx':ids[k],
                        'bbox':[boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0],boxes[k][3]-boxes[k][1]] 
                    }
                )

            result = {
                'imgname': im_name,
                'result': _result
            }

            if hm_data.size()[1] == 49:
                from ..alphapose.utils.vis import vis_frame_dense as vis_frame
            elif self.opt.vis_fast:
                from ..alphapose.utils.vis import vis_frame_fast as vis_frame
            else:
                from ..alphapose.utils.vis import vis_frame
            self.vis_frame = vis_frame

        return result

    def save(self, boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name):
        self.item = (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name)

class SingleImageAlphaPose():
    def __init__(self, args, cfg, device):
        self.args = args
        self.cfg = cfg
        self.device = device

        # Load pose model
        self.pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

        print(f'Loading pose model from {args.checkpoint}...')
        self.pose_model.load_state_dict(torch.load(args.checkpoint, map_location=self.device))
        self.pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)

        self.pose_model.to(self.device)
        self.pose_model.eval()

        self.det_loader = DetectionLoader(self.cfg, self.args, self.device)
        
    def process(self, im_name, image, trackers):
        # Init data writer
        self.writer = DataWriter(self.cfg, self.args)
        pose = None
        try:
            start_time = getTime()
            with torch.no_grad():
                (inps, boxes, scores, ids, cropped_boxes) = self.det_loader.process(trackers, image)
                # Pose Estimation
                inps = inps.to(self.device)
                hm = self.pose_model(inps)
                hm = hm.cpu()
                self.writer.save(boxes, scores, ids, hm, cropped_boxes, image, im_name)
                pose = self.writer.start()
            #print(pose)
            
            # print('===========================> Finish Model Running.')
        except Exception as e:
            print(repr(e))
            print('An error as above occurs when processing the images, please check it')
            pass
        except KeyboardInterrupt:
            print('===========================> Finish Model Running.')

        return pose

    def getImg(self):
        return self.writer.orig_img

    def vis(self, image, pose):
        #if pose is not None:
        image = self.writer.vis_frame(image, pose, self.writer.opt)
        return image

    def writeJson(self, final_result, outputpath, form='coco', for_eval=False):
        from alphapose.utils.pPose_nms import write_json_counter
        write_json_counter(final_result, outputpath, form=form, for_eval=for_eval)
        # print("Results have been written to json.")
