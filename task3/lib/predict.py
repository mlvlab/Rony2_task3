import os
import torch
import pickle

####################################################################################################
### task-3
####################################################################################################
from task3.lib.yolov5.models.experimental import attempt_load
from task3.lib.yolov5.utils.downloads import attempt_download
from task3.lib.yolov5.utils.datasets import LoadImages, LoadStreams
from task3.lib.yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from task3.lib.yolov5.utils.torch_utils import select_device, time_sync
from task3.lib.yolov5.utils.plots import Annotator, colors
from task3.lib.deep_sort_pytorch.utils.parser import get_config
from task3.lib.deep_sort_pytorch.deep_sort import DeepSort
from task3.lib.pose_estimate import PoseEstimate

# import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from torchreid.utils import FeatureExtractor
import numpy as np
import concurrent.futures

def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)

def detect(opt, video_num):

    assert video_num in {1, 2, 3}
    ft_dict = {}

    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, save_jpg, imgsz, evaluate, half = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.save_jpg, opt.img_size, opt.evaluate, opt.half
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    # if not evaluate:
    #     if os.path.exists(out):
    #         pass
    #         shutil.rmtree(out)  # delete output folder
    #     os.makedirs(out)  # make new output folder
    mkdir_if_missing(out)

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        print("USE GPU")
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_sync()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_sync()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            # print(i, det)
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                ### deepsort update 를 하나의 image 내의 detections 마다 수행한다
                ### BB 크기, conf score, class 번호, 원본이미지 를 받아서 update 를 하면 
                ### -> detection을 만들어내고 -> NMS 하고 -> tracker 를 update
                ### tracker update 를 하면 
                ###### 1. track state distributions 를 한 time step 넘겨줌
                ###### 2. measurement update 하고 track 관리함 
                ###### -> 1) match를 확인
                ###### -> 2) match여부에 따라 track set을 업데이트 (!!!여기서 unmatched detection 인 경우, _initiate_track() 실행)
                ###### -> 3) distance metric을 업데이트

                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)): 

                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]
                        crop_img = im0s[bbox_top:bbox_top+bbox_h, bbox_left:bbox_left+bbox_w].copy()
                        pid = video_num * 1000 + output[4]

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True))

                        if id in ft_dict:
                            ft_dict[pid].append(crop_img)
                            #print(pid)
                        else:
                            ft_dict[pid] = [crop_img]
                            #print(pid)
                        
                        if save_jpg:
                            vn_dir = os.path.join(out, '{:d}'.format(video_num))
                            pid_dir = os.path.join(vn_dir, '{:d}'.format(pid))
                            fid_path = os.path.join(pid_dir, '{:04d}_{:04d}.jpg'.format(pid, frame_idx))
                            mkdir_if_missing(vn_dir)
                            mkdir_if_missing(pid_dir)
                            cv2.imwrite(fid_path, crop_img)

                        # to MOT format?
                        if save_txt:
                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx, id, bbox_left, bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

            # else:
            #     deepsort.increment_ages()

            # Print time (inference + NMS)
            # print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            im0 = annotator.result()
            if show_vid:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
    return ft_dict

####################################################################################################


import yaml


def func_task1(args):
    print("run_task_1")


def func_task2(args):
    print("run_task_2")


class func_task3:
    def __init__(self, args):

        self.set_num = args.set_num # set_num01, 1
        self.test_num = args.test
        self.input_paths = [os.path.join(args.dataset_dir, args.set_num, f'set01_drone0{i+1}.mp4') for i in range(self.test_num)]
        #self.input_paths = [os.path.join(args.dataset_dir, 'set_01_original', f'set01_drone0{i+1}.mp4') for i in range(3)]
        self.temporary_dir = os.path.join(args.temporary_dir, f't3_res/{self.set_num}')
        
        if os.path.exists(self.temporary_dir):
            shutil.rmtree(self.temporary_dir)  # delete output folder
            os.makedirs(self.temporary_dir)  # make new output folder

        with open ('task3/conf/task3.yaml') as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)

        self.yolo_weights = conf['yolo_weights']
        self.deep_sort_weights = conf['deep_sort_weights']
        # self.source = conf['source']
        self.output = self.temporary_dir
        self.img_size = conf['img_size']
        self.conf_thres = conf['conf_thres']
        self.iou_thres = conf['iou_thres']
        self.fourcc = conf['fourcc']
        self.device = args.device # conf['device']
        self.show_vid = conf['show_vid']
        self.save_vid = conf['save_vid']
        self.save_txt = conf['save_txt']
        self.save_jpg = conf['save_jpg']
        self.classes = conf['classes']
        self.agnostic_nms = conf['agnostic_nms']
        self.augment = conf['augment']
        self.evaluate = True # conf['evaluate']
        self.config_deepsort = conf["config_deepsort"]
        self.half = conf["half"]
        self.img_size = check_img_size(self.img_size)
        self.mtp = False

        # etc
        self.extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='task3/lib/reid/weights/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth',
        )
        # re-id parameters
        self.reid_ff_path = os.path.join(args.root_dir, f'task3/lib/reid/references/ff.pickle')
        self.reid_thresh = 9.5
        self.ff_rm_thresh = 5.8

    def tracking(self, video_num):
        with torch.no_grad():
            fts = detect(self, video_num)
        return fts

    def save_crop_img(self, crop_img):
        with open(f'{self.temporary_dir}/{self.set_num}.pickle', 'wb') as f:
                pickle.dump(crop_img, f, pickle.HIGHEST_PROTOCOL)

    def reid(self, inputs):

        # fire-fighter pkl
        with open(self.reid_ff_path, 'rb') as f:
            ff_ft = pickle.load(f)

        # torch reid
        ids = []
        reid_ft = []
        id_means = []
        id_maxs = []
        dist_list = []
        rm_ids = []

        for j in range(self.test_num):
            for i in inputs[j]:
                ids.append(i)
                reid_ft.append(self.extractor(inputs[j][i]))

        for idf in reid_ft:
            id_means.append(torch.mean(idf, 0))
        for j in range(len(reid_ft)):
            id_maxs.append(int(max(torch.dist(i, id_means[j]) for i in reid_ft[j]).detach().cpu().clone().numpy()))
        for i in id_means:
            tmp = [float(torch.dist(i, k).detach().cpu().clone().numpy())  for k in id_means]
            dist_list.append(tmp)

        n = len(ids)
        new_ids = [ [] * n for i in range(n) ]
        for i in range(n):
            if i in rm_ids:
                continue
            ff_similarity = np.min([ float(torch.dist(id_means[i], ff).detach().cpu().clone().numpy())  for ff in ff_ft])
            if ff_similarity < self.ff_rm_thresh:
                rm_ids.append(i) 
                continue
            for k in range(n-1):
                if k in rm_ids:
                    continue
                if dist_list[i][k] <= self.reid_thresh:
                    new_ids[i].append(k)
                    if i != k:
                        rm_ids.append(k) 

        print(" id groups: {}".format(new_ids))
        print(" remove: {}".format(rm_ids))
        tmp = list(set(rm_ids))
        for i in np.sort(tmp)[::-1]:
            del new_ids[i]
        print(" new ids : {}".format(new_ids))
        return new_ids[i]

    def run(self):
        print("run_task_3")
        
        # tracking 3 different videos with yolo-deepsort
        crop_img = []
        new_ids = []
        
        if self.mtp: # ! CANNOT USE
            def mtp_run(i): 
                out_dict = self.tracking(i+1)
                crop_img.append(out_dict)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in range(self.test_num):
                    executor.submit(mtp_run, i)
        else:
            for i in range(self.test_num):
                self.source = self.input_paths[i]
                out_dict = self.tracking(i+1)
                crop_img.append(out_dict)

        with open(f'{self.temporary_dir}/{self.set_num}.pickle', 'wb') as f:
            pickle.dump(crop_img, f, pickle.HIGHEST_PROTOCOL)

        # reid extraction & tracklet association with osnet
        new_ids = self.reid(crop_img)
        
        # pose estimation (moving or not) with alphapose
        self.pred_move, self.pred_stay, self.pred_total = 0, len(new_ids), len(new_ids)
        pose_estimator = PoseEstimate(pickle_path=f'{self.temporary_dir}/{self.set_num}.pickle', device=self.device)
        self.pred_move, self.pred_stay, self.pred_total = pose_estimator.check_movement()
                
        return self.pred_move, self.pred_stay, self.pred_total


def func_task4(args):
    print("run_task_4")



