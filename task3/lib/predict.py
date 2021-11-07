import os
import numpy as np, random
import torch
import torch.backends.cudnn as cudnn
from torchreid.utils import FeatureExtractor
import cv2
import yaml
import pickle
import concurrent.futures
from tqdm import tqdm
import logging
import platform
import shutil
import time
import datetime
from pathlib import Path

from task3.lib.yolov5.models.experimental import attempt_load
from task3.lib.yolov5.utils.downloads import attempt_download
from task3.lib.yolov5.utils.datasets import LoadImages, LoadStreams
from task3.lib.yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from task3.lib.yolov5.utils.torch_utils import select_device, time_sync
from task3.lib.yolov5.utils.plots import Annotator, colors
from task3.lib.deep_sort.utils.parser import get_config
from task3.lib.deep_sort.deep_sort import DeepSort
from task3.lib.pose_estimate import PoseEstimate
from task3.lib.remove_ff.RFF import RFF
from utils.torch_utils import select_device

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def set_logging(rank=-1):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)

def detect(opt, video_num):
    assert video_num in {1, 2, 3}

    ft_dict = {}

    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, save_jpg, imgsz, evaluate, half = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.save_jpg, opt.img_size, opt.evaluate, opt.half

    # Initialize
    if opt.release_mode == False:
        set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if opt.release_mode == False:
        print("__device__ :", device)

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half: model.half()  # to FP16
    mkdir_if_missing(out)

    # Set Dataloader
    t0 = time.time()
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz)
    print("__LoadImage[s] : ", time.time()-t0)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Find index corresponding to a person
    idx_person = names.index("person")

    # Deep SORT: initialize the tracker
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(tqdm(dataset)):
        if (frame_idx % opt.frame_skip) != 0:
            continue

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        num_of_persons = 0
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Deep SORT: person class only
                idxs_ppl = (det[:,-1] == idx_person).nonzero(as_tuple=False).squeeze(dim=1)   # 1. List of indices with 'person' class detections
                dets_ppl = det[idxs_ppl,:-1]                                                  # 2. Torch.tensor with 'person' detections    
                num_of_persons = len(idxs_ppl)

                # Deep SORT: convert data into a proper format
                xywhs = xyxy2xywh(dets_ppl[:,:-1]).to("cpu")
                confs = dets_ppl[:,4].to("cpu")

                # Deep SORT: feed detections to the tracker 
                if len(dets_ppl) != 0:
                    trackers, _ = deepsort.update(xywhs, confs, im0)

                    for d in trackers:
                        tl, tt, tr, td = d[0], d[1], d[2], d[3]

                        crop_img = im0s[tt:td, tl:tr].copy()
                        pid = video_num * 1000 + d[-1]

                        if pid in ft_dict:
                            ft_dict[pid].append(crop_img)
                        else:
                            ft_dict[pid] = [crop_img]

                        if save_jpg:
                            vn_dir = os.path.join(out, '{:d}'.format(video_num))
                            pid_dir = os.path.join(vn_dir, '{:d}'.format(pid))
                            fid_path = os.path.join(pid_dir, '{:04d}_{:04d}.jpg'.format(pid, frame_idx))
                            mkdir_if_missing(vn_dir)
                            mkdir_if_missing(pid_dir)
                            cv2.imwrite(fid_path, crop_img)

                    for d in trackers:
                        plot_one_box(d[:-1], im0, label=str(int(d[-1])), color=colors[1], line_thickness=3)

            # Save results (image with detections)
            if save_vid:
                if frame_idx:
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
        print('Results saved to %s' % Path(out))

    print('Done. (%.3fs)' % (time.time() - t0))
    return ft_dict


class func_task3:
    def __init__(self, args):

        self.release_mode = args.release_mode
        self.set_num = args.set_num # set_num01, 1
        self.test_num = args.test
        set_index = int(args.set_num.split('_')[1])
        self.input_paths = [os.path.join(args.dataset_dir, args.set_num, f'set0{set_index}_drone0{i+1}.mp4') for i in range(self.test_num)]
        self.temporary_dir = os.path.join(args.temporary_dir, f't3_res/{self.set_num}')
        
        if self.release_mode == False:
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
        # for speed-up 
        self.frame_skip = conf["frame_skip"]
        # re-id parameters
        self.reid_thresh = conf["reid_thresh"]
        self.ff_rm_thresh = conf["ff_rm_thresh"]
        self.reid_weights_path = 'task3/lib/reid/weights/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'
        # filter-out False Positives 
        self.reid_ff_path = os.path.join(args.root_dir, f'task3/lib/reid/references/ff.pickle')
        self.fff_train_np_path = "task3/lib/remove_ff/fff_train_np.pkl" # train set features of firefighters
        # pose estimation parameters
        self.alphapose_conf = 'task3/lib/alphapose/configs/alphapose.yaml'
        self.alphapose_weights = 'task3/lib/alphapose/pretrained_models/fast_res50_256x192.pth'

        # assert
        assert(os.path.exists(self.yolo_weights))
        assert(os.path.exists(self.deep_sort_weights))
        assert(os.path.exists(self.config_deepsort))
        assert(os.path.exists(self.reid_weights_path))
        assert(os.path.exists(self.fff_train_np_path))
        assert(os.path.exists(self.alphapose_conf))
        assert(os.path.exists(self.alphapose_weights))
        # ???
        assert(os.path.exists('task3/lib/alphapose/detector/yolo/data/yolov3-spp.weights'))
        assert(os.path.exists('task3/lib/alphapose/detector/tracker/data/JDE-1088x608-uncertainty'))

        # torch-reid
        self.extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path=self.reid_weights_path
        )
        # fire-fighter pkl
        with open(self.reid_ff_path, 'rb') as f:
            self.ff_means = pickle.load(f)
        # time
        self.lap_time = {}
        # release
        if self.release_mode == True:
            self.show_vid = False
            self.save_vid = False
            self.save_txt = False
            self.save_jpg = False

    def tracking(self, video_num):
        with torch.no_grad():
            fts = detect(self, video_num)
        return fts

    def save_crop_img(self, crop_img):
        with open(f'{self.temporary_dir}/{self.set_num}.pickle', 'wb') as f:
            pickle.dump(crop_img, f, pickle.HIGHEST_PROTOCOL)

    def reid(self, inputs):

        # torch reid
        ids = []
        reid_ft = []
        id_means = []
        id_maxs = []
        dist_list = []
        rm_ids = []

        for j in range(len(inputs)):
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
            ff_similarity = np.min([ float(torch.dist(id_means[i], ff).detach().cpu().clone().numpy())  for ff in self.ff_means])
            if ff_similarity < self.ff_rm_thresh:
                rm_ids.append(i) 
                continue
            for k in range(n):
                if k in rm_ids:
                    continue
                if dist_list[i][k] <= self.reid_thresh:
                    new_ids[i].append(k)
                    if i != k:
                        rm_ids.append(k) 

        if self.release_mode == False:
            print(" id groups: {}".format(new_ids))
            print(" remove: {}".format(rm_ids))
        tmp = list(set(rm_ids))
        for i in np.sort(tmp)[::-1]:
            del new_ids[i]
        if self.release_mode == False:
            print(" new ids : {}".format(new_ids))

        newIds_for_poseEstim = [ [] * n for i in range(len(new_ids)) ]

        n_move = 0
        for i in range(len(new_ids)):
            if self.release_mode == False:
                print("new id :", i)
            move = 0x000
            fig_uid_n = len(new_ids[i])//2
            for k, uid in enumerate(self.get_uniqueIds_in_newIdxs(i, ids, new_ids)):
                vn = self.get_videoNum_in_dicts(uid, inputs)
                imgs = inputs[vn-1][uid]
                if fig_uid_n == k: 
                    fig_vn = vn 
                    fig_uid = uid
                    fig_n = len(imgs)//2
                if self.release_mode == False: 
                    print(" >", vn, uid, imgs[0].shape) 
                    if vn == 1: move |= 0x100
                    elif vn == 2: move |= 0x010
                    elif vn == 3: move |= 0x001    
                newIds_for_poseEstim[i].append(uid)

            if self.release_mode == False:
                # todo : exists or not in all videos 
                if move in (0x100, 0x110, 0x010):
                    n_move += 1
                    print(" > move") 
        
        # todo 
        move_list = []

        if self.release_mode == False:
            print(" new ids : {}".format(newIds_for_poseEstim))

        return newIds_for_poseEstim, move_list

    # get unique-ids in new-index
    def get_uniqueIds_in_newIdxs(self, i, ids, newIdxs):
        return [ids[k] for k in newIdxs[i]]

    # which video unique_id is in ?
    def get_videoNum_in_dicts(self, uid, dicts):
        for i in range(3):
            if (uid in dicts[i]) == True:
                return i+1

    def run(self):
        print("run_task_3")

        t0 = self.lap_time['start'] = time.time()

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

        if self.release_mode == False:
            with open(f'{self.temporary_dir}/{self.set_num}.pickle', 'wb') as f:
                pickle.dump(crop_img, f, pickle.HIGHEST_PROTOCOL)

        self.lap_time['object-tracking'] = time.time() - t0

        # reid extraction & tracklet association with osnet
        new_ids, _ = self.reid(crop_img)

        self.lap_time['re-identification'] = time.time() - t0
        

        # filter-out False Positives such as FireFighter.
        """ Remove FireFighter"""
        pred_idx = RFF(crop_img)
        crop_img_ids = []
        for vid_dict in crop_img:
            crop_img_ids.extend(vid_dict.keys())
        person_ids = list(np.array(crop_img_ids)[pred_idx])
        new_ids_updated = []
        for new_id in new_ids:
            new_id_updated = []
            for id in new_id:
                if id in person_ids:
                    new_id_updated.append(id)
            if len(new_id_updated)>0:
                new_ids_updated.append(new_id_updated)

        if self.release_mode == False:
            print("new-list updated : ", new_ids_updated)
    
        self.lap_time['remove-firefighter'] = time.time() - t0 # [start - RFF]
        "TODO: 사람으로 판단한 id만 갖고 있는 new_ids_updated를 어떻게 다음으로 넘겨 줄 것인가."
        
        # pose estimation (moving or not) with args-pose
        pose_estimator = PoseEstimate(crop_imgs=crop_img, new_id_list=new_ids, device=self.device, isReleaseMode=self.release_mode)
        self.pred_move, self.pred_stay, self.pred_total = pose_estimator.check_movement()
        
        self.lap_time['pose-estimation'] = time.time() - t0

        for i in self.lap_time.keys():
            if i == 'start': continue
            lt = self.lap_time[i]
            print("{:30s} : {:-10.2f} | {}".format(i, lt, str(datetime.timedelta(seconds=lt))))

        if self.release_mode == False:
            # scoring
            pred_total = self.pred_total
            pred_move = self.pred_move
            pred_stay = self.pred_stay

            answer_move = 4
            answer_stay = 3
            answer_total = answer_stay + answer_move
            error = (pred_move - answer_move)**2 + (pred_stay - answer_stay)**2 + (pred_total - answer_total)**2
            print("answer > move:{} | stay:{} | total:{}".format(answer_move, answer_stay, answer_total))
            print("predict > move:{} | stay:{} | total:{}".format(pred_move, pred_stay, pred_total))
            print("error : ", error) 

        # return self.pred_move, self.pred_stay, self.pred_total
        return {f"{self.set_num}": [self.pred_move, self.pred_stay, self.pred_total]}



