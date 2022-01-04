import pickle
import numpy as np
from PIL import Image
from collections import OrderedDict

import torch
from torchvision import transforms
from torchvision.models import wide_resnet50_2


def transfrom_imgs(img_list):

    # 1. 데이터 변환 준비
    data_transforms = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.Resize((224,224)),
        # transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    # 2. 데이터 변환
    transformed_imgs = []
    for img in img_list:
        # if img.shape[1]
        pil_image = Image.fromarray(img)
        t_pil_image = data_transforms(pil_image)
        transformed_imgs.extend(t_pil_image.unsqueeze(0))
    transformed_imgs = torch.stack(transformed_imgs)
    
    return transformed_imgs


def FE(images_list, gpu_device):

    # load model
    model = wide_resnet50_2()
    model.load_state_dict(torch.load('task3/lib/remove_ff/weights/wide_resnet50_2-95faca4d.pth'))
    model.to(gpu_device)
    model.eval()

    # set model's intermediate outputs
    outputs = []
    def hook(module, input, output):
        outputs.append(output)
    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)
    model.avgpool.register_forward_hook(hook)
    
    output_features = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])

    for img in images_list:

        # model prediction
        with torch.no_grad():
            _ = model(img.unsqueeze(0).to(gpu_device))
        
        # get intermediate layer outputs
        for k, v in zip(output_features.keys(), outputs):
            output_features[k].append(v)

        # initialize hook outputs
        outputs = []
    
    for k, v in output_features.items():
        output_features[k] = torch.cat(v, 0)    
    features_of_id = torch.flatten(output_features['avgpool'], 1)

    return features_of_id

def calc_dist_matrix(x, y):
    # Calculate Euclidean distance matrix with torch.tensor
    # n = x.size(0)
    # m = y.size(0)
    # d = x.size(1)
    # x = x.unsqueeze(1).expand(n, m, d)
    # y = y.unsqueeze(0).expand(n, m, d)
    # dist_matrix = torch.sqrt(torch.pow(x - y, 2).sum(2))

    # Calculate Euclidean distance matrix with numpy array
    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]
    x = np.expand_dims(x, axis=1)
    x = np.repeat(x, m, axis=1)
    y = np.expand_dims(y, axis=0)
    y = np.repeat(y, n, axis=0)

    dist_matrix = np.sqrt(np.power(x-y, 2).sum(2))
    return dist_matrix


def RFF(crop_img, device, top_k = 10):
    select_n = 15
    # print('소방관 제거 시작..')
    # 소방관 피쳐 가져오기
    with open("task3/lib/remove_ff/fff_train_np.pkl", 'rb') as f:
        train_outputs = pickle.load(f)
    
    # fts_per_id = {} # id별 대표 피쳐를 담을 리스트.
    # features_of_id = []
    person_id_list = []
    for vid_num, dict_per_video in enumerate(crop_img): # crop_img의 비디오별 이미지 정보 딕셔너리
        print(f'video. {vid_num}')
        for person_id, imgs_per_id in dict_per_video.items(): # 딕셔너리의 id와 이미지들
            # 이미지 select_n 개 이하 선택 및 변형.
            imgs_per_id_t = [] # id의 이미지들이 변형되어 저장될 리스트.
            if len(imgs_per_id)<=select_n:
                imgs_per_id_t = transfrom_imgs(imgs_per_id)
            else:
                rand_idx = np.random.choice(len(imgs_per_id), select_n)
                rand_imgs_per_id = list(np.array(imgs_per_id, dtype=object)[rand_idx])
                imgs_per_id_t = transfrom_imgs(rand_imgs_per_id)
            
            # 피쳐 추출.
            fts_of_id = FE(images_list=imgs_per_id_t, gpu_device=device)
            # calculate distance matrix
            dist_matrix = calc_dist_matrix(fts_of_id.cpu().detach().numpy(), train_outputs)

            # topk
            topk_values, _ = torch.topk(torch.tensor(dist_matrix), k=top_k, dim=1, largest=False)
            # calculate average
            topk_values_mean = torch.mean(topk_values, dim=1)
            # representative feature of id
            feature_of_id = torch.median(topk_values_mean)
            # features_of_id.append(feature_of_id)
            # add this feature to dict
            # fts_per_id[person_id]=feature_of_id
            # print(f"id. '{person_id}' | img: {len(imgs_per_id)}개 | ft: {feature_of_id} |")
            if  feature_of_id>= 18.0 or feature_of_id<15.5:
                person_id_list.append(int(person_id))
    
    
    return person_id_list