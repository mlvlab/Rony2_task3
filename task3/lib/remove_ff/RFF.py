
import os
import time
import random
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
# import pdb
import torch
from torchvision import transforms
from torchvision import transforms
from torchvision.models import wide_resnet50_2


def transfrom_imgs(img_list):

    # 1. 데이터 변환 준비
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize((224,224)),
            # transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 2. 데이터 변환
    transformed_imgs = []
    for img in img_list:
        pil_image = Image.fromarray(img)
        t_pil_image = data_transforms['val'](pil_image)
        transformed_imgs.extend(t_pil_image.unsqueeze(0))
    transformed_imgs = torch.stack(transformed_imgs)
    
    return transformed_imgs


def FE(images_per_ids):
    """
    각 id의 image들을 받아서, 각 피쳐를 반환해주는 함수.

    @param
    - images_per_ids: [img1, img2, ...]. img는 numpy array.
    - device: gpu device 설정
    @return
    - fts_outputs: 각 레이어에서 뽑은 피쳐 딕셔너리
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f"Feature extraction Start.. device={device}")
    # pdb.set_trace()
    # print("got image:",images_per_ids.shape)
    # load model
    model = wide_resnet50_2(pretrained=True, progress=True)
    model.to(device)
    model.eval()

    # set model's intermediate outputs
    outputs = []
    fts_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])
    def hook(module, input, output):
        outputs.append(output)
    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)
    model.avgpool.register_forward_hook(hook)

    for img_per_id in images_per_ids:
        # print("img_per_id: ",img_per_id.shape)
        # model prediction
        with torch.no_grad():
            _ = model(img_per_id.unsqueeze(0).to(device))
        
        # get intermediate layer outputs
        for k, v in zip(fts_outputs.keys(), outputs):
            fts_outputs[k].append(v)
        # initialize hook outputs
        outputs = []
    
    for k, v in fts_outputs.items():
        fts_outputs[k] = torch.cat(v, 0)
    
    fts_per_id = torch.flatten(fts_outputs['avgpool'], 1)
    # print("fts_per_id:",fts_per_id.shape)
    # m_fts_per_id = torch.median(fts_per_id, 0)
    # return m_fts_per_id.values
    m_fts_per_id = torch.mean(fts_per_id, 0)
    return m_fts_per_id


def RFF(crop_img, top_k = 5):
    
    # 1. crop_img의 id별 피쳐 추출.
    # print("1. feature extraction Start! > > >",end='')
    t_rff_1 = time.time()
    fts_per_id = []
    for ord, dict_per_video in enumerate(crop_img): # crop_img의 비디오별 이미지 정보 딕셔너리
        # print(f"{ord}th video")
        for person_id, imgs_per_id in dict_per_video.items(): # 딕셔너리의 id와 이미지들
            # print(f"id: {person_id}")
            
            # a. 피쳐 추출에 사용할 이미지 선택.
            imgs_per_id_t = []
            if len(imgs_per_id)<10:
                imgs_per_id_t = transfrom_imgs(imgs_per_id)
            else:
                rand_idx = random.sample(range(len(imgs_per_id)),10)
                rand_imgs_per_id = list(np.array(imgs_per_id)[rand_idx])
                imgs_per_id_t = transfrom_imgs(rand_imgs_per_id)
            
            
            # b. 피쳐 추출.
            fts_test = FE(imgs_per_id_t)

            # c. 피쳐 저장.
            fts_per_id.append(fts_test)
    #         print(f"피쳐 {len(fts_per_id)}개 쌓임")
    # print("피쳐 총 ",len(fts_per_id),"개")        
    fts_per_id = torch.stack(fts_per_id, 0).cpu().numpy()
    t_rff_2 = time.time() 
    # print(f"End ({t_rff_2-t_rff_1})s")
    
    
    # 2. 소방관 피쳐 가져오기
    # print("2. Classification Start! > > >",end='')
    fff_path = os.path.join(os.getcwd(),"task3/lib/remove_ff/fff_train_np.pkl" )
    # print('load train set feature from: %s' % fff_path)
    with open(fff_path, 'rb') as f:
        train_outputs = pickle.load(f)
    
    # 3. calculate distance matrix
    dist_matrix = calc_dist_matrix(fts_per_id, train_outputs)
    
    # 4. select K nearest neighbor and take average
    topk_values, topk_indexes = torch.topk(torch.tensor(dist_matrix), k=top_k, dim=1, largest=False)
    scores = torch.mean(topk_values, 1).cpu().detach().numpy()
    threshold_ff = 17.79682
    pred = (scores>=threshold_ff).astype(bool)
    # true가 사람(소방관 아님)
    t_rff_3 = time.time()
    # print(f"End ({t_rff_3-t_rff_2})s")
    
    return pred

def calc_dist_matrix(x, y):
    # Calculate Euclidean distance matrix with torch.tensor
    # n = x.size(0)
    # m = y.size(0)
    # d = x.size(1)
    # x = x.unsqueeze(1).expand(n, m, d)
    # y = y.unsqueeze(0).expand(n, m, d)
    # dist_matrix = torch.sqrt(torch.pow(x - y, 2).sum(2))

    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]
    x = np.expand_dims(x, axis=1)
    x = np.repeat(x, m, axis=1)
    y = np.expand_dims(y, axis=0)
    y = np.repeat(y, n, axis=0)
    dist_matrix = np.sqrt(np.power(x-y, 2).sum(2))
    return dist_matrix