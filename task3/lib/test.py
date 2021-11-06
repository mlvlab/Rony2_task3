import pickle

pth = '/home/ubuntu/task3/temp/t3_res/set_01/set_01.pickle'

with open(pth, 'rb') as pkl:
    videos = pickle.load(pkl)
for vid in videos:
    for i, imgs in vid.items():
        print(i)
        k=1
        for im in imgs:
            print(k)
            print(im.shape)
            k = k+1