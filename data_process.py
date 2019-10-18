#%%
import os
from glob import glob
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from tqdm import tqdm
import math
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

os.environ['DISPLAY'] = ':10'

data_root = 'data/uisee'
label_path = os.path.join(data_root,'train','label.txt')

def get_labels(path):
    labels = []
    with open(path,'r') as f:
        raw_data = f.read().strip().split('\n')
        for data in raw_data:
            index,x,y,z,rx,ry,rz,angle,speed = data.strip().split()
            # labels.append([int(index),float(x),float(y),float(z),float(rx),float(ry),
            #             float(rz),float(angle),float(speed)])
            labels.append([int(index),float(angle),float(speed)/10])
    return labels

def show_image(img_paths,labels,size,wait=0):
    for i in tqdm(range(len(img_paths)),total=len(img_paths)):
        img = cv.imread(img_paths[i])
        angle,speed = labels[i][-2:]
        h,w= img.shape[:2]
        new_w = size
        new_h = int(size*h/w)
        img = cv.resize(img,(new_w,new_h))
        r = size/5
        cv.line(img, (new_w//2, new_h), (new_w//2-int(r*math.sin(angle*math.pi/180)), new_h-int(r*math.cos(angle*math.pi/180))), (101, 67, 252), 3)
        cv.circle(img,(new_w//2, new_h),radius=int(r),color=(152,152,255),thickness=3)
        cv.putText(img, "Angle :%.2f" % (angle),(10,22),cv.FONT_HERSHEY_COMPLEX,1,(101, 255, 110),2)
        cv.putText(img, "Speed:%.2f" % (speed), (10, 50), cv.FONT_HERSHEY_COMPLEX, 1, (101, 255, 110), 2)
        cv.rectangle(img, (int(new_w*3/4), int(new_h - speed*new_h//20)), (int(new_w*5/6), new_h), (255, 80, 80),-1)
        cv.imshow("img",img)
        cv.waitKey(wait)

def get_image_paths(data_root,mode='train'):
    img_paths = glob(os.path.join(data_root,mode)+'/*.jpg')
    return sorted(img_paths,key=lambda x:int(x.split('/')[-1].split('.')[0]))

def check(labels,imgs):
    assert len(labels)==len(imgs), str(len(labels))+" not the same as "+str(len(imgs))
    for i in range(len(labels)):
        assert int(imgs[i].split('/')[-1].split('.')[0]) == labels[i][0]

def get_data(data_root,label_path,mode='train'):
    labels = get_labels(label_path)
    img_paths = get_image_paths(data_root,mode=mode)
    check(labels,img_paths)
    labels = [x[1:] for x in labels]
    return img_paths,labels


def split_data(img_paths,labels,split,val_rate=0.2):
    split_list = [0, 4436, 4760, 5132, len(img_paths)]
    img_paths_list = []
    labels_list = []
    for i in range(4):
        val_len = int((split_list[i+1]-split_list[i])*val_rate)
        train_len = split_list[i+1]-split_list[i] - val_len
        if split == 'train':
            img_paths_list.append(img_paths[split_list[i]:split_list[i+1]-val_len])
            labels_list.append(labels[split_list[i]:split_list[i+1]-val_len])
        else:
            img_paths_list.append(img_paths[split_list[i]+train_len:split_list[i + 1]])
            labels_list.append(labels[split_list[i]+train_len:split_list[i + 1]])
    return img_paths_list,labels_list

def generate_sequence(img_paths_list,labels_list,seq_len=10):
    img_list = []
    label_list = []
    for i in range(len(img_paths_list)):
        for j in range(len(img_paths_list[i])-seq_len):
            img_list.append(img_paths_list[i][j:j+seq_len])
            label_list.append(labels_list[i][j+seq_len-1])
    return img_list,label_list


def plot_line(x,y,y1,label="noname",label1 = "noname",path=None):
    plt.rcParams['figure.figsize'] = (20.0, 8.0)
    ax = sns.lineplot(x=x, y=y,label=label)
    ax = sns.lineplot(x=x, y=y1,label=label1,color="coral")
    plt.savefig(path)
    plt.clf()


# img_paths,labels = get_data(data_root,label_path)
# img_paths_list,labels_list = split_data(img_paths,labels)

# show_image(img_paths_list[2],labels_list[2],600,wait=1000)

# get_labels(label_path)

# show_image(os.path.join(data_root,'train'))


#%%
