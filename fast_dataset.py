from data_process import *
from config import Config
import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
from scipy import ndimage
import PIL
from PIL import Image
import torchvision.transforms.functional as F
import random
import cv2 as cv

os.environ['DISPLAY'] = ':10'

class THODataset(Dataset):
    def __init__(self,data,data_root,label_path=None,dataset='uisee',split = 'train',resize = (360,640),crop_size=(320,576),seq_len = 10,val_rate=0.2):
        self.dataset = dataset
        self.split = split
        self.resize = resize
        self.crop_size = crop_size
        self.seq_len = seq_len
        if self.split !='submit_eval':
            img_paths,labels = get_data(data_root,label_path,'train')
        else:
            img_paths,labels = get_data(data_root,None,'test')
        self.img_paths_list,self.labels_list = split_data(img_paths,labels,split,0.2)
        self.img_list,self.label_list = generate_sequence(self.img_paths_list,self.labels_list,seq_len)
        self.data = data
        self.h,self.w = self.data.shape[1:3]
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        if self.split!='submit_eval':
            label = self.label_list[index]
            buffer,label= self.fast_process(self.img_list[index],label)
            return buffer,label
        else:
            buffer,label= self.fast_process(self.img_list[index],None)
            return buffer
        

    def fast_process(self,img_list,label=None):
        img_list = [int(path.split('/')[-1].split('.')[0]) for path in img_list]

        flip_flag = False
        if self.split == 'train' or self.split=='submit_train':
            if np.random.random() < 0.5:
                flip_flag = True
            
        img = self.data[img_list]

        if flip_flag:
            img = img[:,:,::-1,:].copy()
            if label!=None:
                label[0] = -label[0]
        if label!=None:
            label = torch.from_numpy(np.array(label)).float()
        buffer = torch.from_numpy(img)
        return buffer ,label


if __name__ == '__main__':
    config = Config()
    data = np.load(os.path.join(data_root,'train.npy'))
    dataset = THODataset(data,config.data_root,config.label_path,config.dataset,split='train')
    train_loader = DataLoader(dataset,16,shuffle=True,num_workers=4)
    for i in range(3):
        for input,target in tqdm(train_loader,total=len(train_loader)):
            input = input.cuda().float()/255
            input = input*2 - 1

            input = input[0].cpu().numpy()
            print(target[0][0])
            for j in range(10):
                arr = (input[j]+1)*127
                arr = np.array(arr,dtype=np.uint8)
                cv.imshow("arr",cv.resize(arr,(213,120)))
                cv.waitKey(500)