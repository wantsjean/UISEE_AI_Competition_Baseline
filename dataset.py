from data_process import *
from config import Config
import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import accimage
import PIL
import torchvision.transforms.functional as F
import random
import cv2 as cv
os.environ['DISPLAY'] = ':10'

class THODataset(Dataset):
    def __init__(self,data_root,label_path,dataset='uisee',split = 'train',resize = (360,640),crop_size=(342,608),seq_len = 10,val_rate=0.2):
        self.dataset = dataset
        self.split = split
        self.resize = resize
        self.crop_size = crop_size
        self.seq_len = seq_len
        img_paths,labels = get_data(data_root,label_path)
        self.img_paths_list,self.labels_list = split_data(img_paths,labels,split,0.2)
        self.img_list,self.label_list = generate_sequence(self.img_paths_list,self.labels_list,seq_len)
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        label = self.label_list[index]
        buffer= self.fast_process(self.img_list[index])
        label = torch.from_numpy(np.array(label))
        return buffer,label.float()

    def fast_process(self,img_list):
        w, h = accimage.Image(img_list[0]).size
        new_h = self.resize[0]
        new_w = self.resize[1]
        if self.split == 'train':
            crop_size_h,crop_size_w = self.crop_size
            crop_w = random.randint(0, new_w - crop_size_w)
            crop_h = random.randint(0, new_h - crop_size_h)
        else:
            crop_size_h, crop_size_w = self.resize
            crop_w = (new_w - crop_size_w) // 2
            crop_h = (new_h - crop_size_h) // 2

        buffer = torch.empty((self.seq_len, 3, h, w))

        flip_flag = False
        if self.split == 'train':
            if np.random.random() < 0.5:
                flip_flag = True
        for i in range(self.seq_len):
            # img = accimage.Image(img_list[i]).crop((crop_w,crop_h,crop_w+crop_size_w,crop_h+crop_size_h))
            img = accimage.Image(img_list[i])
            if flip_flag:
                img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            buffer[i, :, :, :] = F.to_tensor(img)
        return buffer * 2 - 1


if __name__ == '__main__':
    config = Config()
    dataset = THODataset(config.data_root,config.label_path,config.dataset,split='train')
    train_loader = DataLoader(dataset,1,shuffle=True)
    for i in range(3):
        for input,target in tqdm(train_loader,total=len(train_loader)):
            input = input[0].numpy().transpose(0,2,3,1)
            for j in range(10):
                arr = (input[j]+1)*127
                arr = np.array(arr,dtype=np.uint8)
                cv.imshow("arr",cv.resize(arr,(213,120)))
                cv.waitKey(500)