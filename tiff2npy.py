import numpy as np
import cv2 as cv
from glob import glob
import os
from tqdm import tqdm

if __name__ == "__main__":

    data_root = 'data/uisee'
    for split in ['train','test']:
        img_paths = glob(os.path.join(data_root,split)+"/*.tiff")
        img_paths = sorted(img_paths,key=lambda x:int(x.split('/')[-1].split('.')[0]))
        np_arr = np.empty((len(img_paths),360,640,3),dtype=np.uint8)
        for i in tqdm(range(len(img_paths))):
            img = cv.imread(img_paths[i])
            img = cv.resize(img,(640,360),interpolation=cv.INTER_LINEAR)
            np_arr[i,:,:,:] = np.array(img,dtype=np.uint8)
        np.save(os.path.join(data_root,split+".npy"),np_arr)