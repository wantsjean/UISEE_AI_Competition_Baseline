import cv2 as cv
import os
from tqdm import tqdm
root = 'data/uisee'
size = (640,360)
for mode in ['train','test']:
    files = os.listdir(os.path.join(root,mode))
    for file in tqdm(files,total=len(files)):
        filename,ext = file.split('.')
        if ext=='tiff':
            img = cv.imread(os.path.join(root,mode,file))
            h,w = img.shape[:2]
            # print(h,w)
            cv.imwrite(os.path.join(root,mode,filename+".jpg"),cv.resize(img,size,interpolation=cv.INTER_LINEAR))
