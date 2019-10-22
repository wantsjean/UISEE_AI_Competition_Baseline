import lmdb
import os
import cv2 as cv
from tqdm import tqdm
from pathlib import Path




class LMDBStorage(object):
    def __init__(self, path):
        super().__init__()
        self.database = lmdb.open(path, map_size=1 << 40)

    def put(self,img_path,key):
        with self.database.begin(write=True, buffers=True) as txn:
            data = (Path(img_path)).open("rb").read()
            key = key
            txn.put(key.encode(), data)
                

    def close(self):
        self.database.close()

if __name__ == "__main__":

    data_root = 'data/uisee'
    lmdb_path = os.path.join(data_root,'lmdb_database')
    db = LMDBStorage(lmdb_path)
    for split in ['train','test']:
        for file in tqdm(os.listdir(os.path.join(data_root,split))):
            img_path = os.path.join(data_root,split,file)
            filename,ext = file.split('.')
            if ext=='tiff':
                img = cv.imread(img_path)
                h,w = img.shape[:2]
                cv.imwrite("tmp.jpg",cv.resize(img,(640,360),interpolation=cv.INTER_LINEAR))
                db.put("tmp.jpg",img_path)

