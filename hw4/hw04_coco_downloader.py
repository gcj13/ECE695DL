
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import argparse
import os
from PIL import Image

class Coco_Downloader():
    def __init__(self,args):
        self.root_path = args.root_path
        self.coco_json_path = args.coco_json_path
        self.class_list = args.class_list
        self.images_per_class = args.images_per_class
        self.coco = COCO(self.coco_json_path)

        
    def save_images(self):
    #make direction
        for i in self.class_list:
#            if not os.path.exists(self.root_path + i):
#                os.makedirs(self.root_path + i)
    #download images
            catIds = self.coco.getCatIds(catNms=[i])
            img_ids = self.coco.getImgIds(catIds = catIds)
#            imgs = self.coco.loadImgs(img_ids)
#            print(imgs)
            #download   - Download COCO images from mscoco.org server.
            #Didn't see anyone using this method in code
#            for j in range(self.images_per_class):
            self.coco.download(tarDir = self.root_path + i, imgIds = img_ids[:self.images_per_class])
#            print(img_ids[self.images_per_class:(self.images_per_class+2)])
            #printing error in download, n-1/n is fully printed
            #check
            x = 0
            while True:
                path, dirs, files = next(os.walk(self.root_path + i))
                if len(files) == self.images_per_class:
                    break
                self.coco.download(tarDir = self.root_path + i, imgIds = img_ids[len(files):(2*self.images_per_class-len(files))])
                if x > 1000:
                    raise Exception("Too many iterations")
            #resize
            path, dirs, files = next(os.walk(self.root_path + i))
            #print(path,dirs,files)
            for file in files:
                im = Image.open(os.path.join(path,file))
                im_resized = im.resize((64, 64), Image.BOX)
                im_resized.save(os.path.join(path,file))
                print(file+" resized!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HW04 COCO downloader')
    parser.add_argument('--root_path', required=True, type= str)
    parser.add_argument('--coco_json_path', required=True, type=str)
    parser.add_argument('--class_list', required=True, nargs='*',type=str)
    parser.add_argument('--images_per_class', required=True, type=int)
    args, args_other = parser.parse_known_args()
    CocoDownloader = Coco_Downloader(args)
    CocoDownloader.save_images()
