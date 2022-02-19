import os
import torch
from torchvision.io import read_image
from torch.utils.data import DataLoader , Dataset
from PIL import Image


class your_dataset_class(Dataset):
    def __init__(self,args,transform):
#    '''
#    Make use of the arguments from the calling routine to initialise the variables
#    e.g. image path lists for cat and dog classes you could also maintain label_array
#    0 -- airplane
#    1 -- boat
#    2 -- cat .
#    .
#    .
#    9 -- truck
#    Initialize the required transform
#    '''
        self.root_path = args.root_path
        self.class_list = args.class_list
        self.transform = transform
        self.img_labels = []
        for cat in self.class_list:
            path, dirs, files = next(os.walk(self.root_path + cat))
            for file in files:
#            for path, dirs, files in os.walk(self.root_path + cat):
            #first image path, second label
                self.img_labels.append([cat+'/'+file,self.class_list.index(cat)])
        
    def __len__(self):
#    '''
#    return the total number of images
#    refer pytorch documentation for more details
#    '''
        l = 0
        for cat in self.class_list:
            path, dirs, files = next(os.walk(self.root_path + cat))
            l += len(files)
        return l

    def __getitem__(self,idx):
#    '''
#    Load color image(s), apply necessary data conversion
#    and transformation
#    e.g. if an image is loaded in HxWXC (Height X Width
#    X Channels) format
#    rearrange it in CxHxW format, normalize values from 0
#    -255 to 0-1
#    and apply the necessary transformation.
#    Convert the corresponding label in 1-hot encoding. Return the processed images
#    and labels in 1-hot encoded format
#    '''
        img_path = os.path.join(self.root_path, self.img_labels[idx][0])
        image = Image.open(img_path)
        label = torch.tensor(self.img_labels[idx][1])
        if self.transform:
            image = self.transform(image)#.to(dtype=torch.float64)
#'''        if self.target_transform:
#            label = self.target_transform(label)'''
        return image, label
        
