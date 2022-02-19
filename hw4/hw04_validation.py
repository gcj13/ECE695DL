import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from hw04_coco_downloader import Coco_Downloader
import torchvision
import torchvision.transforms as tvt
import random
from hw04_dataloader import your_dataset_class
import numpy as np
import os
import copy
import time
import matplotlib.pyplot as plt
from hw04_dataloader import your_dataset_class
from hw04_training import Net
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

def run_code_for_validation(net, validation_data_loader,net_save_path, class_len):
    net = copy.deepcopy(net)
    net = net.to(device)
    Confusion_Matrix = torch.zeros(class_len, class_len)
    for i, data in enumerate(validation_data_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.tolist()
        for label,prediction in zip(labels,predicted):
            Confusion_Matrix[label][prediction] += 1
            
    return Confusion_Matrix
    
if __name__ == '__main__':
##checking for GPU.
    if torch.cuda.is_available() == True:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
#reference:playing with cifar10
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmarks=False
    os.environ['PYTHONHASHSEED'] = str(seed)
    #torch.save(net.state_dict, PATH) # to save model in path file
#parser
    parser = argparse.ArgumentParser(description='HW04 COCO downloader')
    parser.add_argument('--root_path', required=True, type= str)
#    parser.add_argument('--coco_json_path', required=True, type=str)
    parser.add_argument('--class_list', required=True, nargs='*',type=str)
#    parser.add_argument('--images_per_class', required=True, type=int)
    args, args_other = parser.parse_known_args()
    #CocoDownloader = Coco_Downloader(args)
    #CocoDownloader.save_images()
#transform
    transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    validation_dataset = your_dataset_class(args,transform)
#    print(len(validation_dataset))  #__len__ works
#    print(validation_dataset[0][1])
    validation_data_loader = torch.utils.data.DataLoader(dataset = validation_dataset , batch_size=10, shuffle=True, num_workers=0)
#model
    class_len = len(args.class_list)

    model1 = torch.load("net1.pth",map_location=device)
    model1.eval()
    plt.figure(figsize = (10,7))
    Confusion_Matrix = run_code_for_validation(model1,validation_data_loader,"net1.pth",len(args.class_list))
    sns.heatmap(Confusion_Matrix,annot = True, fmt = "", cmap = "Blues", cbar = False, xticklabels = args.class_list, yticklabels = args.class_list)
    plt.title("Net 1")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.savefig("net1_confusion_matrix.jpg")

    model2 = torch.load("net2.pth",map_location=device)
    model2.eval()
    plt.figure(figsize = (10,7))
    Confusion_Matrix = run_code_for_validation(model2,validation_data_loader,"net2.pth",len(args.class_list))
    sns.heatmap(Confusion_Matrix,annot = True, fmt = "", cmap = "Blues", cbar = False, xticklabels = args.class_list, yticklabels = args.class_list)
    plt.title("Net 2")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.savefig("net2_confusion_matrix.jpg")

    model3 = torch.load("net3.pth",map_location=device)
    model3.eval()
    plt.figure(figsize = (10,7))
    Confusion_Matrix = run_code_for_validation(model3,validation_data_loader,"net3.pth",len(args.class_list))
    sns.heatmap(Confusion_Matrix,annot = True, fmt = "", cmap = "Blues", cbar = False, xticklabels = args.class_list, yticklabels = args.class_list)
    plt.title("Net 3")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.savefig("net3_confusion_matrix.jpg")

