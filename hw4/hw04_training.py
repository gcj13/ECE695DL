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
#model
class Net(nn.Module):
    def __init__(self,net_num):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3)       ## (A)
        self.conv2 = nn.Conv2d(128, 128, 3)      ## (B)
        self.pool = nn.MaxPool2d(2, 2)
        self.net_num = net_num
        if net_num == 1:
            self.fc1 = nn.Linear(128*31*31, 1000)        ## (C) 64*64->62*62->31*31
        elif net_num == 2:
            self.fc1 = nn.Linear(128*14*14, 1000)          ## (C) 64*64->62*62->31*31->29*29->14*14
        elif net_num == 3:
            self.conv1 = nn.Conv2d(3, 128, 3, padding = 1)
            self.fc1 = nn.Linear(128*15*15, 1000)           ## (C) 64*64->32*32->30*30->15*15
        self.fc2 = nn.Linear(1000, 10)      #10classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
    ##  Uncomment the next statement and see what happens to the
    ##  performance of your classifier with and without padding.
    ##  Note that you will have to change the first arg in the
    ##  call to Linear in line (C) above and in the line (E)
    ##  shown below.  After you have done this experiment, see
    ##  if the statement shown below can be invoked twice with
    ##  and without padding.  How about three times?
        if self.net_num == 1:
            x = x.view(-1, 128*31*31)    ## (E)
        elif self.net_num == 2:
            x = self.pool(F.relu(self.conv2(x)))        ## (D)
            x = x.view(-1, 128*14*14)
        elif self.net_num == 3:
            x = self.pool(F.relu(self.conv2(x)))        ## (D)
            x = x.view(-1, 128*15*15)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
def run_code_for_training(net,train_data_loader,net_save_path):
    net = copy.deepcopy(net)
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    epochs = 10
    Loss_runtime = []
    for epoch in range(epochs):
        start_time = time.time()
        print("---------starting epoch %s ---------" % (epoch + 1))
        running_loss = 0.0
        for i, data in enumerate(train_data_loader):
            inputs, labels = data
#            print(type(inputs))
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % 500 == 0:
                print("\n[epoch:%d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / float(500)))
                Loss_runtime.append(running_loss / float(500))
                running_loss = 0.0
                i = time.time()
                print("---------time executing for epoch %s : %s seconds---------" % (epoch + 1, (time.time() - start_time)))
    torch.save(net,net_save_path)
    return Loss_runtime

                
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
    parser.add_argument('--coco_json_path', required=True, type=str)
    parser.add_argument('--class_list', required=True, nargs='*',type=str)
    parser.add_argument('--images_per_class', required=True, type=int)
    args, args_other = parser.parse_known_args()
    #CocoDownloader = Coco_Downloader(args)
    #CocoDownloader.save_images()
#transform
    transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = your_dataset_class(args,transform)
#    print(len(train_dataset))  #__len__ works
#    print(train_dataset[0][1])
    train_data_loader = torch.utils.data.DataLoader(dataset = train_dataset , batch_size=10, shuffle=True, num_workers=0)
#model
    plt.figure()
    model1 = Net(1)
    loss1 = run_code_for_training(model1,train_data_loader,"net1.pth")
    plt.plot(loss1, label = "Net1 Training Loss")
    model2 = Net(2)
    loss2 = run_code_for_training(model2,train_data_loader,"net2.pth")
    plt.plot(loss2, label = "Net2 Training Loss")
    model3 = Net(3)
    loss3 = run_code_for_training(model3,train_data_loader,"net3.pth")
    plt.plot(loss3, label = "Net3 Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig("train_loss.jpg")
