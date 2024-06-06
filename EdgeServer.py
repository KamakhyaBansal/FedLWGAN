import torch

from threading import Thread 
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import random
import math
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot
from torchvision.utils import save_image

from pathlib import Path

import requests
import pickle
import gzip
import thread
import torch
import math
import torch.nn.functional as F
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import socket

# %config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from math import log2
import os
from tqdm import tqdm
import matplotlib.pylab as plt
#Loading data into Transforms
from torchvision import datasets, transforms
import torch.utils.data as data
from torch.utils.data import random_split

def check_loader():
    loader,_ = get_loader(128)
    image,_  = next(iter(loader))
    _,ax     = plt.subplots(3,3,figsize=(8,8))
    plt.suptitle('Some real samples')
    ind = 0
    for k in range(3):
        for kk in range(3):
            ax[k][kk].imshow((image[ind].permute(1,2,0)+1)/2)
            ind +=1
#check_loader()

#Get my ip address
hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)
 
print("Your Computer Name is:" + hostname)
print("Your Computer IP Address is:" + IPAddr)

number_of_samples = 2
host = str(IPAddr)  
port = 8061  
tclient = number_of_samples
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((host, port))  
sock.listen(tclient)  
connections = []
# Here, we are establishing the Connections to the server  
print('Initiating the clients')  
name_of_models_1 = []
name_of_models_2 = []
update_gen_path = "update_gen_0.pth"
update_disc_path = "update_disc_0.pth"

import time
def on_new_client(conn, i):
    connections.append(conn)
    buf_size = int(conn[0].recv(100).decode()) + 1
    print(buf_size)
    data = conn[0].recv(buf_size)
    if not data:  
        print("No file shared")  
    # Here, we are creating a new file at the server end and writing the data  
    filename = 'edge_gen_' + str(i) +'.pth'  
    f = open(filename, "wb")          # here, we are opening a file in the write mode  
    f.write(data)             # here, we are writing the data to the file   

    print('Received successfully! The New filename is:', filename)  
    f.close() 
    model_dict_1['model'+str(i)] = torch.load(filename)
    name_of_models_1.append('model'+str(i))

    buf_size = int(conn[0].recv(100).decode()) + 1
    print(buf_size)
    data = conn[0].recv(buf_size)
    if not data:  
        print("No file shared")  
    # Here, we are creating a new file at the server end and writing the data  
    filename = 'edge_disc_' + str(i) +'.pth'  
    f = open(filename, "wb")          # here, we are opening a file in the write mode  
    f.write(data)             # here, we are writing the data to the file   

    print('Received successfully! The New filename is:', filename)  
    f.close() 
    model_dict_2['model'+str(i)] = torch.load(filename)
    name_of_models_2.append('model'+str(i))

def on_new_client_send(conn, i):
    #connections.append(conn)
    #print('Connected with the client for sending updated weights', i+1)   

    f = open(update_gen_path, "rb")
    data = f.read()
    print("Data length", str(len(data)))
    conn[0].send(str(len(data)).encode())

    if not data:
        print("No Data")
    else:
        conn[0].send(data) 
    #print(len(data))
    print('Sent successfully! filename ', update_gen_path)  
    f.close()          # here, we are closing the file 
    time.sleep(3)
    f = open(update_disc_path, "rb")
    data = f.read()
    print("Data length", str(len(data)))
    conn[0].send(str(len(data)).encode())

    if not data:
        print("No Data")
    else:
        conn[0].send(data) 
    #print(len(data))
    print('Sent successfully! filename ', update_disc_path)  
    f.close()          # here, we are closing the file 

def get_averaged_weights():
    #load average weights to main model
    gen = Generator(
        Z_DIM, W_DIM, IN_CHANNELS, CHANNELS_IMG
    ).to(DEVICE)
    gen_dict = gen.state_dict()
    for i in range(number_of_samples):
        for key in model_dict_1[name_of_models_1[i]].keys():
            gen_dict[key] += model_dict_1[name_of_models_1[i]][key]
    for key in model_dict_1[name_of_models_1[i]].keys():
        gen_dict[key] = gen_dict[key]/number_of_samples


    disc_dict = critic.state_dict()
    for i in range(number_of_samples):
        for key in model_dict_2[name_of_models_1[i]].keys():
            disc_dict[key] += model_dict_2[name_of_models_2[i]][key]
    for key in model_dict_2[name_of_models_2[i]].keys():
        disc_dict[key] = disc_dict[key]/number_of_samples
    gen.load_state_dict(gen_dict)
    critic.load_state_dict(disc_dict)
    torch.save(gen_dict,update_gen_path)
    torch.save(disc_dict,update_disc_path)

while (1):
    threads  = []
    print("In ")
    for i in range(tclient):  
        conn = sock.accept()  
        print('Successfully Connected with the client', i+1, conn) 
        t = Thread(target=on_new_client, args=[conn,i])
        threads.append(t)
        t.start()
        print(i)
    for t in threads:
        t.join()
    get_averaged_weights()
    print("Weights averaged ")
    threads = []
    for conn in connections:  
        print("Sending...") 
        t = Thread(target=on_new_client_send, args=[conn,i])
        threads.append(t)
        t.start()
        for t in threads:
            t.join()
    '''for conn in connections:  
            conn[0].close()   '''   

    
#Try various combinations
'''
4: 0.8, 0.1, 0.1
5: [0.1,0.8,0.1]
6: [0.1,0.1,0.8]
'''
#model_weights = [weights(0.34,1.84), weights(0.29,2.03), weights(0.38,1.83)]
model_weights = [0.1,0.1,0.8]

files_list = [('/content/drive/MyDrive/my_gan_run/f1/model_2.pt'),'/content/drive/MyDrive/my_gan_run/f2/model_2.pt','/content/drive/MyDrive/my_gan_run/f3/model_2.pt']
def get_averaged_weights():
    #load average weights to main model

    gen_dict = torch.load(files_list[0])
    for i in range(0,len(model_weights)):
        for key in gen_dict['GAN'].keys() :
            if i == 0:
                gen_dict['GAN'][key] =  gen_dict['GAN'][key] * model_weights[0]
            else:
                gen_dict['GAN'][key] += model_weights[i] * torch.load(files_list[i])['GAN'][key]

    for key in gen_dict['GAN'].keys():
        gen_dict['GAN'][key] = gen_dict['GAN'][key]/sum(model_weights)

    torch.save(gen_dict,'/content/drive/MyDrive/my_gan_run/f1/model_6.pt')
get_averaged_weights()
