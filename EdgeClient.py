!pip install lightweight-gan
from threading import Thread 
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import random
import math
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot


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
from torchvision.utils import save_image
def get_loader(image_size):
  transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.Resize((image_size,image_size)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                     ])
  train_set =  datasets.ImageFolder("D:\\Data\\modified-dataset\\train\\", transform=transform)
  #val_set =  datasets.ImageFolder('/content/drive/MyDrive/ConsumerElectronics/val/', transform=transform)
  #dataset_size = len(train_set) + len(val_set)


  transform1 = transforms.Compose([transforms.RandomVerticalFlip(),
                                        transforms.Resize((image_size,image_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                      ])
  train_set1 =  datasets.ImageFolder('D:\\Data\\modified-dataset\\train\\', transform=transform1)
  #train_subset1 = Subset(train_set1, train_indices)

  transform2 = transforms.Compose([transforms.RandomVerticalFlip(),
                                  transforms.RandomRotation(10),
                                        transforms.Resize((image_size,image_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                      ])
  train_set2 =  datasets.ImageFolder('D:\\Data\\modified-dataset\\train\\', transform=transform2)
  #train_subset2 = Subset(train_set2, train_indices)

  transform3 = transforms.Compose([transforms.RandomVerticalFlip(),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomRotation(5),
                                        transforms.Resize((image_size,image_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                      ])
  train_set3 =  datasets.ImageFolder('D:\\Data\\modified-dataset\\train\\', transform=transform3)
  #train_subset3 = Subset(train_set3, train_indices)

  transform4 = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.Resize((image_size,image_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                      ])
  train_set4 =  datasets.ImageFolder('D:\\Data\\modified-dataset\\train\\', transform=transform4)
  #train_subset4 = Subset(train_set4, train_indices)

  transform5 = transforms.Compose([transforms.RandomVerticalFlip(),
                                        transforms.Resize((image_size,image_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                      ])
  train_set5 =  datasets.ImageFolder('D:\\Data\\modified-dataset\\train\\', transform=transform5)
  #train_subset5 = Subset(train_set5, train_indices)

  transform6 = transforms.Compose([transforms.RandomVerticalFlip(),
                                  transforms.RandomRotation(10),
                                        transforms.Resize((image_size,image_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                      ])
  train_set6 =  datasets.ImageFolder('D:\\Data\\modified-dataset\\train\\', transform=transform6)
  #train_subset6 = Subset(train_set6, train_indices)

  transform7 = transforms.Compose([transforms.RandomVerticalFlip(),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomRotation(5),
                                        transforms.Resize((image_size,image_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                      ])
  train_set7 =  datasets.ImageFolder('D:\\Data\\modified-dataset\\train\\', transform=transform7)
  #train_subset7 = Subset(train_set7, train_indices)

  i_dataset = torch.utils.data.ConcatDataset([train_set,train_set1])
  i_dataset1 = torch.utils.data.ConcatDataset([train_set2,train_set3])
  i_dataset2 = torch.utils.data.ConcatDataset([train_set4,train_set5])
  i_dataset3 = torch.utils.data.ConcatDataset([train_set6,train_set7])
  i_dataset4 = torch.utils.data.ConcatDataset([i_dataset,i_dataset1])
  i_dataset5 = torch.utils.data.ConcatDataset([i_dataset2,i_dataset3])
  increased_dataset = torch.utils.data.ConcatDataset([i_dataset4,i_dataset5])
  batch_size = BATCH_SIZES[int(log2(image_size/4))]
  # We define a set of data loaders that we can use for various purposes later.
  train_loader = data.DataLoader(increased_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
  #val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)
  return train_loader, increased_dataset

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
check_loader()
!lightweight_gan --data /content/drive/MyDrive/FedData/Split2 --image-size 32 --name /content/drive/MyDrive/my_gan_run/f1 --batch-size 16 --gradient-accumulate-every 4 --num-train-steps 1500 --amp

!lightweight_gan \
  --name /content/drive/MyDrive/my_gan_run/f1 \
  --load-from 1 \
  --generate \
  --num-image-tiles 100

from PIL import Image
import matplotlib.pyplot as plt
import glob
import math
import os

# Path to the directory containing the generated images
generated_images_dir = '/content/drive/MyDrive/my_gan_run/f1-generated-1'

# Use glob to get the paths of all generated images
generated_image_paths = glob.glob(os.path.join(generated_images_dir, '*.jpg'))

# Calculate the number of rows and columns for the grid
num_rows = 8
num_cols = 8

# Create a subplot with the specified number of rows and columns
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))

# Display each generated image in the grid
for i in range(num_rows):
    for j in range(num_cols):
        # Calculate the index for the current image
        index = i * num_cols + j

        # Check if the index is within the number of generated images
        if index < len(generated_image_paths):
            img_path = generated_image_paths[index]
            img = Image.open(img_path)

            # Display the image on the current subplot
            axes[i, j].imshow(img)
            axes[i, j].axis('off')

# Adjust layout to prevent clipping of titles
plt.tight_layout()
plt.show()

import torch
#After update
'''
4: 0.8, 0.1, 0.1
5: [0.1,0.8,0.1]
6: [0.1,0.1,0.8]
'''
#model_weights = [weights(0.34,1.84), weights(0.29,2.03), weights(0.38,1.83)]
model_weights = [0.1,0.1,0.8]

files_list = [('/content/drive/MyDrive/my_gan_run/f1/model_2.pt'),'/content/drive/MyDrive/my_gan_run/f2/model_2.pt','/content/drive/MyDrive/my_gan_run/f3/model_2.pt']
import socket
host = xx-xx-xx-xx
port = 8062
edge_no = 0

torch.save(critic.state_dict(),'edge_gen_'+str(edge_no)+'.pth')
torch.save(gen.state_dict(),'edge_disc_'+str(edge_no)+'.pth')

import time
def retrain_with_updated_weights():
	#Train the model
	#model, acc = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=5)
	# Save weights
	wght_gen_file = 'edge_gen_'+str(edge_no)+'.pth'
	wght_disc_file = 'edge_disc_'+str(edge_no)+'.pth'
	torch.save(critic.state_dict(),wght_disc_file)
	torch.save(gen.state_dict(),wght_gen_file)
	#Read weights from weight file
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	# Connecting with Server
	sock.connect((host, port))
	print("Connected with server")
	
	#Send weights to server
	f = open(wght_gen_file, "rb")
	data = f.read()
	sock.send(str(len(data)).encode())
	if not data:
		print("No Data")
	else:
		sock.send(data)
		print("Generator Weights Sent")
	time.sleep(2)
	print("sending Disc weights")
	f = open(wght_disc_file, "rb")
	data = f.read()
	sock.send(str(len(data)).encode())
	if not data:
		print("No Data")
	else:
		sock.send(data)
		print("Discriminator Weights Sent")
	'''
	#Receive updated weights from server
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	# Connecting with Server
	sock.connect((host, port))
	print("Connected with server")'''
	write_file_gen = 'update_gen_'+str(edge_no)+'.pth'
	buf_size_str = sock.recv(100).decode()
	#print(buf_size_str)
	buf_size = int(buf_size_str) + 1
	#print(buf_size)
	data = sock.recv(buf_size)
	# print(data)
	f = open(write_file_gen, "wb")
	if not data:
		print("No data in file")
	else:
		f.write(data)
		f.close()
		print("Successfully written ",write_file_gen)
	
	write_file_disc = 'update_disc_'+str(edge_no)+'.pth'
	buf_size_str = sock.recv(100).decode()
	#print(buf_size_str)
	buf_size = int(buf_size_str) + 1
	#print(buf_size)
	data = sock.recv(buf_size)
	# print(data)
	f = open(write_file_disc, "wb")
	if not data:
		print("No data in file")
	else:
		f.write(data)
		f.close()
		print("Successfully written ",write_file_disc)

	#Load new weights
	gen.load_state_dict(torch.load(write_file_gen))
	critic.load_state_dict(torch.load(write_file_disc))
	#Retrain
	step = int(log2(START_TRAIN_IMG_SIZE / 4)+4)
	for num_epochs in PROGRESSIVE_EPOCHS[step:]:
		alpha = 1e-7
		loader, dataset = get_loader(4*2**step)
		print('Curent image size: '+str(4*2**step))
		for epoch in range(num_epochs):
			print(f'Epoch [{epoch + 1}/ {num_epochs}')
			alpha = train_fn(
				critic, gen, loader, dataset, step, alpha, opt_critic, opt_gen
			)
		generate_examples(gen, step)
		step +=1
	return 1

retrain_with_updated_weights()
