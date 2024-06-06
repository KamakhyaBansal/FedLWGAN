!pip install lightweight-gan

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
