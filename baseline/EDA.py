# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pickle
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

path = '/raid/data/ml/cifar-10-batches-py/'
os.listdir(path)


# +
def read_data(name):
    with open(name,'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

def get_image(data):
    return np.rollaxis(data.reshape([3,32,32]),0,3)


# -

data = read_data(f'{path}/data_batch_1')

list(data.keys())

y = data[b'labels']
len(y)

data[b'data'].shape

32*32*3

for i in range(0,9):
    plt.subplot(330+1+i)
    plt.imshow(get_image(data[b'data'][i]))
plt.show()

print("Editting the script!!")
