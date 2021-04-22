import numpy as np
from matplotlib import pyplot as plt
import os


def save_card_baselines():
    imgs = [plt.imread('G-0.png'), plt.imread('G-1.png'), plt.imread('G-2.png'), plt.imread('G-3.png'),
            plt.imread('G-4.png'), plt.imread('G-5.png'), plt.imread('G-6.png'), plt.imread('G-7.png'),
            plt.imread('G-8.png'), plt.imread('G-9.png'), plt.imread('G-P.png'), plt.imread('G-R.png'),
            plt.imread('G-S.png'), plt.imread('4.png'), plt.imread('E.png'), plt.imread('uno-wild.jpg'),
            plt.imread('U.png')]
    np.save('card_baselines.npy', imgs)


def read_images(image_dir):
    image_files = os.listdir(image_dir)
    image_list = []
    for im in image_files:
        img = plt.imread(image_dir + im)
        image_list.append(img)
    return image_list
