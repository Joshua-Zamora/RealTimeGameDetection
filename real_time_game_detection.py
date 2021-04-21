import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import transform as tf
import cv2
import os
from scipy import interpolate as spy_int
import matplotlib._color_data as mcd
import time


def read_images(image_dir):
    image_files = os.listdir(image_dir)
    image_list = []
    for im in image_files:
        img = mpimg.imread(image_dir + im)
        image_list.append(img)
    return image_list

def select_matches_ransac(pts0, pts1):
    H, mask = cv2.findHomography(pts0.reshape(-1, 1, 2), pts1.reshape(-1, 1, 2), cv2.RANSAC, 5.0)
    choice = np.where(mask.reshape(-1) == 1)[0]
    return pts0[choice], pts1[choice]

def get_keys_and_descriptors(image_list):
    orb = cv2.ORB_create()
    keys = np.empty(len(image_list), dtype = object)
    descriptors = np.empty(len(image_list), dtype = object)
    for i in range(len(image_list)):
        keys[i],descriptors[i] = orb.detectAndCompute(image_list[i], mask=None)
    return keys,descriptors


def find_match(image_list,image):
    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    keys,descriptors = get_keys_and_descriptors(image_list)
    image_keys, image_descriptor = orb.detectAndCompute(image, mask= None)
    
    num_matches = np.empty(len(image_list), dtype=object)
    dest_points = np.array([p.pt for p in image_keys])
    
    for i in range(len(image_list)):
        src_points = np.array([p.pt for p in keys[i]])
        matches = matcher.match(descriptors[i], image_descriptor)
        
        dist = np.array([m.distance for m in matches])
        ind1 = np.array([m.queryIdx for m in matches])
        ind2 = np.array([m.trainIdx for m in matches])
        ds = np.argsort(dist)
        
        good_matches, _ = select_matches_ransac(src_points[ind1[ds]], dest_points[ind2[ds]])
        num_matches[i] = good_matches.shape[0]
    
    ind = num_matches.argmax()
    print(ind)
    return image_list[ind]
        
        
    
    


if __name__ == "__main__":
    path = '.\\uno\\'
    im1 = plt.imread('uno_test_1.jpg')
    im2 = plt.imread('uno_test_2.jpg')
    im3 = plt.imread('uno_test_3.jpg')
    im4 = plt.imread('uno_test_4.jpg')
    im5 = plt.imread('uno_test_5.jpg')
    uno_list = read_images(path)
    
    start = time.time()
    find_match(uno_list,im1)
    end = time.time()
    print(end-start)
    
    
    
    
    fig, ax = plt.subplots(ncols = 2)
    ax[0].imshow(im1)
    ax[1].imshow(find_match(uno_list,im1))
    
    fig, ax = plt.subplots(ncols = 2)
    ax[0].imshow(im2)
    ax[1].imshow(find_match(uno_list,im2))
    
    fig, ax = plt.subplots(ncols = 2)
    ax[0].imshow(im3)
    ax[1].imshow(find_match(uno_list,im3))
    
    fig, ax = plt.subplots(ncols = 2)
    ax[0].imshow(im4)
    ax[1].imshow(find_match(uno_list,im4))
    
    fig, ax = plt.subplots(ncols = 2)
    ax[0].imshow(im5)
    ax[1].imshow(find_match(uno_list,im5))
    
    