import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_images(image_dir):
    image_files = os.listdir(image_dir)
    image_list = []
    for im in image_files:
        img = plt.imread(image_dir + im)
        image_list.append(img)
    return image_list


def select_matches_ransac(pts0, pts1):
    H, mask = cv2.findHomography(pts0.reshape(-1, 1, 2), pts1.reshape(-1, 1, 2), cv2.RANSAC, 5.0)
    choice = np.where(mask.reshape(-1) == 1)[0]
    return pts0[choice], pts1[choice]


def get_best_match(cards, frame):
    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    keys = np.empty(len(cards), dtype=object)
    descriptors = np.empty(len(cards), dtype=object)

    for i in range(len(cards)):
        keys[i], descriptors[i] = orb.detectAndCompute(cards[i], mask=None)
    frame_keys, frame_desc = orb.detectAndCompute(frame, mask=None)

    most_matches = 0
    best_match = None
    coordinates = None

    frame_points = np.array([p.pt for p in frame_keys])
    for i in range(len(cards)):
        matches = matcher.match(frame_desc, descriptors[i])

        dist = np.array([m.distance for m in matches])
        ind1 = np.array([m.queryIdx for m in matches])
        ind2 = np.array([m.trainIdx for m in matches])
        card_points = np.array([p.pt for p in keys[i]])
        ds = np.argsort(dist)

        current_coordinates, _ = select_matches_ransac(frame_points[ind1[ds]], card_points[ind2[ds]])

        if current_coordinates.shape[0] > most_matches:
            most_matches = current_coordinates.shape[0]
            best_match = cards[i]
            coordinates = current_coordinates

    return best_match


def main():
    cards = np.load('card_images.npy', allow_pickle=True)
    frames = np.load('frames.npy', allow_pickle=True)

    start = time.time()
    get_best_match(cards, frames[0])
    end = time.time()
    print(end - start)

    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(frames[0])
    ax[1].imshow(get_best_match(cards, frames[0]))

    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(frames[1])
    ax[1].imshow(get_best_match(cards, frames[1]))

    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(frames[2])
    ax[1].imshow(get_best_match(cards, frames[2]))

    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(frames[3])
    ax[1].imshow(get_best_match(cards, frames[3]))

    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(frames[4])
    ax[1].imshow(get_best_match(cards, frames[4]))

    plt.show()


if __name__ == "__main__":
    main()
