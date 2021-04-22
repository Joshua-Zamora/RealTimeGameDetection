import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys


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


def get_best_match_info(cards, frame):
    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    most_matches = 0
    best_match = None
    coordinates = None

    frame_keys, frame_desc = orb.detectAndCompute(frame, mask=None)
    frame_points = np.array([p.pt for p in frame_keys])

    for i in range(len(cards)):
        keys, descriptor = orb.detectAndCompute(cards[i], mask=None)
        matches = matcher.match(frame_desc, descriptor)

        ind1 = np.array([m.queryIdx for m in matches])
        ind2 = np.array([m.trainIdx for m in matches])
        card_points = np.array([p.pt for p in keys])

        current_coordinates, _ = select_matches_ransac(frame_points[ind1], card_points[ind2])

        if current_coordinates.shape[0] > most_matches:
            most_matches = current_coordinates.shape[0]
            best_match = cards[i]
            coordinates = current_coordinates

    return best_match


def main():
    cards = np.load('card_images.npy', allow_pickle=True)

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video capture unresponsive")
            break

        cv2.imshow('frame', frame)
        cv2.imshow('match', get_best_match_info(cards, frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
