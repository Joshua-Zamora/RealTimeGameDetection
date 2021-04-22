import numpy as np
from matplotlib import pyplot as plt
import os


def save_card_baselines():
    path = './cards/'
    imgs = [plt.imread(path + 'G-0.png'), plt.imread(path + 'G-1.png'), plt.imread(path + 'G-2.png'),
            plt.imread(path + 'G-3.png'), plt.imread(path + 'G-4.png'), plt.imread(path + 'G-5.png'),
            plt.imread(path + 'G-6.png'), plt.imread(path + 'G-7.png'), plt.imread(path + 'G-8.png'),
            plt.imread(path + 'G-9.png'), plt.imread(path + 'G-P.png'), plt.imread(path + 'G-R.png'),
            plt.imread(path + 'G-S.png'), plt.imread(path + '4.png'), plt.imread(path + 'E.png'),
            plt.imread(path + 'W.png'), plt.imread(path + 'U.png')]

    imgs = [img.astype('uint8') * 255 for img in imgs]
    np.save('baselines.npy', imgs)


def save_all_cards():
    path = './cards/'
    cards = dict()
    cards['4'] = plt.imread(path + '4.png')
    cards['E'] = plt.imread(path + '4.png')
    cards['U'] = plt.imread(path + 'U.png')
    cards['W'] = plt.imread(path + 'W.png')
    cards['G-0'] = plt.imread(path + 'G-0.png')
    cards['G-1'] = plt.imread(path + 'G-1.png')
    cards['G-2'] = plt.imread(path + 'G-2.png')
    cards['G-3'] = plt.imread(path + 'G-3.png')
    cards['G-4'] = plt.imread(path + 'G-4.png')
    cards['G-5'] = plt.imread(path + 'G-5.png')
    cards['G-6'] = plt.imread(path + 'G-6.png')
    cards['G-7'] = plt.imread(path + 'G-7.png')
    cards['G-8'] = plt.imread(path + 'G-8.png')
    cards['G-9'] = plt.imread(path + 'G-9.png')
    cards['G-P'] = plt.imread(path + 'G-P.png')
    cards['G-R'] = plt.imread(path + 'G-R.png')
    cards['G-S'] = plt.imread(path + 'G-S.png')
    cards['R-0'] = plt.imread(path + 'R-0.png')
    cards['R-1'] = plt.imread(path + 'R-1.png')
    cards['R-2'] = plt.imread(path + 'R-2.png')
    cards['R-3'] = plt.imread(path + 'R-3.png')
    cards['R-4'] = plt.imread(path + 'R-4.png')
    cards['R-5'] = plt.imread(path + 'R-5.png')
    cards['R-6'] = plt.imread(path + 'R-6.png')
    cards['R-7'] = plt.imread(path + 'R-7.png')
    cards['R-8'] = plt.imread(path + 'R-8.png')
    cards['R-9'] = plt.imread(path + 'R-9.png')
    cards['R-P'] = plt.imread(path + 'R-P.png')
    cards['R-R'] = plt.imread(path + 'R-R.png')
    cards['R-S'] = plt.imread(path + 'R-S.png')
    cards['B-0'] = plt.imread(path + 'B-0.png')
    cards['B-1'] = plt.imread(path + 'B-1.png')
    cards['B-2'] = plt.imread(path + 'B-2.png')
    cards['B-3'] = plt.imread(path + 'B-3.png')
    cards['B-4'] = plt.imread(path + 'B-4.png')
    cards['B-5'] = plt.imread(path + 'B-5.png')
    cards['B-6'] = plt.imread(path + 'B-6.png')
    cards['B-7'] = plt.imread(path + 'B-7.png')
    cards['B-8'] = plt.imread(path + 'B-8.png')
    cards['B-9'] = plt.imread(path + 'B-9.png')
    cards['B-P'] = plt.imread(path + 'B-P.png')
    cards['B-R'] = plt.imread(path + 'B-R.png')
    cards['B-S'] = plt.imread(path + 'B-S.png')
    cards['Y-0'] = plt.imread(path + 'Y-0.png')
    cards['Y-1'] = plt.imread(path + 'Y-1.png')
    cards['Y-2'] = plt.imread(path + 'Y-2.png')
    cards['Y-3'] = plt.imread(path + 'Y-3.png')
    cards['Y-4'] = plt.imread(path + 'Y-4.png')
    cards['Y-5'] = plt.imread(path + 'Y-5.png')
    cards['Y-6'] = plt.imread(path + 'Y-6.png')
    cards['Y-7'] = plt.imread(path + 'Y-7.png')
    cards['Y-8'] = plt.imread(path + 'Y-8.png')
    cards['Y-9'] = plt.imread(path + 'Y-9.png')
    cards['Y-P'] = plt.imread(path + 'Y-P.png')
    cards['Y-R'] = plt.imread(path + 'Y-R.png')
    cards['Y-S'] = plt.imread(path + 'Y-S.png')

    np.save('all_cards.npy', [cards])


def read_images(image_dir):
    image_files = os.listdir(image_dir)
    image_list = []
    for im in image_files:
        img = plt.imread(image_dir + im)
        image_list.append(img)
    return image_list


def show_image(image):
    f, ax = plt.subplots(1, 1)
    ax.imshow(image)
    plt.show()