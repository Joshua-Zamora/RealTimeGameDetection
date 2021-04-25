import cv2
import numpy as np
import helper_functions

orb = cv2.ORB_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def select_matches_ransac(pts0, pts1):
    H, mask = cv2.findHomography(pts0.reshape(-1, 1, 2), pts1.reshape(-1, 1, 2), cv2.RANSAC, 5.0)
    choice = np.where(mask.reshape(-1) == 1)[0]
    return pts0[choice], pts1[choice]


def get_best_match_info(cards, keys_descriptors, frame):
    most_matches = 0  # Keeps track of most Orb matches
    index = None  # Index # of image
    best_match = None  # Best matching image
    coordinates = None  # Coordinates of the orb similarities

    frame_keys, frame_desc = orb.detectAndCompute(frame, mask=None)
    frame_points = np.array([p.pt for p in frame_keys])

    for i in range(len(cards)):
        keys = keys_descriptors[i][0]
        descriptor = keys_descriptors[i][1]
        matches = matcher.match(frame_desc, descriptor)

        ind1 = np.array([m.queryIdx for m in matches])
        ind2 = np.array([m.trainIdx for m in matches])
        card_points = np.array([p.pt for p in keys])

        current_coordinates, _ = select_matches_ransac(frame_points[ind1], card_points[ind2])

        if current_coordinates.shape[0] > most_matches:
            most_matches = current_coordinates.shape[0]
            best_match = cards[i]
            coordinates = current_coordinates
            index = i

    return best_match, coordinates, index


def get_card_color(coordinates, match):
    max_x = np.amax(coordinates[:, 0]).astype('int')  # creating a bounding box
    max_y = np.amax(coordinates[:, 1]).astype('int')
    min_x = np.amin(coordinates[:, 0]).astype('int')
    min_y = np.amin(coordinates[:, 1]).astype('int')

    red = np.mean(match[min_x: max_x, min_y: max_y][:, :, 2])  # Average red channel value
    green = np.mean(match[min_x: max_x, min_y: max_y][:, :, 1])  # Average green channel value
    blue = np.mean(match[min_x: max_x, min_y: max_y][:, :, 0])  # Average blue channel value

    if red > 200 and green > 200 and blue < 50:  # Yellow color is defined by (255, 255, 0)
        color = 'Y'
    elif red > green and red > blue:
        color = 'R'
    elif green > red and green > blue:
        color = 'G'
    else:
        color = 'B'

    return color


def main():
    baselines = np.load('baselines.npy', allow_pickle=True)
    cards = np.load('all_cards.npy', allow_pickle=True)[0]
    keys_descriptors = helper_functions.get_keys_and_descriptors()

    identifiers = ['0', '1', '2', '3',
                   '4', '5', '6', '7',
                   '8', '9', 'P', 'R',
                   'S', '4', 'E', 'W', 'U']  # Card identities

    cap = cv2.VideoCapture(0)  # May need this parameter -> CAP_DSHOW
    cap.set(3, 1280)
    cap.set(4, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video capture unresponsive")
            break
        cv2.imshow('frame', frame)  # Show web cam frame

        best_match, coordinates, index = get_best_match_info(baselines, keys_descriptors, frame)

        if index < 13:  # If image is colored
            best_match = cards[get_card_color(coordinates, frame) + '-' + identifiers[index]]
        else:
            best_match = cards[identifiers[index]]

        cv2.imshow('match', cv2.cvtColor(best_match, cv2.COLOR_BGR2RGB))  # Show match

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
