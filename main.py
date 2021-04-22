import cv2
import numpy as np


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

    return best_match, coordinates


def get_card_color(coordinates, match):
    max_x = np.amax(coordinates[:, 0])
    max_y = np.amax(coordinates[:, 1])
    min_x = np.amin(coordinates[0, :])
    min_y = np.amin(coordinates[1, :])

    average_color = np.mean(match[min_x: max_x, min_y, max_y])

    if average_color[0] > 255 - 50 and average_color[1] > 255 - 50 and average_color[2] < 50:
        color = 'y'
    elif average_color[0] > average_color[1] and average_color[0] > average_color[2]:
        color = 'r'
    elif average_color[1] > average_color[0] and average_color[1] > average_color[2]:
        color = 'g'
    else:
        color = 'b'

    return color


def main():
    cards = np.load('card_.npy', allow_pickle=True)

    cap = cv2.VideoCapture(0)  # CAP_DSHOW
    cap.set(3, 1280)
    cap.set(4, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video capture unresponsive")
            break

        cv2.imshow('frame', frame)

        best_match, coordinates = get_best_match_info(cards, frame)
        print(best_match[0][0])
        exit()
        get_card_color(coordinates, best_match)
        cv2.imshow('match', best_match)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
