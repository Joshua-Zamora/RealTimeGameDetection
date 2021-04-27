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
    coordinates_x = None  # Coordinates of the orb similarities
    coordinates_y = None

    frame_keys, frame_desc = orb.detectAndCompute(frame, mask=None)
    frame_points = np.array([p.pt for p in frame_keys])

    for i in range(len(cards)):
        keys = keys_descriptors[i][0]
        descriptor = keys_descriptors[i][1]
        matches = matcher.match(frame_desc, descriptor)

        dist = np.array([m.distance for m in matches])
        ind1 = np.array([m.queryIdx for m in matches])
        ind2 = np.array([m.trainIdx for m in matches])
        card_points = np.array([p.pt for p in keys])
        ds = np.argsort(dist)

        current_coordinates_x, current_coordinates_y = select_matches_ransac(frame_points[ind1[ds]], card_points[ind2[ds]])

        if current_coordinates_x.shape[0] > most_matches:
            most_matches = current_coordinates_x.shape[0]
            coordinates_x = current_coordinates_x
            coordinates_y = current_coordinates_y
            index = i

    return coordinates_x, coordinates_y, index


def get_card_color(coordinates, match):
    max_x = np.amax(coordinates[:, 0]).astype('int')  # creating a bounding box
    max_y = np.amax(coordinates[:, 1]).astype('int')
    min_x = np.amin(coordinates[:, 0]).astype('int')
    min_y = np.amin(coordinates[:, 1]).astype('int')

    red = np.mean(match[min_x: max_x, min_y: max_y][:, :, 2])  # Average red channel value
    green = np.mean(match[min_x: max_x, min_y: max_y][:, :, 1])  # Average green channel value
    blue = np.mean(match[min_x: max_x, min_y: max_y][:, :, 0])  # Average blue channel value

    if red > 175 and green > 175 and blue < 75:  # Yellow color is defined by (255, 255, 0)
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

        coordinates_x, coordinates_y, index = get_best_match_info(baselines, keys_descriptors, frame)

        if index < 13:  # If image is colored
            best_match = cards[get_card_color(coordinates_x, frame) + '-' + identifiers[index]]
        else:
            best_match = cards[identifiers[index]]

        combined = helper_functions.get_control_lines(cv2.cvtColor(best_match, cv2.COLOR_BGR2RGB),
                                                      frame, coordinates_y, coordinates_x)

        cv2.imshow('match', combined)  # Show frame with matching correspondence points

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
