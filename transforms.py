import numpy as np
import cv2

def order_points(points):
    ordered_arr = np.zeros((4,2), dtype='float32')
    sum_coords = np.sum(points, axis=1)
    diff_coords = np.diff(points, axis=1)

    ordered_arr[0] = points[np.argmin(sum_coords)]
    ordered_arr[1] = points[np.argmin(diff_coords)]
    ordered_arr[2] = points[np.argmax(sum_coords)]
    ordered_arr[3] = points[np.argmax(diff_coords)]

    return ordered_arr

def distance(pt1, pt2):
    return np.sqrt(((pt1[0] - pt2[0]) ** 2) + ((pt1[1] - pt2[1]) ** 2))
    

def four_point_transform(image, points):
    points_ordered = order_points(points)
    (top_left, top_right, bottom_right, bottom_left) = points_ordered

    top_width = distance(top_left, top_right)
    bottom_width = distance(bottom_left, bottom_right)
    final_width = int(max(top_width, bottom_width))

    left_height = distance(top_left, bottom_left)
    right_height = distance(top_right, bottom_right)
    final_height = int(max(left_height, right_height))

    points_transform = np.array([
        [0, 0],
        [final_width-1, 0],
        [final_width-1, final_height-1],
        [0, final_height-1]
    ], dtype='float32')

    M = cv2.getPerspectiveTransform(points_ordered, points_transform)
    image_transform = cv2.warpPerspective(image, M, (final_width, final_height))

    return image_transform