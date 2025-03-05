#Author Kyryl Serediuk and Vladyslav Varfolomeev

import cv2
import numpy as np
import sys
import os
import utils.crop_black as crop_black


def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([
        [0, 0],
        [0, rows1],
        [cols1, rows1],
        [cols1, 0]
    ])
    list_of_points_1 = list_of_points_1.reshape(-1, 1, 2)

    temp_points = np.float32([
        [0, 0],
        [0, rows2],
        [cols2, rows2],
        [cols2, 0]
    ])
    temp_points = temp_points.reshape(-1, 1, 2)

    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]

    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img2,
                                     H_translation.dot(H),
                                     (x_max - x_min, y_max - y_min))
    output_img[translation_dist[1]:rows1 + translation_dist[1],
    translation_dist[0]:cols1 + translation_dist[0]] = img1

    return output_img


def warp(img1, img2, matcher, detector_name, min_match_count=10):
    if detector_name == "sift":
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    elif detector_name == "orb":
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    elif detector_name == "akaza":
        akaze = cv2.AKAZE_create()
        keypoints1, descriptors1 = akaze.detectAndCompute(img1, None)
        keypoints2, descriptors2 = akaze.detectAndCompute(img2, None)
    else:
        raise Exception("Sorry, don't valid detector")
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    fl = None
    if matcher == "flann":
        fl = cv2.FlannBasedMatcher(index_params, search_params)
    elif matcher == "bf":
        fl = cv2.BFMatcher()
    else:
        raise Exception("Sorry, don't valid matcher")

    matches = fl.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m1, m2 in matches:
        if m1.distance < 0.7 * m2.distance:
            good_matches.append(m1)
    src_pts = np.float32([keypoints1[good_match.queryIdx].pt
                          for good_match in good_matches]).reshape(-1, 1, 2)

    dst_pts = np.float32([keypoints2[good_match.trainIdx].pt
                          for good_match in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    result = warpImages(img2, img1, M)
    return result


def save_image(directory, file_name, image):
    print("ok")
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(directory + '\\' + file_name, image)




def stitch_image_in_sub_directory(input_directory, output_directory, matcher="flann", detector_name="sift"):
    image_collage = {}
    file_name = 'final_image_stitched.jpg'
    try:
        if (os.path.isdir(input_directory) == True):
            for dirname, _, filenames in os.walk(input_directory):
                # sort file name in a directory
                filenames = sorted(filenames)
                print('Stitching is start...')
                print('---------------------------------')
                image_collage = cv2.imread(os.path.join(dirname, filenames[0]))
                previous_image = filenames[0]
                temp_num = 1

                for index, item in enumerate(filenames):
                    if index == 0:
                        continue
                    main_image = cv2.imread(os.path.join(dirname, filenames[index]))

                    print('{}. Stitching {} AND {} in process'.format(index, previous_image, filenames[index]))
                    print('---------------------------------')
                    image_collage = warp(image_collage, main_image, matcher, detector_name)
                    image_collage = crop_black.crop_Black(image_collage)
                    previous_image = filenames[index]
                    save_image(output_directory, file_name, image_collage)
    except:
        save_image(output_directory, file_name, image_collage)


#####

if __name__ == "__main__":
    if len(sys.argv) == 3:
        stitch_image_in_sub_directory(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        stitch_image_in_sub_directory(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5:
        stitch_image_in_sub_directory(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("Error. Invalid number of arguments.")


