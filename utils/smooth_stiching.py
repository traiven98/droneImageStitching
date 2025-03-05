#Author Kyryl Serediuk and Vladyslav Varfolomeev
import cv2
import imutils as imutils
import numpy as np
import os
import sys

cv2.ocl.setUseOpenCL(False)

def blend(image):
        result = None
        result_mask = None
        result, result_mask = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO).belnd(result, result_mask)
        result = cv2.convertScaleAbs(result)
        return result, result_mask

def detectAndDescribe(image, method=None): 
    assert method is not None, "You need to define a feature detection method. Values are: 'sift', 'surf'"

    # detect and extract features from the image
    if method == 'sift':
        descriptor = cv2.SIFT_create(nfeatures=124)
    elif method == 'surf':
        descriptor = cv2.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()

    # get keypoints and descriptors
    (kps, features) = descriptor.detectAndCompute(image, None)

    return (kps, features)

def createMatcher(method, crossCheck):
    "Create and return a Matcher Object"

    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf

def matchKeyPointsBF(featuresA, featuresB, method):
    bf = createMatcher(method, crossCheck=True)

    # Match descriptors.
    best_matches = bf.match(featuresA, featuresB)

    # Sort the features in order of distance.
    # The points with small distance (more similarity) are ordered first in the vector
    rawMatches = sorted(best_matches, key=lambda x: x.distance) 
    return rawMatches

def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    bf = createMatcher(method, crossCheck=False)
    # compute the raw matches and initialize the list of actual matches
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    #print("Raw matches (knn):", len(rawMatches))
    matches = []

    # loop over the raw matches
    for m,n in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches

def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    # convert the keypoints to numpy arrays
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])

    if len(matches) > 4:

        # construct the two sets of points
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])

        # estimate the homography between the sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                         reprojThresh)

        return (matches, H, status)
    else:
        return None



feature_extractor = 'sift'
feature_matching = 'bf'
 
 

def stiching_with_blending(FirstImage, SecondImage, layer):

    trainImg = cv2.imread(FirstImage) 
    trainImg = cv2.cvtColor(trainImg, cv2.COLOR_BGR2RGB)
    trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_RGB2GRAY)


    queryImg = cv2.imread(SecondImage) 
    queryImg = cv2.cvtColor(queryImg, cv2.COLOR_BGR2RGB)    
    queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_RGB2GRAY) 

    kpsA, featuresA = detectAndDescribe(trainImg_gray, method=feature_extractor)
    kpsB, featuresB = detectAndDescribe(queryImg_gray, method=feature_extractor)  


    if feature_matching == 'bf':
        matches = matchKeyPointsBF(featuresA, featuresB, method=feature_extractor)
        img3 = cv2.drawMatches(trainImg, kpsA, queryImg, kpsB, matches[:100],
                            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    elif feature_matching == 'knn':
        matches = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, method=feature_extractor)
        img3 = cv2.drawMatches(trainImg, kpsA, queryImg, kpsB, np.random.choice(matches, 100),
                            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) 

    M = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=4)
    if M is None:
        print("Error!")
    (matches, H, status) = M
    
    width = trainImg.shape[1] + queryImg.shape[1]
    height = trainImg.shape[0] + queryImg.shape[0]

    result = cv2.warpPerspective(trainImg, H, (width, height))

    result_copy = result.copy()
    gp_result = [result_copy]
    queryImg_copy = queryImg.copy()
    gp_queryImg = [queryImg_copy]
    for i in range(layer):
        result_copy = cv2.pyrDown(result_copy)
        gp_result.append(result_copy)
        queryImg_copy = cv2.pyrDown(queryImg_copy)
        gp_queryImg.append(queryImg_copy)

    result_copy = gp_result[layer - 1]
    lp_result = [result_copy]
    queryImg_copy = gp_queryImg[layer - 1]
    lp_queryImg = [queryImg_copy]

    for i in range(layer - 1, 0, -1):
        height, width, channels = gp_result[i-1].shape
        dst_size = (width,height)
        gaussian_expanded = cv2.pyrUp(gp_result[i], dstsize=dst_size)
        laplacian = cv2.subtract(gp_result[i-1], gaussian_expanded)
        lp_result.append(laplacian)

    for i in range(layer - 1, 0, -1):
        height, width, channels = gp_queryImg[i-1].shape
        dst_size = (width,height)
        gaussian_expanded = cv2.pyrUp(gp_queryImg[i], dstsize=dst_size)
        laplacian = cv2.subtract(gp_queryImg[i-1], gaussian_expanded)
        lp_queryImg.append(laplacian)

    panorama_pyramid = []
    n = 0

    for result_lap, queryImg_lap in zip(lp_result, lp_queryImg):
        n += 1
        result_lap[0:queryImg_lap.shape[0], 0:queryImg_lap.shape[1]] = queryImg_lap
        laplacian = result_lap
        panorama_pyramid.append(laplacian)



    panorama_reconstruct = panorama_pyramid[0]
    for i in range(1, layer): 
        height, width, channels = panorama_pyramid[i].shape
        dst_size = (width,height) 
        panorama_reconstruct = cv2.pyrUp(panorama_reconstruct, dstsize=dst_size) 
        panorama_reconstruct = cv2.add(panorama_pyramid[i], panorama_reconstruct)

    temp_result = panorama_reconstruct
    gray = cv2.cvtColor(temp_result, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

            # Finds contours from the binary image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    c = max(cnts, key=cv2.contourArea)

    (x, y, w, h) = cv2.boundingRect(c)

            # crop the image to the bbox coordinates
    temp_result = temp_result[y:y + h, x:x + w]

    
    temp_result = cv2.cvtColor(temp_result, cv2.COLOR_RGB2BGR)
    cv2.imwrite('result_image.jpg', temp_result)
    return temp_result  

def stiching_with_blending_dataset(input_directory = 'dir_for_test', output_directory = 'dir_for_test_res', layer = 4):
    name = 'result_image'
    for dirname, _, filenames in os.walk(input_directory): 
        image_collage = cv2.imread(os.path.join(dirname, filenames[0]))
        previous_image = input_directory + '/' + filenames[0]   
        temp_num = 1
        print(len(filenames))   
        for index, item in enumerate(filenames):
                print(index)
                if index == 0:
                    continue 
                print(previous_image)
                print(filenames[index])
                trainImg = cv2.imread(previous_image) 
                trainImg = cv2.cvtColor(trainImg, cv2.COLOR_BGR2RGB)
                trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_RGB2GRAY)


                queryImg = cv2.imread(input_directory + '/' + filenames[index]) 
                queryImg = cv2.cvtColor(queryImg, cv2.COLOR_BGR2RGB)    
                queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_RGB2GRAY) 

                kpsA, featuresA = detectAndDescribe(trainImg_gray, method=feature_extractor)
                kpsB, featuresB = detectAndDescribe(queryImg_gray, method=feature_extractor) 
 


                if feature_matching == 'bf':
                    matches = matchKeyPointsBF(featuresA, featuresB, method=feature_extractor)
                    img3 = cv2.drawMatches(trainImg, kpsA, queryImg, kpsB, matches[:100],
                                        None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                elif feature_matching == 'knn':
                    matches = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, method=feature_extractor)
                    img3 = cv2.drawMatches(trainImg, kpsA, queryImg, kpsB, np.random.choice(matches, 100),
                                        None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) 

                M = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=4)
                if M is None:
                    print("Error!")
                (matches, H, status) = M 
                width = trainImg.shape[1] + queryImg.shape[1]
                height = trainImg.shape[0] + queryImg.shape[0]

                result = cv2.warpPerspective(trainImg, H, (width, height))

                result_copy = result.copy()
                gp_result = [result_copy]
                queryImg_copy = queryImg.copy()
                gp_queryImg = [queryImg_copy]
                for i in range(layer):
                    result_copy = cv2.pyrDown(result_copy)
                    gp_result.append(result_copy)
                    queryImg_copy = cv2.pyrDown(queryImg_copy)
                    gp_queryImg.append(queryImg_copy)

                result_copy = gp_result[layer - 1]
                lp_result = [result_copy]
                queryImg_copy = gp_queryImg[layer - 1]
                lp_queryImg = [queryImg_copy]

                for i in range(layer - 1, 0, -1):
                    height, width, channels = gp_result[i-1].shape
                    dst_size = (width,height)
                    gaussian_expanded = cv2.pyrUp(gp_result[i], dstsize=dst_size)
                    laplacian = cv2.subtract(gp_result[i-1], gaussian_expanded)
                    lp_result.append(laplacian)

                for i in range(layer - 1, 0, -1):
                    height, width, channels = gp_queryImg[i-1].shape
                    dst_size = (width,height)
                    gaussian_expanded = cv2.pyrUp(gp_queryImg[i], dstsize=dst_size)
                    laplacian = cv2.subtract(gp_queryImg[i-1], gaussian_expanded)
                    lp_queryImg.append(laplacian)

                panorama_pyramid = []
                n = 0

                for result_lap, queryImg_lap in zip(lp_result, lp_queryImg):
                    n += 1
                    result_lap[0:queryImg_lap.shape[0], 0:queryImg_lap.shape[1]] = queryImg_lap
                    laplacian = result_lap
                    panorama_pyramid.append(laplacian)



                panorama_reconstruct = panorama_pyramid[0]
                for i in range(1, layer): 
                    height, width, channels = panorama_pyramid[i].shape
                    dst_size = (width,height)
                    panorama_reconstruct = cv2.pyrUp(panorama_reconstruct, dstsize=dst_size)
                    panorama_reconstruct = cv2.add(panorama_pyramid[i], panorama_reconstruct)

                temp_result = panorama_reconstruct
                gray = cv2.cvtColor(temp_result, cv2.COLOR_BGR2GRAY)
                thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

                        # Finds contours from the binary image
                cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)

                c = max(cnts, key=cv2.contourArea)

                (x, y, w, h) = cv2.boundingRect(c)

                        # crop the image to the bbox coordinates
                temp_result = temp_result[y:y + h, x:x + w]

                
                temp_result = cv2.cvtColor(temp_result, cv2.COLOR_RGB2BGR)
                new_name = output_directory + '/' + name + str(index) + '.jpg'
                cv2.imwrite(new_name, temp_result)
                previous_image = new_name

if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == 'True': 
        stiching_with_blending_dataset(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif len(sys.argv) > 2:
        stiching_with_blending(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    else:
        print("no input params")