#Author Kyryl Serediuk and Vladyslav Varfolomeev
import cv2
import os
import sys

def resize(filePath, resultPath, scale):
    if not os.path.exists(resultPath):
        os.mkdir(resultPath)
    for filename in os.listdir(filePath):
        if filename.endswith(".JPG"):
            image = cv2.imread(filePath + '\\' + filename)
            width = int(image.shape[1] * scale)
            height = int(image.shape[0] * scale)
            dim = (width, height)

    # resize image
            result = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite(resultPath + '/' + filename, result)

if __name__ == "__main__":
    if not len(sys.argv) == 4:
        print("Error. Invalid number of arguments.")
    else:
        resize(sys.argv[1], sys.argv[2], float(sys.argv[3]))
