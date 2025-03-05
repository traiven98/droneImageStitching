#Author Kyryl Serediuk and Vladyslav Varfolomeev
import os
import shutil
import sys

def copy_files(imagePath, resultPath, num_d):
    if not os.path.exists(resultPath):
        os.mkdir(resultPath)
    num_k = num_d
    for filename in os.listdir(imagePath):
        if filename.endswith(".JPG"):
            if num_k >= num_d:
                num_k = 0
                shutil.copy2(imagePath + '\\' + filename, resultPath + '\\' + filename)
                continue
            num_k+=1

if __name__ == "__main__":
    if not len(sys.argv) == 4:
        print("Error. Invalid number of arguments.")
    else:
        copy_files(sys.argv[1], sys.argv[2], int(sys.argv[3]))

