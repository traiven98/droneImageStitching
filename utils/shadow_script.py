#Author Kyryl Serediuk and Vladyslav Varfolomeev
from shadow import add_spot_light, add_parallel_light
from shadow2 import shadow_V
import os 
import random
import cv2
import sys



def add_shadows(directory = '../test_images', new_path = '../shadowed_images', type_of_shadow = "random", probability = 0.3):
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    if type_of_shadow == "simple":
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            a = random.randint(1, 100)
            new_image = cv2.imread(f)
            if (a <= probability * 100):
                new_image = shadow_V(f, 0.3)
            n = os.path.join(new_path, filename)
            cv2.imwrite(n, new_image)
    elif type_of_shadow == "realistic":
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            a = random.randint(1, 100)
            new_image = cv2.imread(f)
            if (a <= probability * 100):
                shadow_c = random.choice([add_spot_light, add_parallel_light])
                new_image = shadow_c(f)
            n = os.path.join(new_path, filename)
            cv2.imwrite(n, new_image)
    elif type_of_shadow == "random":
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            a = random.randint(1, 100)
            new_image = cv2.imread(f)
            if (a <= probability * 100):
                shadow_c = random.choice([shadow_V, shadow_V ,add_spot_light, add_parallel_light])
                new_image = shadow_c(f)
            n = os.path.join(new_path, filename)
            cv2.imwrite(n, new_image)
    else:
        print("Wrong type!")



if __name__ == "__main__":
    if len(sys.argv) > 1:
        add_shadows(sys.argv[1], sys.argv[2], sys.argv[3],  float(sys.argv[4]))
    else:
        add_shadows()