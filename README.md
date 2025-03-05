This project aims to merge images taken from a drone. The main.py file accepts the input directory, output directory, matcher detector_name. The matcher can be "flann" or "bf", the detector can be "sift," "orb," or "akaza." Here's an example call for the main.py:

python main.py copiedD3 result flann orb

Additionally, there is a utils folder containing several scripts, including:

copy_files.py: accepts the original image directory, the directory where it will write the result, and the number of images to skip between two messages.
crop_black.py: used inside the main.py.
resize.py: accepts the name of the input image directory, the output image directory, and the value of the scale parameter.
shadow_script.py: accepts input and output folders, type_of_shadow, and probability. Type_of_shadow can be "random," "simple," or "realistic," and is responsible for the type of shadow that will be applied to the image. If probability is set to 1, a shadow will be applied to all images.
shadow.py: used inside the shadow_script.py.
shadow2.py: used inside the shadow_script.py.
smooth_stiching.py: accepts the first image path, second image path, and the number of layers in blending pyramid. In order to merge all images from the folder type True than original image directory, the directory where it will write the result and the number of layers in blending pyramid. 

In addition, in the "tests" folder there are several .bat scripts that represent several use cases.
There is also a "test_images" folder inside which contains 25 images from the main dataset.