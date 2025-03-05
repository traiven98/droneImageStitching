python ../utils/copy_files.py ../test_images ../copiedD3 3
python ../utils/resize.py ../copiedD3 ../copiedD3 0.3
python ../utils/shadow_script.py ../copiedD3 ../copiedD3 random 1
python ../main.py ../copiedD3 ../result
