#Author Kyryl Serediuk and Vladyslav Varfolomeev
import cv2
import numpy as np

def crop_Black(img):
    height, width = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)

    y, x = np.nonzero(mask)
    top, left = np.min(y), np.min(x)
    bottom, right = np.max(y), np.max(x)

    left = max(left - 20, 0)
    top = max(top - 20, 0)
    right = min(right + 20, width)
    bottom = min(bottom + 20, height)
    img_cropped = img[top:bottom, left:right]
    return img_cropped
