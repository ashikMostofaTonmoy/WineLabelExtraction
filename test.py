# In[]
from matplotlib import pyplot as plt  # plot images
import os  # folder directory navigation
from paddleocr import PaddleOCR, draw_ocr  # main OCR dependencies
import cv2  # opencv
import sys
import platform
print(platform.python_version())
print(sys.path)
# %%

# need to run only once to download and load model into memory
# ocr = PaddleOCR(det_algorithm='EAST', use_angle_cls=True, lang="en")
ocr = PaddleOCR(use_angle_cls=True, lang="en")
img_path = 'data/paired_lavbel/WhatsApp Image 2022-06-14 at 2.34.35 PM.jpeg'
img = cv2.imread(img_path)
result = ocr.ocr(img_path, cls=True)

# for line in result:
#     print(line)


# %%

# Extracting detected components
boxes = [res[0] for res in result]
texts = [res[1][0] for res in result]
scores = [res[1][1] for res in result]

# Specifying font path for draw_ocr method
font_path = os.path.join('PaddleOCR', 'doc', 'fonts', 'latin.ttf')

# imports image
img = cv2.imread(img_path)

# draw annotations on image
annotated = draw_ocr(img, boxes, texts, scores, font_path=font_path)

# Saving the image
file = os.path.split(img_path)[1]
saved_image_name = os.path.join('data', 'output', file)
cv2.imwrite(saved_image_name, annotated)

# %%
