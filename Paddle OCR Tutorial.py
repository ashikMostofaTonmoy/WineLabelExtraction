# %%
from matplotlib import pyplot as plt  # plot images
import os  # folder directory navigation
from paddleocr import PaddleOCR, draw_ocr  # main OCR dependencies
import cv2  # opencv
import sys
import platform
print(platform.python_version())
print(sys.path)

# %%
# !which python

# %% [markdown]
# # 1. Install and Import Dependencies

# %%
# GitHub repo installation of paddle
# !python -m pip install paddlepaddle-gpu==2.0.0 -i https://mirror.baidu.com/pypi/simple
# !python -m pip install paddlepaddle-gpu==2.3.0 -i https://mirror.baidu.com/pypi/simple
# conda install -c paddle paddlepaddle-gpu

# %%
# Install paddle OCR
# !pip install paddleocr
# !conda install -c esri paddleocr -y

# %%
# Clone paddle OCR repo - get FONTS for visualization
# !git clone https://github.com/PaddlePaddle/PaddleOCR

# %%

# %% [markdown]
# # 2. Instantiate Model and Detect

# %%
# Setup model
ocr_model = PaddleOCR(use_angle_cls=True, lang='en')


# %%
dataDirectory = 'data\paired_lavbel'
for root, d_name, f_name in os.walk(dataDirectory):
    for file in f_name:
        img_path = os.path.join(root, file)
        # print(img_path)

        # Run the ocr method on the ocr model
        result = ocr_model.ocr(img_path)

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
        saved_image_name = os.path.join('data', 'output', file)
        cv2.imwrite(saved_image_name, annotated)


# # %%
# img_path = os.path.join('.', 'drug1.jpg')

# # %%
# # Run the ocr method on the ocr model
# result = ocr_model.ocr(img_path)

# # %%
# result

# # %%
# for res in result:
#     print(res[1][0])

# # %% [markdown]
# # # 3. Visualise Results

# # %%
# # Extracting detected components
# boxes = [res[0] for res in result]
# texts = [res[1][0] for res in result]
# scores = [res[1][1] for res in result]

# # %%
# # Specifying font path for draw_ocr method
# font_path = os.path.join('PaddleOCR', 'doc', 'fonts', 'latin.ttf')

# # %%
# # Import our image - drug 1/2/3
# # imports image
# img = cv2.imread(img_path)

# # reorders the color channels
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # %%
# # Visualize our image and detections
# # resizing display area
# plt.figure(figsize=(15, 15))

# # draw annotations on image
# annotated = draw_ocr(img, boxes, texts, scores, font_path=font_path)

# # show the image using matplotlib
# plt.imshow(annotated)

# # %%
# # Saving the image
# # filename = os.path.join('data','output')
# # reorderannotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
# # cv2.imwrite('data/output/filename.jpeg', reorderannotated )
# # cv2.imwrite('data/output/filename.jpeg', annotated)

# # %%
# img.shape

# # %%

# %%
