import streamlit as st
import requests

import os
import io
import base64
from base64 import decodebytes
from io import BytesIO

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



def results_to_json(results, model):
    ''' Converts yolo model output to json (list of list of dicts)
    '''
    return [
        [
            {
                "class": int(pred[5]),
                "class_name": model.model.names[int(pred[5])],
                "normalized_box": pred[:4].tolist(),
                "confidence": float(pred[4]),
            }
            for pred in result
        ]
        for result in results.xyxyn
    ]


def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


def yolo_detect(image):
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path='../yolov5_works/runs/exp10/weights/best.pt')
    results = model(image.copy(), size=640)
    json_result = results_to_json(results, model)
    return json_result

##########
##### Set up sidebar.
##########

# Add in location to select image.

st.sidebar.write('#### Select an image to upload.')
uploaded_file = st.sidebar.file_uploader('',
                                         type=['png', 'jpg', 'jpeg'],
                                         accept_multiple_files=False)

##########
##### Set up main app.
##########

## Title.
st.write('# Konjac Weight Estimation')

## Pull in default image or user-selected image.
if uploaded_file is None:
    # Default image.
    url = 'https://github.com/matthewbrems/streamlit-bccd/blob/master/BCCD_sample_images/BloodImage_00038_jpg.rf.6551ec67098bc650dd650def4e8a8e98.jpg?raw=true'
    image = Image.open(requests.get(url, stream=True).raw)

else:
    # User-selected image.
    image = Image.open(uploaded_file)
    json_result = yolo_detect(image)

## Subtitle.
st.write('### Inferenced Image')

# Convert to JPEG Buffer.
buffered = io.BytesIO()
image.save(buffered, quality=90, format='JPEG')

# Base 64 encode.
img_str = base64.b64encode(buffered.getvalue())
img_str = img_str.decode('ascii')

# Convert to JPEG Buffer.
buffered = io.BytesIO()
image.save(buffered, quality=90, format='JPEG')

# Display image.
st.image(image,
         use_column_width=True)

## Summary statistics section in main app.
st.write('### Summary Statistics')
# st.write(
#     f'Number of Bounding Boxes (ignoring overlap thresholds): {len(confidences)}')
# st.write(
#     f'Average Confidence Level of Bounding Boxes: {(np.round(np.mean(confidences),4))}')

## Histogram in main app.
st.write('### Histogram of Confidence Levels')
# fig, ax = plt.subplots()
# ax.hist(confidences, bins=10, range=(0.0, 1.0))
# st.pyplot(fig)

## Display the JSON in main app.
st.write('### JSON Output')
if json_result != None:
    st.write(json_result[0])
else:
    st.write('### JSON is none')
