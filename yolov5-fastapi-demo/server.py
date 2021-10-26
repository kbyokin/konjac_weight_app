# from dextr.helpers.helpers import extreme_points
from importlib.util import decode_source
import json
from dextr.inference import dextr
import numpy as np
import seaborn
import pyzbar.pyzbar as pyzbar
import joblib
import random
import base64
import torch
import pandas as pd
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.pyplot import pie
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError, validator
from typing import List, Optional

import os
import sys
import time

sys.path.append("/home/kabin/web_konjac/yolov5-fastapi-demo/dextr")


app = FastAPI()
templates = Jinja2Templates(directory='templates')

# so we can read main.css
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

# for bbox plotting
colors = [tuple([random.randint(0, 255) for _ in range(3)])
          for _ in range(100)]

model_selection_options = ['yolov5s', 'yolov5m',
                           'yolov5l', 'yolov5x', 'konjac', 'custom']
model_dict = {model_name: None for model_name in model_selection_options}

model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='../yolov5_works/runs/exp10/weights/best.pt')
rdf_model = joblib.load('/home/kabin/web_konjac/random_forest/rdf.joblib')

@app.get("/")
def home(request: Request):
    '''
    Returns html jinja2 template render for home page form
    '''

    return templates.TemplateResponse('home.html', {
        "request": request,
        "model_selection_options": model_selection_options,
    })


def random_forest(xyxy, img, pixel_per_metric):
    width, height = img.size
    x = np.array(((xyxy[2] * width) / pixel_per_metric, (xyxy[3] * height) / pixel_per_metric)).reshape(1, -1)
    # print(x)
    result = rdf_model.predict(x)
    # print(f'result {result.tolist()[0]}')
    return result.tolist()[0]

def decode(im, ref_size):
    # Find barcodes and QR codes
    decodedObjects = pyzbar.decode(im)
    print(decodedObjects)
    # Print results
    if decodedObjects != []: 
        for obj in decodedObjects:
            data = obj.data
            w = obj.rect.width
            h = obj.rect.height
            pixel_per_metric = w / ref_size
            print(w/10, h/10)
            print('Type : ', obj.type)
            print('Data : ', obj.data, '\n')

        return decodedObjects, pixel_per_metric, data
    else:
        return []


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


def etpoints(xyxy):
    # width, height = img.size
    width, height = 640, 640

    # width, height = width/2, height/2
    # get point
    xyxy = [int(width * xyxy[0]),
            int(height * xyxy[1]),
            int(width * xyxy[2]),
            int(height * xyxy[3]),
            ]
    # print(xyxy)
    # * Return w, h value and w_, h_ temporary use for extreme points
    w, h = xyxy[2], xyxy[2]
    w_, h_ = w/2, h/2

    points = (xyxy[0]-w_, xyxy[1]), (xyxy[0], xyxy[1]+h_
                                     ), (xyxy[0]+w_, xyxy[1]), (xyxy[0], xyxy[1]-h_)
    # return (w, h), points
    return xyxy


def points_to_json(results, model):
    ''' Converts yolo model output to json (list of list of dicts)
    '''
    return [
        [
            {
                "class": int(pred[5]),
                "class_name": model.model.names[int(pred[5])],
                "points": etpoints(pred[:4].tolist()),
                "confidence": float(pred[4]),
            }
            for pred in result
        ]
        for result in results.xyxyn
    ]


def draw_qr_box(decodeObjects, img, color=(255, 99, 71), line_thickness=None):
    color = color or [random.randint(0, 255) for _ in range(3)]
    print(f'color {color}')
    draw = ImageDraw.Draw(img)
    for decodeObject in decodeObjects:
        xyxy = [int(decodeObject.rect.left), 
                int(decodeObject.rect.top), 
                int(decodeObject.rect.left + decodeObject.rect.height), 
                int(decodeObject.rect.top + decodeObject.rect.width)]
        draw.rectangle(xyxy, outline=color, width=2)


def plot_one_box(xyxy, img, color=(255, 255, 255), label=None, line_thickness=None):
    # function based on yolov5/utils/plots.py plot_one_box()
    # implemented using PIL instead of OpenCV to avoid converting between PIL and OpenCV formats

    color = color or [random.randint(0, 255) for _ in range(3)]
    width, height = img.size
    # print(f'image size {img.size}')

    xyxy = [int(width * xyxy[0]),
            int(height * xyxy[1]),
            int(width * xyxy[2]),
            int(height * xyxy[3]),
            ]

    draw = ImageDraw.Draw(img)
    draw.rectangle(xyxy, outline=color, width=3)

    if label:
        # drawing text in PIL is much harder than OpenCV due to needing ImageFont class
        # for some reason PIL doesn't have a default font that scales...
        try:
            # works on Windows
            fnt = ImageFont.truetype("arial.ttf", 36)
        except:
            '''
            linux might have issues with the above font, so adding this section to handle it
            this method is untested. based on:
            https://stackoverflow.com/questions/24085996/how-i-can-load-a-font-file-with-pil-imagefont-truetype-without-specifying-the-ab
            '''
            fnt = ImageFont.truetype(
                "/usr/share/fonts/truetype/freefont/FreeMono.ttf", 36, encoding="unic")

        txt_width, txt_height = fnt.getsize(label)

        draw.rectangle([xyxy[0], xyxy[1]-txt_height-2,
                       xyxy[0]+txt_width+2, xyxy[1]], fill=color)
        draw.text((xyxy[0], xyxy[1]-txt_height),
                  label, fill=(0, 0, 0), font=fnt)


class YOLORequest(BaseModel):
    ''' Class used for pydantic validation 
    Documentation: https://pydantic-docs.helpmanual.io/usage/validators/
    '''
    model_name: str
    img_size: int

    @validator('model_name')
    def validate_model_name(cls, v):
        assert v in model_selection_options, f'Invalid model name. Valid options: {model_selection_options}'
        return v

    @validator('img_size')
    def validate_img_size(cls, v):
        assert v % 32 == 0 and v > 0, f'Invalid inference size. Must be multiple of 32 and greater than 0.'
        return v


@app.post("/")
async def detect_via_web_form(request: Request,
                              file_list: List[UploadFile] = File(...),
                            #   model_name: str = Form(...),
                              img_size: int = Form(640),
                              ref_size: int = Form(10)):

    img_batch = []
    for i, file in enumerate(file_list):
        img = Image.open(BytesIO(await file.read()))
        img.filename = f'image{i}.jpg'
        print(f'image size {img.size}')
        img_batch.append(img)

    results_ = model(img_batch.copy(), size=img_size)
    json_results_ = results_to_json(results_, model)

    img_str_list = []
    weight = []
    # plot bboxes on the image
    for img, bbox_list in zip(img_batch, json_results_):
        # * Check weather algorithm can detect QR code
        check_qrcode = pyzbar.decode(img)
        print(f'QR code not None: {len(check_qrcode) != 0}')
        if len(check_qrcode) != 0:
            decodedObjects, pixel_per_metric, data = decode(img, ref_size)
            print(f'decoded data {decodedObjects}')
            draw_qr_box(decodeObjects=decodedObjects, img=img, line_thickness=3)
            for bbox in bbox_list:
                # label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
                plot_one_box(bbox['normalized_box'], img,
                            color=colors[int(bbox['class'])], line_thickness=3)
                weight.append(random_forest(bbox['normalized_box'], img, pixel_per_metric))

            # base64 encode the image so we can render it in HTML
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str_list.append(base64.b64encode(
                buffered.getvalue()).decode('utf-8'))

            dextr_point = []
            points = points_to_json(results_, model)
            dextr_point.append(points)

            # escape the apostrophes in the json string representation
            encoded_json_results = str(json_results_).replace(
                "'", r"\'").replace('"', r'\"')
            # encoded_json_results = str(dextr_point).replace(
            #     "'", r"\'").replace('"', r'\"')
            # json_to_csv = pd.DataFrame(dextr_point)
            # print(json_to_csv)

            print(f'pixel_per_metric:{pixel_per_metric}')
            print(f'weight {weight}')

            return templates.TemplateResponse('show_results.html', {
                'request': request,
                # zip here, instead of in jinja2 template
                # 'bbox_image_data_zipped': zip(img_str_list, points),
                'bbox_image_data_zipped': img_str_list,
                'bbox_data_str': encoded_json_results,
                'weight': weight,
                'min_weight': min(weight),
                'max_weight': max(weight),
                'container_name': str(data).replace("'", '').replace('b', '').upper()
            })
        else:
            for bbox in bbox_list:
                # label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
                plot_one_box(bbox['normalized_box'], img, color=colors[int(bbox['class'])], line_thickness=3)
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str_list.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
            return templates.TemplateResponse('show_results.html', {
                'request': request,
                # zip here, instead of in jinja2 template
                'bbox_image_data_zipped': img_str_list,
                'weight': weight,
                'bbox_data_str': None,
            })

@app.post("/detect_/")
async def detect_via_api(request: Request,
                         file_list: List[UploadFile] = File(...),
                         model_name: str = Form(...),
                         img_size: Optional[int] = Form(640),
                         download_image: Optional[bool] = Form(False)):

    '''
    Requires an image file upload, model name (ex. yolov5s). 
    Optional image size parameter (Default 640)
    Optional download_image parameter that includes base64 encoded image(s) with bbox's drawn in the json response

    Returns: JSON results of running YOLOv5 on the uploaded image. If download_image parameter is True, images with
                    bboxes drawn are base64 encoded and returned inside the json response.

    Intended for API usage.
    '''
    try:
        yr = YOLORequest(model_name=model_name, img_size=img_size)
    except ValidationError as e:
        return JSONResponse(content=e.errors(), status_code=422)

    if model_dict[model_name] is None:
        model_dict[model_name] = torch.hub.load(
            'ultralytics/yolov5', model_name, pretrained=True)

    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path='../yolov5_works/runs/exp10/weights/best.pt')

    img_batch = []
    dextr_point = []
    for i, file in enumerate(file_list):
        img = Image.open(BytesIO(await file.read()))
        # for https://github.com/WelkinU/yolov5-fastapi-demo/issues/5
        img.filename = f'image{i}.jpg'
        img_batch.append(img)

    if download_image:
        results = model_dict[model_name](img_batch.copy(), size=img_size)
        json_results = results_to_json(results, model_dict[model_name])

        results_ = model(img_batch.copy(), size=img_size)
        json_results_ = results_to_json(results_, model)

        for idx, (img, bbox_list) in enumerate(zip(img_batch, json_results_)):
            for bbox in bbox_list:
                label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
                plot_one_box(bbox['normalized_box'], img, label=label,
                             color=colors[int(bbox['class'])], line_thickness=3)

                # TODO: return extreme points
                points = etpoints(json_results_, img)
                dextr_point.append(points)

            # base64 encode the image so we can render it in HTML
            buffered = BytesIO()
            img.save(buffered, format="JPEG")

            payload = {'image_base64': base64.b64encode(buffered.getvalue()).decode('utf-8'),
                       'width': img.size[0],
                       'height': img.size[1]}
            json_results_[idx].append(payload)

    else:
        # if we're not downloading the image with bboxes drawn on it, don't do img_batch.copy()
        # results = model_dict[model_name](img_batch, size = img_size)
        # json_results = results_to_json(results,model_dict[model_name])
        results_ = model(img_batch.copy(), size=img_size)
        json_results_ = results_to_json(results_, model)

        for idx, (img, bbox_list) in enumerate(zip(img_batch, json_results_)):
            for bbox in bbox_list:
                points = etpoints(bbox['normalized_box'], img)
                dextr_point.append(points)

        # TODO: consider whether pass once or each bbox, put everything into JSON format
        start = time.time()
        print(
            "-----------------------------start to get mask-----------------------------")
        # dextr_ = np.array(dextr_point)
        # area_mask = dextr(dextr_, img)
        # print(area_mask)
        print(np.array(img).shape)
        # print(dextr_point)
        # print(type(np.array(dextr_point)))
        # print(dextr_.shape)
        end = time.time()
        print(
            f"-----------------------------total {end-start}s-----------------------------")

        # print(type(np.array(img)))
        # area = dextr(type(dextr_)point, img)
        # print(area)

    # return dextr_


@app.post("/detect/")
async def detect_via_api(request: Request,
                         file_list: List[UploadFile] = File(...),
                         model_name: str = Form(...),
                         img_size: Optional[int] = Form(640),
                         download_image: Optional[bool] = Form(False)):

    '''
    Requires an image file upload, model name (ex. yolov5s). 
    Optional image size parameter (Default 640)
    Optional download_image parameter that includes base64 encoded image(s) with bbox's drawn in the json response

    Returns: JSON results of running YOLOv5 on the uploaded image. If download_image parameter is True, images with
                    bboxes drawn are base64 encoded and returned inside the json response.

    Intended for API usage.
    '''
    # try:
    #     yr = YOLORequest(model_name=model_name, img_size=img_size)
    # except ValidationError as e:
    #     return JSONResponse(content=e.errors(), status_code=422)

    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path='../yolov5_works/runs/exp10/weights/best.pt')

    img_batch = []
    dextr_point = []

    for i, file in enumerate(file_list):
        img = Image.open(BytesIO(await file.read()))
        # for https://github.com/WelkinU/yolov5-fastapi-demo/issues/5
        img.filename = f'image{i}.jpg'
        img_batch.append(img)

    if download_image:
        results = model(img_batch.copy(), size=img_size)
        json_results = results_to_json(results, model)

        for idx, (img, bbox_list) in enumerate(zip(img_batch, json_results)):
            for bbox in bbox_list:
                label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
                plot_one_box(bbox['normalized_box'], img, label=label,
                             color=colors[int(bbox['class'])], line_thickness=3)

                # TODO: return extreme points
                points = etpoints(json_results, img)
                dextr_point.append(points)

            # base64 encode the image so we can render it in HTML
            buffered = BytesIO()
            img.save(buffered, format="JPEG")

            payload = {'image_base64': base64.b64encode(buffered.getvalue()).decode('utf-8'),
                       'width': img.size[0],
                       'height': img.size[1]}
            json_results[idx].append(payload)

    else:
        # if we're not downloading the image with bboxes drawn on it, don't do img_batch.copy()
        # results = model_dict[model_name](img_batch, size = img_size)
        # json_results = results_to_json(results,model_dict[model_name])
        results = model(img_batch.copy(), size=img_size)
        json_results = results_to_json(results, model)

        for idx, (img, bbox_list) in enumerate(zip(img_batch, json_results)):
            for bbox in bbox_list:
                # TODO: compute width, height, extreme points
                # ? seperate width, height
                points = etpoints(bbox['normalized_box'])
                dextr_point.append(points)

        start = time.time()
        print(
            "-----------------------------start to get mask-----------------------------")
        # TODO: img shape (640, 640), now is (3024, 4032)
        print(np.array(img).shape)

        # TODO: consider whether pass once or each bbox, put everything into JSON format
        dextr_ = np.array(dextr_point, dtype="object")
        # dextr_ = np.around(dextr_)
        print(dextr_[1])
        print(dextr_.shape)
        # area_mask = dextr(dextr_, img)
        # print(area_mask)

        # TODO: compute weight
        end = time.time()
        print(
            f"-----------------------------total {end-start}s-----------------------------")

        print("img_batch", img_batch)

    # ?: what to return --> weight, mask
    # return dextr_


@app.get("/about/")
def about_us(request: Request):
    '''
    Display about us page
    '''

    return templates.TemplateResponse('about.html',
                                      {"request": request})


if __name__ == '__main__':
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--precache-models', action='store_true',
                        help='pre-cache all models in memory upon initialization')
    opt = parser.parse_args()

    if opt.precache_models:
        # pre-load models
        model_dict = {model_name: torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
                      for model_name in model_selection_options}

    # make the app string equal to whatever the name of this file is
    app_str = 'server:app'
    uvicorn.run(app_str, host='localhost', port=8000, reload=True, workers=1)
