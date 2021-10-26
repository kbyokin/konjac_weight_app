from yolov5_works.inference import detect

def get_yolo(image):
    conf_thres = 0.90
    iou_thres = 0.45
    # image = "./dataset/konjac/images/IMG_8863.jpg" # change to streamlit later
    xyxy = detect(out="./yolov5_works/inference/outputs", 
                    source=image, 
                    weights="./yolov5_works/runs/exp10/weights/best.pt", 
                    imgsz=640, view_img=False, 
                    save_txt=False, 
                    conf_thres=conf_thres, 
                    iou_thres=iou_thres)

pred = get_yolo()