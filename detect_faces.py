import torch 
import argparse 
import os 
import numpy as np 
import cv2 as cv
import albumentations as A
from time import time
from crop_images import crop_img, wait_till_specified_time
from datetime import datetime


CONFIG = {
    'video_cap'       : 0,
    'score_threshold' : .7,
    'device'          : torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'time_to_wait'    : 10,
    'curr_time'       : datetime.now()
}

def parsed_args():
    parser = argparse.ArgumentParser(description = 'Video capturing and hyper-parameter tuning')
    parser.add_argument('--vc', type = int,    default = str(CONFIG['video_cap']),   help = f"cv.videoCapture : 0, 1 or video path, ( default : {CONFIG['video_cap']} )")
    parser.add_argument('--s',  type = float,  default = CONFIG['score_threshold'],   help = f"score threshold, should be in range 0.0 - 1.0, ( default : {CONFIG['score_threshold']} )")
    parser.add_argument('--d',  type = str,    default = CONFIG['device'],   help = f"device to train, [cpu, cuda], ( default : {CONFIG['device']} )")
    parser.add_argument('--t',  type = float,  default = CONFIG['time_to_wait'],   help = f"time to wait after capturing one frame in ms, ( default : {CONFIG['time_to_wait']} )")
    parser.add_argument('--interval', type = int, default = 0, help = 'sleep for specified time in min')
    parser.add_argument('--verbose', const = True, action = 'store_const', help = "if verbose , print out the predicted logs")
    parser.add_argument('--crop-images', const = True, action = 'store_const', help = 'if true, crop the detected faces into images dir' )
    parser.add_argument('--show-label', const = True, action = 'store_const', help = 'display the respective class label above the bounding box')
    
    return parser.parse_args()


def load_model(mname, device):
    path = os.path.join('weights', mname)
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path = path, device = device)
        model.eval()
    except FileNotFoundError as err:
        print(err)
    return model


def resize_img_bbox(x, target, orig_size, bbox_format = 'pascal_voc'):
    transform = A.Compose([
        A.Resize(*orig_size)
    ], bbox_params = A.BboxParams(format = bbox_format, label_fields = 'class_labels'))
    
    
    bboxes = [t[:4] for t in target]
    scores = [t[-2].item() for t in target]
    class_labels = [int(t[-1].item()) for t in target]
    print(class_labels)
    timg = transform(image = x, bboxes = bboxes, class_labels = class_labels)
    
    target = []
    for i in range(len(timg['bboxes'])):
        target.append([
            *timg['bboxes'][i], scores[i], timg['class_labels'][i] 
        ])
        
    return timg['image'], target


def predict_image(model, img, return_bbox = False, bbox_format = 'xywh', normalize = False, verbose = False):
    if normalize:
        img = img / 255
    res = model([img])
    if verbose:
        res.print()
    if return_bbox:
        if bbox_format == 'xywh':
            return res.xywh
        else:
            return res.xyxy
    return res.render()


def predict_images(model, imgs, return_bbox = False, bbox_format = 'xywh', normalize = False, verbose = False):
    if normalize:
        imgs = [ img / 255 for img in imgs ]
    res = model(imgs)
    if verbose:
        res.print()
    if return_bbox:
        if bbox_format == 'xywh':
            return res.xywh
        else:
            return res.xyxy
    return res.render()


def filter_predictions(predictions, score_threshold):
    new_predictions = []
    for pred in predictions:
        if pred[4] > score_threshold:
            new_predictions.append(pred)

    return new_predictions

def close_cam(cap):
    cap.release()
    cv.destroyAllWindows()
    
def display_pred_in_cam(cap, image, target, transformed = False, fps = 0, color = (0, 0, 255), show_label = False):
    if target:
        cv.putText(image, f'fps : {round(fps, 1)}', (int(10), int(20) + 1), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv.LINE_AA)
        
        label_enc = {
            1 : ['human face', color],   
        }
        
        boxes  = [t[:4] for t in target]
        if not transformed:
            scores = [t[-2].item() for t in target]
            labels = [int(t[-1].item()) for t in target]
        else:
            scores = [t[-2] for t in target]
            labels = [t[-1] for t in target]
        for ind in range(len(labels)):
            x1,y1, x2,y2 = boxes[ind]
            label, color = label_enc[labels[ind]]
            cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
            if show_label:
                cv.putText(image, label, (int(x1), int(y1)), cv.FONT_HERSHEY_PLAIN, 1, color, 1, cv.LINE_AA)
            
    cv.imshow('Face Detection', image)

def main():
    global args
    
    args = parsed_args()
    device = CONFIG['device']
    model = load_model('best_ep2.pt', device)
    score_threshold = args.s
    vc = args.vc
        
    cap = cv.VideoCapture(vc)
    old_fps, new_fps = 0, 0
    
    CONFIG['curr_time'] = datetime.now()
    while cap.isOpened():
        
        _, frame = cap.read()
        orig_img = np.array(frame)
        
        new_fps = time()
        predictions  = predict_image(model, orig_img, return_bbox = True, bbox_format = 'xyxy', verbose = args.verbose)[0]
        fpredictions = filter_predictions(predictions, score_threshold)
    
        
        fps  = 1/(new_fps - old_fps)
        old_fps = new_fps
        display_pred_in_cam(cap, orig_img, fpredictions, fps = fps, show_label = args.show_label)
        
        
        if args.crop_images:
            if wait_till_specified_time(CONFIG['curr_time'], args.interval, args):
                crop_img(orig_img, fpredictions, args = args)
                CONFIG['curr_time'] = datetime.now()
        
        if cv.waitKey(args.t) == ord('q'):
            close_cam(cap)
            break
        
if __name__ == '__main__':
    if 'weights' not in os.listdir():
        os.mkdir('weights')
    main()

