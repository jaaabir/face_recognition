import os 
import cv2 as cv
from uuid import uuid4
from datetime import datetime, timedelta



def crop_img(img, targets, base_path = 'images', ftype = 'png', args = None, save_imgs = True):
    if base_path not in os.listdir():
        os.mkdir(base_path)
        
    images = []
    for ind, target in enumerate(targets):
        x, y, x2, y2 = list(map(int, target[:4]))
        cimg = img[ y: y2, x: x2 ]
        if save_imgs:
            fname = os.path.join(base_path, f'{uuid4().hex}.{ftype}')
            cv.imwrite(fname, cimg)
        else:
            images.append(cimg)
    
    if args:   
        if args.verbose:
            print(f'Cropped and saved {ind + 1} face images ...')
            
    if not save_imgs:
        return images[0]
        
        
def wait_till_specified_time(curr_time, wait_min, args = None):
    incre_min = timedelta(minutes = wait_min)
    new_time = curr_time + incre_min    
    if args:
        if args.verbose:
            print(f'Waiting till : {new_time.time()}, current time : {datetime.now().time()}')
    return True if datetime.now() >= new_time else False 
    
        
