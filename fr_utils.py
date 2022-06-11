import cv2 as cv 
import os 
from datetime import datetime
import numpy as np 
from detect_faces import predict_images, filter_predictions, close_cam
from crop_images import crop_img
from pymongo import MongoClient 
from bson.objectid import ObjectId
import matplotlib.pyplot as plt 

def info(i):
    star = '*' * (len(i) + 4)
    print(f'''
          {star}
           {i}
          {star}
          ''')
    
def read_img(path):
    return np.array(cv.imread(path))
    
def capture_images(vc = 0, n = 2):
    cap = cv.VideoCapture(vc, cv.CAP_DSHOW)
    st = datetime.now().second
    images = []
    info
    while cap.isOpened():
        _, frame = cap.read()
        image = np.array(frame)
        images.append(image)
        ed = datetime.now().second
        if n <= abs(ed - st):
            break
        
        if cv.waitKey(100) == ord('q'):
            close_cam(cap)
            break
        
    return images

def save_image(image, path, ind):
    path = os.path.join(path, f'{ind}.jpg')
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    cv.imwrite(path, image)
       
def create_user(od_model, cli = True, store_locally = True):
    if not cli:
        return 
    name = input('Name* : ')
    info('Make sure the providing images contains only one faces')
    cap_img_now = input('Take live images now* (y/n) : ')
    folder = 'images'
    path = os.path.join(folder, name)
    if name not in os.listdir(folder):
        os.makedirs(path)
        
        
    processed_images = []
    if cap_img_now[0].lower() == 'y':
        images = capture_images()  
        info(f'collected {len(images)} images')
    else:
        info(f'please upload more than 5 images to {path}')
        images = [plt.imread(path, img) for img in os.listdir(path)]
        info(f'found {len(images)} in {path}')
    
    info('Processing the images, please wait for a moment')
    predictions  = predict_images(od_model, images, return_bbox = True, bbox_format = 'xyxy', normalize = False, verbose = True)
    for ind, image in enumerate(images):
        fpredictions = filter_predictions(predictions[ind], 0.6)[:1]
        if len(fpredictions) > 0:   
            image = crop_img(image, fpredictions, save_imgs = False)
            if store_locally:
                save_image(image, path, ind)
            else:
                processed_images.append(image.tolist())
        
    if not store_locally:
        db = DataBase(None)
        db.upload_data(name, processed_images)
        
    info(f'Saved {ind + 1} images to {path if store_locally else "the DB"}')
                
        
    print(f'Successfully created a user : {name} ...')
           
        
class DataBase:
    def __init__(self, uri):
        uri = 'mongodb+srv://cisasfddb:facedetection123@cluster0.83yabke.mongodb.net/?retryWrites=true&w=majority'
        self.cluster = MongoClient(uri)
        self.db = self.cluster['face_reg']
        self.collection = self.db['users']
    
    def upload_data(self, name, images, **kwargs):
        self.collection.insert_one({'name' : name,  'images' : images, **kwargs})
        
    def change_collection(self, collection):
        self.collection = collection
        
    def retrieve_data(self, name = None):
        if not name:
            result = self.collection.find({})
        else:
            result = self.collection.find({'name' : name})
        return result
        