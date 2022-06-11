import cv2 as cv 
import os 
from datetime import datetime
import numpy as np 
from detect_faces import predict_images, filter_predictions, close_cam
from crop_images import crop_img
from pymongo import MongoClient 
from bson.objectid import ObjectId
import matplotlib.pyplot as plt 
import pandas as pd 
import tensorflow as tf
import albumentations as A

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

def save_image(image, path, fname):
    path = os.path.join(path, f'{fname}.jpg')
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
                save_image(image, path, f'{name}_{ind}')
            else:
                processed_images.append(image.tolist())
        
    if not store_locally:
        db = DataBase(None)
        db.upload_data(name, processed_images)
        
    info(f'Saved {ind + 1} images to {path if store_locally else "the DB"}')
                
        
    print(f'Successfully created a user : {name} ...')
    
    
def generate_match_pairs(images_path):
    match_pairs = []
    names = os.listdir(images_path)
    for name in names:
        path = os.path.join(images_path, name)
        images = os.listdir(path)
        leng = len(images)
        for num1 in range(leng):
            for num2 in range(leng):
                match_pairs.append([name, num1, num2, name, '1'])
                
    return pd.DataFrame(match_pairs, columns = ['name1', 'imgnum1', 'imgnum2', 'name2', 'match']).sample(frac = 1.0).reset_index(drop = True)

def generate_mismatch_pairs(images_path, total_mismatches = 6000):
    mismatch_pairs = []
    names          = {}
    
    for name in os.listdir(images_path):
        num_images = len(os.path.join(images_path, name))
        names[name] = num_images
    
    for name1 in names.keys():
        for name2 in names.keys():
            if name1 != name2:
                for num1 in range(names[name1]):
                    for num2 in range(names[name2]):
                        mismatch_pairs.append([name1, num1, num2, name2, '0'])
    
    data = pd.DataFrame(mismatch_pairs, columns = ['name1', 'imgnum1', 'name2', 'imgnum2', 'match'])
    if data.shape[0] > 0:
        data = data.sample(n = total_mismatches).reset_index(drop = True)        
    return data
        
        
def merge_dfs(dfs : list, axis : int = 0):
    return pd.concat(dfs, axis = axis)


class GenerateIdx:
    def __init__(self, n, test_size, seed):
        np.random.seed(seed)
        idx = [i for i in range(n)]
        train_size = int(n - (n * test_size))
        self.train_idx = np.random.choice(idx, train_size, replace = False).tolist()
        self.test_idx  = [i for i in idx if i not in self.train_idx]

class ImgDataset:
    def __init__(self, df, img_path, batch = 16, img_size = (64, 64), transform = True, subset = 'train', 
                 test_size = 0.2, seed = 123, shuffle = True, sanity_check = False):
        self.data = df.copy()
        if shuffle:
            self.data = self.data.sample(frac = 1.0).reset_index(drop = True)
        idx = GenerateIdx(self.data.shape[0], test_size, seed)
        if subset == 'train':
            self.data = self.data.iloc[idx.train_idx, :]
        elif subset == 'validation':
            self.data = self.data.iloc[idx.test_idx, :]
            
        if sanity_check:
            self.data = self.data.head()
            
        self.img_path = img_path 
        self.batch = batch
        self.transform = transform 
        self.img_size = img_size
    
    def load_dataset_V2(self):
        images1 = self.data[['name1', 'imgnum1']].values
        images2 = self.data[['name2', 'imgnum2']].values
        targets = self.data['match'].astype(float).values
        
        print(f'Length of anchor images       : {images1.shape[0]}')
        print(f'Length of verification images : {images2.shape[0]}')
        print(f'Length of targets images      : {targets.shape[0]}')
        
        images1 = tf.data.Dataset.from_tensor_slices(images1)
        images2 = tf.data.Dataset.from_tensor_slices(images2)
        targets = tf.data.Dataset.from_tensor_slices(targets).cache().batch(self.batch)
        
        if self.transform:
            images1 = (images1
                        .map(self.tf_read_img)
                        .map(self.tf_augment)
                        .map(self.normalize).cache().batch(self.batch)
                      )
            images2 = (images2
                        .map(self.tf_read_img)
                        .map(self.tf_augment)
                        .map(self.normalize).cache().batch(self.batch)
                      )
        else:
            images1 = (images1
                        .map(self.tf_read_img)
                        .map(self.normalize).cache().batch(self.batch)
                      )
            images2 = (images2
                        .map(self.tf_read_img)
                        .map(self.normalize).cache().batch(self.batch)
                      )
            
        return images1, images2, targets
            
    def read_preprocess(self, img):
        image = read_img(img)
        image = cv.resize(image, self.img_size)
        channel = image.shape[-1]
        if channel == 1:
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        elif channel == 4:
            image = cv.cvtColor(image, cv.COLOR_RGBA2RGB)
            
        return image
    
    def read_images(self, data):
        name, num = data
        path  = self.join_paths(self.img_path, name.decode('utf-8'), num.decode('utf-8'))
        image = self.read_preprocess(path)
        return image
    
    def tf_read_img(self, data):
        images = tf.numpy_function(func = self.read_images, inp = [data], Tout = tf.uint8)
        return images
    
    def tf_augment(self, image):
        aug_img = tf.numpy_function(func = self.transform, inp = [image], Tout = tf.uint8)
        return aug_img
    
    def tf_crop_imgs(self, image):
        crop_img = tf.numpy_function(func = self.crop_faces, inp = [image], Tout = tf.uint8)
        return crop_img     
            
    def __len__(self):
        return self.data.shape[0]
    
    @staticmethod
    def get_image_path(rdf, base_path):
        name1 = rdf['name1']
        name2 = rdf['name2']
        num1  = str(rdf['imgnum1'])
        num2  = str(rdf['imgnum2'])
        
        return (
                  self.join_paths(base_path, name1, num1),
                  self.join_pathsths(base_path, name2, num2)
                )
    
    @staticmethod
    def join_paths(base_path, name, num):
        return os.path.join(base_path, name, f"{name}_{num}.jpg")
    
    @staticmethod
    def normalize(x):
        img = x / 255
        return img
    
def alb_transform(xs):
    transform = A.Compose([
        A.HorizontalFlip(0.5),
        A.Rotate(limit = 35, border_mode = cv.BORDER_CONSTANT),
        A.GaussianBlur(blur_limit = (3, 3)),
        A.CoarseDropout(10)
    ])
    
    tx = transform(image = xs)['image']
    return tx

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
        