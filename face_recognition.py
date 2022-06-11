from matplotlib import pyplot
import tensorflow as tf 
import tensorflow.keras.backend as k
import tensorflow.keras as keras 
from tensorflow.keras.losses import BinaryCrossentropy
from detect_faces import load_model, close_cam, predict_image, filter_predictions, display_pred_in_cam
from fr_utils import create_user, DataBase, info, read_img
import os
import numpy as np 
import cv2 as cv
from datetime import date, datetime 
from time import time 
import matplotlib.pyplot as plt 

class Distance(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        
    def euclidean_dist(self, vectors):
        vec1, vec2 = vectors
        dist = k.sum(k.square(vec1 - vec2), axis = 1, keepdims = True)
        return k.sqrt(k.maximum(dist, k.epsilon()))
    
    def call(self, anchor, verification):
        return self.euclidean_dist([anchor, verification])
    
def load_recog_model(mname):
    path = os.path.join('weights', mname)
    model = keras.models.load_model(path, custom_objects = {'Distance' : Distance, 'BinaryCrossEntropy' : BinaryCrossentropy})
    return model

def predict_on_batch(model, x, N, thresh):
    result = []
    for batch in x:
        pred = model.predict(batch)
        preds = (pred > 0.5).astype(int).tolist() 
        result += preds 
        print(pred)
    
    result = np.array(result)
    result = (np.sum(result) / N) >= thresh
    return result

def verify_face(model, anchor, verifications, N = 10, thresh = 0.5, normalize = False, img_size = (128, 128)) -> bool:
    len_verf = len(verifications)
    N = min(N, len_verf)
    anchors = [cv.resize(anchor, img_size) for _ in range(N)]
    verifications = [cv.resize(verification, img_size) for verification in verifications[:N]]
    if normalize:
        anchors = [a / 255 for a in anchors]
        verifications = [v / 255 for v in verifications]
    
    anchors = tf.data.Dataset.from_tensor_slices(anchors).batch(N // 2)
    verifications = tf.data.Dataset.from_tensor_slices(verifications).batch(N // 2)
    x = tf.data.Dataset.zip((anchors, verifications)).as_numpy_iterator()
    result = predict_on_batch(model, x, N, thresh)   
    return result
    

def get_persons_images():
    db = DataBase(None)
    users = db.retrieve_data()
    return users

def get_verification_images(user):
    return user.name, user.images 

def live_demo(od_model, reg_model, vc = 0, store_locally = True, show_live_images = True, verbose = True):
    cap = cv.VideoCapture(vc, cv.CAP_DSHOW)
    score_threshold = 0.6
    old_fps, new_fps = 0, 0
    while cap.isOpened():
        _, frame = cap.read()
        anchor_image = np.array(frame)
        nt = datetime.now()
        new_fps = time()
        predictions  = predict_image(od_model, anchor_image, return_bbox = True, bbox_format = 'xyxy', verbose = verbose)[0]
        fpredictions = filter_predictions(predictions, score_threshold)[:1]
        fps  = 1/(new_fps - old_fps)
        old_fps = new_fps
        authorize_person = False
        verified_name = None
        if len(fpredictions) > 0:
            if store_locally:
                names = os.listdir('images')
                for ind, name in enumerate(names):
                    path = os.path.join('images', name)
                    verification_images = [read_img(os.path.join(path, f'{ind}.jpg')) for ind in range(len(os.listdir(path)))]
                    authenticate = verify_face(reg_model, anchor_image, verification_images, normalize = True)
                    if authenticate:
                        authorize_person = True
                        verified_name = name
                        break 
            else:
                db = DataBase(None)
                for docs in db.retrieve_data():
                    name = docs.name
                    verification_images = np.array(docs.images)
                    authenticate = verify_face(reg_model, anchor_image, verification_images, normalize = True)
                    if authenticate:
                        authorize_person = True
                        verified_name = name
                        break 
                    
            if authorize_person:
                info(f'name : {verified_name} , detected ...')
            else:
                info(f'Unknown person ... , please contact the registration ...')
            
        if show_live_images:
            color = (0, 0, 255) if not authorize_person else (0, 255, 0)
            display_pred_in_cam(None, anchor_image, fpredictions, color = color, fps=fps)
            
        
        if cv.waitKey(10) == ord('q'):
            close_cam(cap)
            break

def main():
    # reg_model = load_recog_model('siamese_model.h5')
    info(
    '''
    (1) - create a new user 
    (2) - start live demo             
    ''')
    option = int(input('>>> '))
    store_locally = True
    if option == 1:
        od_model  = load_model('best_ep2.pt', 'cpu')
        create_user(od_model, store_locally = store_locally)
    elif option == 2:
        od_model = load_model('best_ep2.pt', 'cpu')
        reg_model = load_recog_model('siamese_model.h5')
        live_demo(od_model, reg_model, vc = 0, store_locally = store_locally, show_live_images = True, verbose = True)

if __name__ == "__main__":
    main()

     


    