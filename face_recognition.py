from calendar import EPOCH
import tensorflow as tf 
import tensorflow.keras.backend as k
import tensorflow.keras as keras 
from tensorflow.keras.losses import BinaryCrossentropy
from detect_faces import load_model, close_cam, predict_image, filter_predictions, display_pred_in_cam
from fr_utils import create_user, DataBase, info, read_img, ImgDataset, merge_dfs, generate_match_pairs, generate_mismatch_pairs, alb_transform
import os
import numpy as np 
import cv2 as cv
from datetime import datetime 
from time import time 

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
                    verification_images = [read_img(os.path.join(path, f'{name}_{ind}.jpg')) for ind in range(len(os.listdir(path)))]
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

def get_dataset(img_path):
    pairs  = generate_match_pairs(img_path)
    mpairs = generate_mismatch_pairs(img_path, pairs.shape[0])
    df = merge_dfs([pairs, mpairs], 0)
    df[['imgnum1', 'imgnum2', 'match']] = df[['imgnum1', 'imgnum2', 'match']].astype(str)
    batch = 4
    SIZE = 128
    
    train_data = ImgDataset(df, img_path, batch, img_size = (SIZE, SIZE), test_size = 0.0, 
                                seed = 123, transform = alb_transform) 
    train_data = train_data.load_dataset_V2()
    train_data = tf.data.Dataset.zip(train_data).prefetch(1)
    return train_data

@tf.function
def train_on_batch(batch, model, opt, loss_func, verbose = True):
    with tf.GradientTape() as tape: 
        X = batch[:2]
        y = batch[2]
        yhat = model(X, training = True)
        loss = loss_func(y, yhat)
    
    del X, y, yhat
    print(f'Loss : {loss}')
    grad = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grad, model.trainable_variables))
    
    return loss

def train_model(data, model, opt, loss_func, epochs = 5, save_model = 0, verbose = True):
    history = {
        'loss' : [],
        'accuracy' : []
    }
    for epoch in range(1, epochs + 1):
        print(f'Epoch : {epoch} / {epochs} ', end = ' ')
        progbar = tf.keras.utils.Progbar(len(data))
        l = []
        for idx, batch in enumerate(data):
            loss = train_on_batch(batch, model, opt, loss_func, verbose = verbose)
            progbar.update(idx + 1)
            l.append(loss)
            
        if save_model > 0:
            if epoch % save_model == 0:
                checkpoint(model)
                
        history['loss'].append(np.mean(l))
        del l
        
    return model, opt, history

def checkpoint(model, name = 'siamese_model.h5', remove_prev_model = False):
    path = os.path.join('weights', name)
    mname = os.path.basename(path)
    if mname in os.listdir('weights'):
        if remove_prev_model:
            os.unlink(path)
        else:
            name, ftype = mname.split('.')
            num = len(os.listdir('weights')) - 1
            mname = f'{name}{num}.{ftype}'
            path = path = os.path.join('weights', mname)
    model.save(path)

def main():
    info(
    '''
    (1) - create a new user 
    (2) - start live demo  
    (3) - update the model            
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
    elif option == 3:
        EPOCHS = 2
        reg_model = load_recog_model('siamese_model.h5')
        train_data = get_dataset('images')
        train_model(train_data, reg_model, keras.optimizers.Adam(learning_rate = 0.001), BinaryCrossentropy(), EPOCHS, 2)
        info('siamese model has finished training')
    else:
        exit()
        

if __name__ == "__main__":
    main()

     


    