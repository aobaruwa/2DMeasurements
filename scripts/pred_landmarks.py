from landmark import utils
from landmark import cnn_classifier
from tqdm import tqdm
import numpy as np
import os
import pickle
import tensorflow as tf

def main(obj_file, xml_fname, model_dir):
    landmarks = {}
    all_models = [os.path.join(model_dir,x) for x in os.listdir(model_dir) if x.endswith('.h5') and 'test' not in x]
    # convert obj_file to pc
    pc = utils.read_obj(obj_file)
    pc,offsets= utils.align_point_cloud(pc)
    for model_file in tqdm(all_models):
        lndmrk_name = os.path.basename(model_file)[:-3]
        landmarks[lndmrk_name] = pred_landmark(model_file, np.expand_dims(pc, axis=0))
    utils.write_landmark_xml(xml_fname, landmarks)

    return 

def pred_landmark(model_file, pc):
    """Predict a single landmark on the pointcloud from one 
       single model. """
    model=cnn_classifier.build_model(pc.shape[1],-999)
    model.compile(loss='mse',optimizer='adam')

    model.load_weights(model_file)
    predictions=model.predict(pc).squeeze() 
    return predictions

if __name__ == "__main__":
    model_dir = '/projects/datascience/shared/models'
    scan_dir = '/projects/datascience/shared/Reference scans (Susan)'
    obj_file = os.path.join(scan_dir, 'FL071819F04_B1_LINED.obj')
    xml_fname =  '/projects/datascience/jsearcy/landmark/scripts/ml_pred_FL071819F04_B1_LINED.xml'
    main(obj_file, xml_fname, model_dir)
