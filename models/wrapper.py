import jax
import re
import reformer
import os
import trax.fastmath.numpy as jnp
from cnn_classifier import PointNet 

class Model():
    def __init__(self, model_arch="pointnet", model_dir="", landmark_names_file="None", 
                track_errors=False, config=None):
        """ A wrapper for pretrained 3d point estimators 
        Args:
            model_arch: class of 3d point esitmator
            model_dir: directory containing (all) pretrained models
            landmark_names_file: text file of caesar landmark names
            track_errors: raise error for undiscovered pretrained model in model_dir
            config: model configuration : n_heads, n_enc_layers ...
        """
        self.model_dir = model_dir
        self.model_arch = model_arch
        self.landmark_names_file = landmark_names_file
        self.config = config 
        self.track_errors=track_errors

    def get_lnd_names(self):
        f = open(self.landmark_names_file, 'r')
        names = f.readlines()
        names = [x.strip() for x in names]
        f.close()
        return sorted(names)

    def normalize_name(self, name):
        norm_name = re.sub(r"[/#]", "", 
                            name.lower()).replace(" ", "_")
        return norm_name

    def load_model(self, landmark_names):
        """ Load model weights from the given directory
        Input:
        landmark_names - a list of pointnet models to load
        Returns: 
        model - dictionary of model_name and weight (pointnet);
                jit model function (reformer)
        """
        if self.model_arch == "pointnet":
            models = {}
            for name in landmark_names:
                ckpt_file = os.path.join(self.model_dir, name) + ".h5"
                existing_ckpt = os.path.exists(ckpt_file)
                if not existing_ckpt: 
                    if self.track_errors:
                        raise FileNotFoundError(f"ckpt file - {name} not found")
                    else:
                        continue
                net = cnn_classifier.PointNet(ckpt_file)
                models[name] = net
            print(models)
            return models

        elif self.model_arch == "reformer":
            model = reformer.build_model(self.config)
            model.init_from_file(
                file_name=os.path.join(self.model_dir, "*.pkl.gz"))
            model_fn = jax.jit(model)
            return model_fn
        return

    def _predict(self, xs):
        """ Get landmark estimates
        Inputs: 
            xs: A numpy array of a single body scan 
            landmark_names - a list of pointnet models to load
        Output: 
            landmarks - Dictionary of landmark objects: {name:xyz}
        """
        landmark_names = self.get_lnd_names()
        
        preds = {name:None for name in landmark_names}
        
        norm_names = {self.normalize_name(name):name for name in landmark_names}
        # load model
        model = self.load_model(norm_names.keys())
    
        if self.model_arch == "pointnet": 
            for name, net in model.items(): 
                pred_lnd = net._predict(xs)
                preds[norm_names[name]] = pred_lnd

        elif self.model_arch == "reformer":
            preds = model([jnp.array(xs)]) # all 101 landmarks
        return preds, landmark_names
      
