from glob import glob
from tqdm import tqdm
import numpy as np
import os
import pickle
import random
import re


class Dataset():
    """Set of helpful utilities to build a pointcloud 
    body scan dataset for modeling."""
    def __init__(self, name="CAESAR"):

        self.dataset_name = name
        self.pc_data = None
        self.data_dir = None
        self.landmark_names = set()
        self.max_pcsize = None

    def build(self, data_dir):
        """Create dataset from a point cloud directory containing
        pointclouds and their landmarks"""
        self.pc_data = []
        self.max_pcsize = 0 
        self.data_dir = data_dir
        pc_files = glob(self.data_dir, '**')
        for fname in pc_files:
            obj = pickle.load(open(fname, "rb"))
            key =list(obj.keys())[0]
            landmarks = {}
            for landmark in obj[key]['LM'].items():
                name, xyz = landmark[0]
                name = re.sub(r"[/#]", "", name) # remove forward slash and pound with 
                landmarks[name] = xyz
                data_point = {'PC':obj[key]['PC'], 'LM': landmarks}
                # add  to the list of landmark names
                self.landmark_names.add(name)
            self.pc_data.append(data_point)
            self.max_pcsize = max(self.max_pcsize, data_point['PC'].shape[0])
            xs, ys, tokens = self.preprocess(self.pc_data, save=False)
        return xs, ys, tokens
    
    def __len__(self):
       return len(self.pc_data)
    
    def preprocess(self, pc_data, size = 256000, save=False):
        xs = []; ys=[]; tokens=[]

        for dd in tqdm(pc_data): 
            size=256000

            if len(dd['PC']) < size:
                select=np.random.choice(range(len(dd['PC'])),len(dd['PC']),replace=False)
                xs.append(np.expand_dims(np.pad(dd['PC'][select],[[0,size-len(dd['PC'])],[0,0]]),0))
            else:
                select=np.random.choice(range(size),size,replace=False)
                xs.append(np.expand_dims(dd['PC'][select],0))

            lndmks = []
            token=[]
            for i,l in enumerate(self.landmark_names):
                if l in dd['LM']:
                    lndmks.append(dd['LM'])
                    token.append(i+1)
            ys.append(np.expand_dims(np.pad(lndmks,[[0,128-len(lndmks)],[0,0]] ),0))
            tokens.append(np.expand_dims(np.pad(token,[[0,128-len(token)]]),0))
        
        xs=np.concatenate(xs,axis=0)
        ys=np.concatenate(ys,axis=0)
        tokens=np.concatenate(tokens,axis=0)

        if save:
            np.save(os.path.join(self.data_dir, 'pc_data.npy'), list(zip(xs,tokens,ys)))
        return xs, ys, tokens

    def split(self, x, part=0.8, shuffle=True):
        
        if shuffle:
            random.shuffle(x)
        size = len(x)
        train = x[:size*part]
        val = x[size*part:]
        return train, val
    
    def load(self, data_dir):
        """Load dataset from a serialized numpy file"""
        data_file = glob(data_dir, '*.npy')
        assert os.path.exists(data_file), "no numpy files exists in not in your directory"
        xs, tokens, ys = np.load(data_file, allow_pickle=True)
        return xs, tokens, ys
    
    def save(self, out_file):
        """ save point cloud data to file as numpy dump """
        np.save(self.pc_data, out_file)

    def get_landmark_names(self):
        """List out all the landmark names of the current dataset"""
        names = self.landmarks_names
        return sorted(names)

    def sample(self, n):
        """Pick n random data points (with labels)"""
        return random.choices(self.pc_data, k=n)