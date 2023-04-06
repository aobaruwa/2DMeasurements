import numpy as np
import pickle
import os
import sys
from tqdm import tqdm
from glob import glob

def create(all_data, landmark_names):
    names = sorted([x.strip() for x in landmark_names])
    xs = []; ys=[]; tokens=[]; 
    maxsize=0
    # extract train_data
    for dd in tqdm(all_data): 
        key=list(dd.keys())[0] # Only one
        size=256000
#        if len(dd[key]['PC'])  > maxsize:
#            maxsize=len(dd[key]['PC']) 
#            print(maxsize)
        if len(dd[key]['PC']) < size:
            select=np.random.choice(range(len(dd[key]['PC'])),len(dd[key]['PC']),replace=False)
            xs.append(np.expand_dims(np.pad(dd[key]['PC'][select],[[0,size-len(dd[key]['PC'])],[0,0]]),0))
        else:
            select=np.random.choice(range(size),size,replace=False)

            xs.append(np.expand_dims(dd[key]['PC'][select],0))

        lndmks = []
        token=[]
        for i,l in enumerate(names):
            if l in dd[key]['LM']:
                lndmks.append(dd[key]['LM'].get(l, [-999,-999,-999]))
                token.append(i+1)
        ys.append(np.expand_dims(np.pad(lndmks,[[0,128-len(lndmks)],[0,0]] ),0))
        tokens.append(np.expand_dims(np.pad(token,[[0,128-len(token)]]),0))
        
    xs=np.concatenate(xs,axis=0)
    ys=np.concatenate(ys,axis=0)
    tokens=np.concatenate(tokens,axis=0)

    # extract val_data
    print("size: ",len(ys))
    # separate the point cloud list into 6673 train set and 2000 val set
    np.save(os.path.join(data_path, 'train.npy'), list(zip(xs[:-2000],tokens[:-2000],ys[:-2000])))
    np.save(os.path.join(data_path, 'val.npy'), list(zip(xs[-2000:],tokens[-2000:], ys[-2000:])))
    landmark_names.close()

def clean_landmark_names(data_path):
    """Clean data by removing landmark names that end with pound symbol, #
    'Lt. Iliocristale#' becomes 'Lt. Iliocristale' """
    fnames = glob(os.path.join(data_path, 'train/*')) + glob(os.path.join(data_path, 'val/*'))
    all_data = []
    
    for fname in tqdm(fnames):
        dd=pickle.load(open(fname,'rb'))
        key=list(dd.keys())[0] # Only one
        clean_landmarks = {}
        for l in dd[key]['LM'].items():
            clean_landmarks[l[0].replace('#','').strip()] = l[1]
        data_point = {key: {'PC':dd[key]['PC'],'LM':clean_landmarks}}
        all_data.append(data_point)
    return all_data

if __name__ == '__main__':

    data_path = sys.argv[1] #'/home/abaruwa/datascience/data'
    landmark_names = open(sys.argv[2], 'r')  # '/home/abaruwa/datascience/landmark_mine/scripts/landmark_names.txt'
    all_data = clean_landmark_names(data_path)
    create(all_data,landmark_names)