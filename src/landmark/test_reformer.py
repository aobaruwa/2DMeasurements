from reformer import gen, build_model
from trax.fastmath import numpy as jnp
from tqdm import tqdm, trange
from itertools import chain
import jax
import numpy as np
import pickle
import os
import sys
import trax 
import pdb
trax.fastmath.use_backend('jax')

def main(model_fpath, data_dir, landmark_names, pc_feat_size):
    """Predict the landmarks given a trained reformer model and a (labelled) test dataset.
       Generate the tail distribution of landmark prediction errors in meters from the 95th 
       percentile of errors.
    """
    batch_size = 32
    test_data = jnp.load(data_dir, allow_pickle=True)
    test_stream = gen(test_data, pc_feat_size)
    test_pipeline = trax.data.Serial(trax.data.Batch(batch_size))
    test_batches = test_pipeline(test_stream)
#    sample_batch = next(test_batches)
#    eb=[jnp.array(i) for i in sample_batch]
  
    print("loading model")

    model = build_model()
#    model.init(trax.shapes.signature(eb))
    model.init_from_file(file_name=model_fpath)#, weights_only=True)
    
    model_fn = jax.jit(model)
    print("running predictions") 
    pred_batches = [model_fn([jnp.array(i) for i in next(test_batches)])[0] for _ in trange(len(test_data)//batch_size)]
    predictions= list(chain.from_iterable(pred_batches))
    print(len(predictions))

    truth_landmarks = [pc_data[2] for pc_data in test_data]
    
    pred_errors = {name:[] for name in landmark_names} # gather all prediction errors in all pclouds for each landmark
    
    for i, name in enumerate(tqdm(landmark_names)):
        for j in range(len(predictions)):
            tokens2idx = {tok:idx for idx,tok in enumerate(test_data[j][1])} 
            if i+1 not in tokens2idx: # landmark name not in pointcloud
                print("test pcloud", j, "does not have", name)
                continue
            
            pred_lndmrk = predictions[j][tokens2idx[i+1]]
            truth_lndmrk = truth_landmarks[j][tokens2idx[i+1]]
            pred_errors[name].append(np.linalg.norm(pred_lndmrk - truth_lndmrk))

    error_tails = {name: np.percentile(pred_errors[name], 95) for name in pred_errors.keys() if pred_errors[name]}
    print (error_tails)
    pickle.dump(error_tails, open(os.path.join(os.path.dirname(model_fpath), 'ref_err_tails.pkl'), 'wb'))
    pdb.set_trace()

    return


if __name__ == '__main__':
    
    model_fpath = sys.argv[1]
    data_dir = sys.argv[2]
    pc_feat_size = int(sys.argv[3])
    landmark_names_file = sys.argv[4] 
    f = open(landmark_names_file, 'r')
    names = sorted([x.strip() for x in f.readlines()])
   
    f.close()

    main(model_fpath, data_dir, names, pc_feat_size)

"""                           
python test_reformer.py "/home/abaruwa/datascience/landmark_mine/reformer_out_gpu_64000/model.pkl.gz" \
                        "/home/abaruwa/datascience/data/val.npy" \
                        64000 \
                        "/home/abaruwa/datascience/landmark_mine/scripts/landmark_names.txt"
"""
