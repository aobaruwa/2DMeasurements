from glob import glob
from landmark import cnn_classifier
import pickle
import os 
import numpy as np
import tensorflow as tf
from landmark import utils

all_data={}


ys=[]
land_mark='10th Rib Midspine'


example_obj=utils.read_obj('/gpfs/projects/datascience/shared/OtherScans/3DMDScan.obj')

xs=np.array([example_obj])

model=cnn_classifier.build_model(xs.shape[1],-999)
model.compile(loss='mse',optimizer='adam')
model.summary()

model.load_weights('tests.h5')

prediction=model.predict(xs)
lm={land_mark:prediction[0,:]}
utils.write_landmark_xml('test.xml',lm)


#if not os.path.exists(land_mark+'.npy'):

#    for fname in glob('data/*'):
#        dd=pickle.load(open(fname,'rb'))
#        key=list(dd.keys())[0] # Only one
#        if land_mark in dd[key]['LM']:
#            xs.append(dd[key]['PC'])
#            ys.append(dd[key]['LM'][land_mark])
#    np.save(land_mark,(xs,ys))
#else:
#    xs,ys=np.load(land_mark+'.npy',allow_pickle=True)

#xs=tf.keras.preprocessing.sequence.pad_sequences(xs,dtype='float32',value=-999)
#ys=np.concatenate([np.expand_dims(i,0) for i in ys],axis=0)
#print('Finished with data')


#predictions=model.predict(xs)

#np.save("truth_and_pred",(ys,predictions))


