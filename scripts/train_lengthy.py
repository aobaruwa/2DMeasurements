from glob import glob
from landmark import cnn_classifier
from tqdm import trange
from tqdm import tqdm
import numpy as np
import operator
import os
import pickle
import random
import sys
import tensorflow as tf

land_mark='10th Rib Midspine'

def load_data(val_split=0.2, batch_size=32, shuffle=True):
    all_data={}

    xs=[]
    ys=[]

    if not os.path.exists(land_mark+'.npy'):

        for fname in glob('data/*'):
            dd=pickle.load(open(fname,'rb'))
            key=list(dd.keys())[0] # Only one
            if land_mark in dd[key]['LM']:
                xs.append(dd[key]['PC'])
                ys.append(dd[key]['LM'][land_mark])
        np.save(land_mark,(xs,ys))
    else:
        xs,ys=np.load(land_mark+'.npy',allow_pickle=True)
    xs=tf.keras.preprocessing.sequence.pad_sequences(xs,dtype='float32',value=-999)
    ys=np.concatenate([np.expand_dims(i,0) for i in ys],axis=0)

    if shuffle:
        dat = tf.data.Dataset.from_tensor_slices((xs,ys)).shuffle(buffer_size=10000)
    else: 
        dat = tf.data.Dataset.from_tensor_slices((xs, ys))
    print('Finished with data\n')
    # arrange data into batches
    batches=dat.batch(batch_size,drop_remainder=False)
    return batches

def train(epochs, batch_size, fptrs):
    train_data = load_data(batch_size=batch_size)
    tot_train_loss, tot_val_loss = 0,0
    #feat_size = train_data.take(1)
    #print(f"Feature size- {feat_size}\n")
    model = cnn_classifier.build_model(228156,-999)
    opt = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()
    for epoch in range(epochs):
        # train step
        for step, (x_batch, y_batch) in enumerate(tqdm(train_data)):
            train_loss = train_step(x_batch, y_batch, model, opt, loss_fn)
            tot_train_loss += train_loss
            # record loss
            fptrs[0].write(f"{epoch}\t{step}\t{train_loss}\n")
        # val step
        val_loss = val_step(x_batch, y_batch, model, loss_fn)
        tot_val_loss += val_loss
        fptrs[1].write(f"{epoch}\t{val_loss}\n")
    return

@tf.function
def train_step(x_batch, y_batch, model, opt, loss_fn):
    # forward pass
    with tf.GradientTape() as tape:
        preds = model(x_batch, training=True)
        train_loss = loss_fn(y_batch, preds)
    # get gradients
    grads = tape.gradient(train_loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return train_loss

@tf.function
def val_step(x_batch, y_batch, model, loss_fn):
    preds = model(x_batch, training=False)
    val_loss = loss_fn(y_batch, preds)
    return val_loss

def main():
    epochs = 50 # put this in an argparser!
    batch_size = 32
    # create logfiles
    output_dir = sys.argv[1]
    train_log_file = os.path.join(output_dir, 'train_log.txt')
    val_log_file = os.path.join(output_dir, 'val_log.txt')
    fpt1 = open(train_log_file, 'a')
    fpt1.write("epoch\tstep\ttrain_loss\n")
    fpt2 = open(val_log_file, 'a')
    # train classifier
    train(epochs, batch_size, (fpt1, fpt2))
    fpt1.close()
    fpt2.close()
    

if __name__=="__main__":
    main()