from models import cnn_classifier
from data.dataset import Dataset
from tqdm import tqdm
import numpy as np
import os
import time
import tensorflow as tf 

class Trainer():
    def __init__(self, config):
        
        self.config=config
        self.optimizers = {'Adam': tf.keras.optimizers.Adam(learning_rate=self.config["lr"]),
                           'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=self.config["lr"])}
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.train_summ_writer = tf.summary.create_file_writer(self.config['log_dir'])
        self.val_summ_writer = tf.summary.create_file_writer(self.config['log_dir'])

    def prepare_batch(self, dataset):
        dataset = Dataset(self.config['dataset_name'])
        xs, ys, tokens = dataset.build(self.config['data_dir'])
        pairs = list(zip(xs,ys,tokens))
        train, val = dataset.split(pairs, 0.8) 
        xs_train, ys_train, tokens_train = zip(*train)
        xs_val, ys_val, tokens_val = zip(*val)

        xs_train=tf.keras.preprocessing.sequence.pad_sequences(xs_train,dtype='float32',value=-999)
        ys_train=np.concatenate([np.expand_dims(i,0) for i in ys_train],axis=0)
        
        xs_val=tf.keras.preprocessing.sequence.pad_sequences(xs_val,dtype='float32',value=-999)
        ys_val=np.concatenate([np.expand_dims(i,0) for i in ys_val],axis=0)

        dat_train = tf.data.Dataset.from_tensor_slices((xs_train,tokens_train, ys_train)).shuffle(buffer_size=10000)
        dat_val = tf.data.Dataset.from_tensor_slices((xs_val, tokens_val, ys_val)).shuffle(buffer_size=10000)
       
        # arrange data into batches
        train_batches=dat_train.batch(self.config['train_batch_size'],drop_remainder=False)
        val_batches = dat_val.batch(self.config['val_batch_size'],drop_remainder=False)
        return train_batches, val_batches

    def train_step(self, x_batch, y_batch, step):
        with tf.GradientTape() as tape:
            preds= self.model(x_batch, training=True)
            train_loss = self.loss_fn(y_batch, preds)
        grads = tape.gradient(train_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        with self.train_summ_writer.as_default():
            tf.summary.scalar('train_loss', train_loss, step=step)
        return train_loss
    
    def val_step(self, x_batch, y_batch, step):
        preds = self.model(x_batch, training=False)
        val_loss = self.loss_fn(y_batch, preds)
        with self.train_summ_writer.as_default():
            tf.summary.scalar('train_loss', val_loss, step=step)
        return val_loss
        
    def train(self):
        train_data, val_data = self.prepare_batch()
        self.optimizer = self.optimizers['Adam']
        loss_fn = tf.keras.losses.MeanSquaredError()
        if self.config['init_from_ckpt']:
            self.model, self.optimizer = self.load_checkpoint
        for epoch in range(self.config['n_epochs']):
            # train step
            for step, (x_batch, y_batch) in enumerate(train_data):
                train_step_loss = self.train_step(x_batch, y_batch, step)
            # eval step
            for step, (x_batch, y_batch) in enumerate(val_data):
                val_step_loss = self.val_step(x_batch, y_batch, step)
        return 

    def save_checkpoint(self,):
        cur_time= '_'.join(time.ctime().split())
        model_dir = os.path.join(self.config['log_dir'], cur_time)
        os.makedirs(model_dir, exist_ok=True)
        self.model.save_weights(
            os.path.join(model_dir, 'model.h5'))
        return 

    def load_checkpoint(self):
        if not self.model:
            self.model = cnn_classifier.build_model()  # to do: handle other model kinds
        self.model.load_weights(self.model)
        return 

if __name__ == "__main__":
    train_manager = Trainer()
    config = {'train_batch_size': 32, 
             'eval_batch_size': 32,
             'train_val_split': 0.2,
             'n_epochs': 100,
             'init_from_ckpt': False,
             'optimizer': 'Adam',
             'learning_rate':1e-5,
             'lr_schedule': 'warm_up',
             'log_dir': None,
             'n_gpu': 1,
             'data_dir': '/datascience/data',
             'dataset_name': 'CAESAR',
            }
    train_manager.train(config)