import numpy as np
import tensorflow as tf
    
class PointNet():
    def __init__(self, ckpt_file=None, mask_value=-999):
        self.mask_value = mask_value
        self.ckpt_file= ckpt_file

    def build_model(self, max_pc_size=200000):
        
        pc_input=tf.keras.layers.Input((max_pc_size,3))
        masked_input=tf.keras.layers.Masking(mask_value=self.mask_value)(pc_input)
        convolution=tf.keras.layers.Conv1D(128,1)(masked_input)    
        convolution=tf.keras.layers.LeakyReLU()(convolution)
        convolution=tf.keras.layers.Conv1D(128,1)(convolution)
        convolution=tf.keras.layers.LeakyReLU()(convolution)
        convolution=tf.keras.layers.Conv1D(128,1)(convolution)
        convolution=tf.keras.layers.LeakyReLU()(convolution)
        features=tf.keras.layers.GlobalMaxPooling1D()(convolution)
        features=tf.keras.layers.Dense(512)(features)
        features=tf.keras.layers.LeakyReLU()(features)
        output=tf.keras.layers.Dense(3)(features)

        model=tf.keras.models.Model([pc_input],[output])
        model.compile(loss='mse',optimizer='adam')

        ### Load pretrained weights
        if self.ckpt_file:
            model.load_weights(self.ckpt_file)
        return model

    #    @tf.function
    def squared_dist(self, y_true,y_pred):
        ex_a = tf.expand_dims(y_true, 1)
        ex_b = tf.expand_dims(y_pred, 2)
        dist=tf.reduce_sum(tf.square(ex_a-ex_b),axis=-1)
        return dist#tf.math.squared_difference(ex_a, ex_b)

    #    @tf.function
    def chamfer_dist(self, y_true,y_pred):
        dists=self.squared_dist(y_true,y_pred)
        print(dists)
        s1=tf.reduce_sum(tf.reduce_min(dists,axis=1),axis=-1)
        s2=tf.reduce_sum(tf.reduce_min(dists,axis=2),axis=-1)
        chamfer=tf.reduce_mean(s1+s2)
        print('chamfer',chamfer)
        return chamfer

    def train(self):
        """Condense the pointnet trainer script here"""
        raise NotImplementedError

    def _predict(self, xs):
        """
        Inputs: 
            pretrained_models - List of file paths of CNN models
            xs - numpy body scan object : NxWxC 
            locations - landmark names to be identified on
            the 3D body.
        Outputs:
            point_esimate - 3D point estimate of a landmark
            on the body scan object
        """
        xs = np.expand_dims(xs, axis=0) if len(xs.shape) !=3 else xs  # add batch dimension
        model = self.build_model(xs.shape[1])
        prediction=model.predict(xs)[0,:]
        return prediction
  
