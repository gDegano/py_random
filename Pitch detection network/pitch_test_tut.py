# -*- coding: utf-8 -*-
"""
Created on Sat Jun 09 17:28:47 2018

@author: gxd606
"""

import numpy as np
import librosa
import keras
from future.utils import implements_iterator  # for python 2 compatibility for __next__()
from matplotlib import pyplot as plt

plt.rc('figure', titlesize=20)  
plt.rc('font', size=20)
plt.rc('xtick', labelsize=12)

def sin_wave(secs, freq, sr, gain):
    '''
    Generates a sine wave of frequency given by freq, with duration of secs.
    '''
    t = np.arange(sr * secs)
    return gain * np.sin(2 * np.pi * freq * t / sr)

def whitenoise(gain, shape):
    '''
    Generates white noise of duration given by secs
    '''
    return gain * np.random.uniform(-1., 1., shape)

class DataGen:
    
    '''
    Generates some training data
    '''

    def __init__(self, sr=16000, batch_size=128):
        np.random.seed(1209)
        self.pitches = [440., 466.2, 493.8, 523.3, 554.4, 587.3,
                        622.3, 659.3, 698.5, 740., 784.0, 830.6]

        self.sr = sr
        self.n_class = len(self.pitches)  # 12 pitches
        self.secs = 1.
        self.batch_size = batch_size
        self.sins = []
        self.labels = np.eye(self.n_class)[range(0, self.n_class)]  # 1-hot-vectors

        for freq in self.pitches:
            cqt = librosa.cqt(sin_wave(self.secs, freq, self.sr, gain=0.5), sr=sr,fmin=220, n_bins=36, filter_scale=2)[:, 1]  # use only one frame!
            cqt = librosa.amplitude_to_db(cqt, ref=np.min)
            cqt = cqt / np.max(cqt)
            self.sins.append(cqt)

        self.cqt_shape = cqt.shape  # (36, )

    def __next__(self):
        choice = np.random.choice(12, size=self.batch_size, replace=True)
        noise_gain = 0.1 * np.random.random_sample(1)  # a random noise gain 
        noise = whitenoise(noise_gain, self.cqt_shape)  # generate white noise
        xs = [noise + self.sins[i] for i in choice]  # compose a batch with additive noise
        ys = [self.labels[i] for i in choice] # corresponding labels

        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    next = __next__
    
    
datagen = DataGen()
print("Input: A frame of CQT in a shape of: {}".format(datagen.cqt_shape))
x, y = next(datagen)
print("Input batch: CQT frames, {}".format(x.shape))
print("Number of classes (pitches): {}".format(datagen.n_class))
#plt.figure(figsize=(20, 6))
#
#for i in range(2):
#    x, y = next(datagen)
#    plt.subplot(2, 2, i+1)
#    plt.imshow(x.transpose(), cmap=plt.get_cmap('Blues'))
#    plt.xlabel('data sample index')
#    plt.ylabel('pitch index')
#    plt.title('Batch {} (x, input)'.format(i+1))
#    plt.subplot(2, 2, i+3)
#    plt.imshow(y.transpose(), cmap=plt.get_cmap('Blues'))
#    plt.title('Batch {} (y, label)'.format(i+1))
#
#print('')

val_datagen = DataGen()
    
model = keras.models.Sequential()
model.add(keras.layers.Dense(datagen.n_class, use_bias=False,
                             input_shape=datagen.cqt_shape)) # A dense layer (36 input nodes --> 12 output nodes)
model.add(keras.layers.Activation('softmax'))  # Softmax because it's single-label classification

model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9,  # a pretty standard optimizer
                                             decay=1e-6, nesterov=True),
              loss='categorical_crossentropy',  # categorical crossentropy makes sense with Softmax
              metrics=['accuracy'])  # we'll also measure the performance but it's NOT a loss function    
    
    
history = model.fit_generator(datagen, steps_per_epoch=200, epochs=25, verbose=1,
                             validation_data=val_datagen, validation_steps=4)   
    
    
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['acc'], label='training')
plt.plot(history.history['val_acc'], label='validation', alpha=0.7)
plt.title('Accuracy')
plt.xlabel('epoch')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='training')
plt.plot(history.history['val_loss'], label='validation', alpha=0.7)
plt.title('Loss')
plt.xlabel('epoch')
plt.legend()

    
    
    
    
    
    