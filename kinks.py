
#Imports
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import random as ran
import json
import h5py
import pickle
from pprint import pprint
from __future__ import print_function
import numpy
import matplotlib.image as mpimg
import csv

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, LocallyConnected2D
from keras import backend as K
from keras import optimizers
from keras import regularizers
from keras.utils import to_categorical
from sklearn.preprocessing import normalize
from keras.utils import to_categorical
from sklearn.preprocessing import normalize
from pprint import pprint
from keras.models import model_from_json
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, Dense

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from keras.layers.normalization import BatchNormalization
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.patheffects as PathEffects


#Get samples of 2d and train the model with those 2d data, this will help to analysis on kink
def samples_2d():
    # generate samples
    X = np.load('./data_kinks/x_2d_moon.npy')
    y = np.load('./data_kinks/y_2d_moon.npy')
    # one hot encode output variable
    y = to_categorical(y)
    # split into train and test
    n_train = 300
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]
    return trainX, trainy, testX, testy
# prepare data
x_train_2d, y_train_2d, x_test_2d, y_test_2d = samples_2d()
print(x_train_2d.shape)
print(y_test_2d.shape)

#From the 2d samples, make them sequential so that at first all red (0) labels, then all blue (1) labels
train_labels_2d = y_train_2d.argmax(1)

#Creating the samples of red (0) and blue (1) as if they are sequentially
x_train_2d_label0_inds = [i for i in range(len(train_labels_2d)) if train_labels_2d[i]==0]
x_train_2d_label1_inds = [i for i in range(len(train_labels_2d)) if train_labels_2d[i]==1]
x_train_2d_label_0_to_1_inds = x_train_2d_label0_inds+ x_train_2d_label1_inds

x_train_2d_seq = x_train_2d[x_train_2d_label_0_to_1_inds]
y_train_2d_seq = y_train_2d[x_train_2d_label_0_to_1_inds]
print('The shape of 2d data: ',x_train_2d_seq.shape)
print('The sequential labels are:\n',y_train_2d_seq.argmax(1))


#Define a model with those 2d data
#Considering one hidden ReLU layer
def fully_connected_model_2d():
    model = Sequential()
    model.add(Dense(256, input_dim=2, activation='relu', bias_initializer='RandomNormal', kernel_initializer='RandomUniform')) #Using one ReLU layer like in the paper Quantizes ReLU
    model.add(Dense(2, activation='softmax'))
    return model

#Configuring the checkpointer
filepath3="./save_models/weights_model_{epoch:02d}.hdf5"
checkpointer3_2d = keras.callbacks.ModelCheckpoint(filepath3, monitor='val_loss',
                                                verbose=1, save_best_only=False,save_weights_only=False, mode='max', period=1)

#Train and saving the models
fully_model_2d=fully_connected_model_2d()
sgd = optimizers.SGD()
fully_model_2d.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
fully_model_2d.save('./save_models/Model_0th_epoch_not_trained_initialized_al.hdf5')

fully_model_2d_log = fully_model_2d.fit(x_train_2d_seq, y_train_2d_seq, batch_size=32, 
                          epochs=7000, verbose=1, validation_data=(x_train_2d_seq, y_train_2d_seq),callbacks=[checkpointer3_2d])

score = fully_model_2d.evaluate(x_test_2d, y_test_2d, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#Number of kinks and distance matrix calculation for a particular model
def get_numbers_of_kinks_for_a_model(model_conisdered, weights, biases, layer_outs,threshold):
    r = threshold
    numbers_of_kinks = []
    distance_matrix = []
    for i in range(x_train_2d_seq.shape[0]):
        kink_count = 0
        for j in range(layer_outs.shape[2]):
            w1 = weights[0][j]
            w2 = weights[1][j]
            b = biases[j]
            #print("\nInput to ReLU for sample number ", i,'and ', 'neuron ',j,': ', (w1*x_train_2d_seq[i][0] + w2*x_train_2d_seq[i][1] + b ) )
            numerator_of_eqn = (w1*x_train_2d_seq[i][0] + w2*x_train_2d_seq[i][1] + b )
            sqrt_w1_sq_w2_sq = math.sqrt((w1**2)+(w2**2))
            #print(i, ' ', math.sqrt((w1**2)+(w2**2)))
            #lhs_calculated_dist = abs(layer_outs[0][i][j])/sqrt_w1_sq_w2_sq #You can use that
            lhs_calculated_dist = abs(numerator_of_eqn)/sqrt_w1_sq_w2_sq #Or alternatively this one works too
            distance_matrix.append(lhs_calculated_dist)
            #print(i, ' ', j, ' ', lhs_calculated_dist)
            if(lhs_calculated_dist<=r):
                kink_count = kink_count+1
        #print('\nThe number of kinks are for sample', i ,': ', kink_count)
        numbers_of_kinks.append(kink_count)
    distance_matrix_np = np.array(distance_matrix).reshape(300,256)
    return numbers_of_kinks, distance_matrix_np

#considered_model_nums = [2000] #For a single model check
considered_model_nums = [1000, 2000, 3000]
#considered_model_nums = [0, 1, 5, 10, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
numbers_of_kinks_model_all = []
distance_matrix_model_all = []
threshold = 0.005
print('\nThe considered threshold (r) is: ', threshold)
print("Here I defined Kinks as when (w1*x1 + w2*x2 + b)/ (sqrt(w1**2 + w2**2)) <= r \n")
print("The shape of 2d training samples is: ", x_train_2d_seq.shape)
for j,i in enumerate(considered_model_nums):
    #Load the model
    model_filepath_fully = './models_kinks/weights_model'+'_'+str(i)+'.hdf5'
    model_loaded_fully = load_model(model_filepath_fully)
    print('\nLoaded Model number is:',i);
    #Get weights and biases of the loaded model
    weights = model_loaded_fully.layers[0].get_weights()[0]
    biases = model_loaded_fully.layers[0].get_weights()[1]
    inp = model_loaded_fully.input 
    outputs = [layer.output for layer in model_loaded_fully.layers]          # all layer outputs
    x = outputs[0].op.inputs[0]
    functor_k = K.function([inp, K.learning_phase()], [x])   # evaluation function
    layer_outs = np.array(functor_k([x_train_2d_seq, 0.])) #Get the input to ReLU
    #Getting the number of kinks and distance matrix for the 2d, 300 training samples on a model
    numbers_of_kinks_a_model, distance_matrix_a_model = get_numbers_of_kinks_for_a_model(model_loaded_fully, weights, biases, layer_outs, threshold)
    print('Kinks for this considered model on 300 training 2d samples:\n', numbers_of_kinks_a_model)
    numbers_of_kinks_model_all.append(numbers_of_kinks_a_model)
    distance_matrix_model_all.append(distance_matrix_a_model)
numbers_of_kinks_model_all_np = np.array(numbers_of_kinks_model_all)
distance_matrix_model_all_np = np.array(distance_matrix_model_all)
#Distance matrix for the 2d samples calculated by the formula (w1*x1 + w2*x2 + b)/ (sqrt(w1**2 + w2**2))
#print(distance_matrix_model_all_np.shape)
#print(distance_matrix_model_all_np)

#Save the results to a csv for further analysis
#f1 = open('./results_kinks/kinks_numbers_some_epochs_r_.005_for_300_2d_training_samples_4.csv', 'w')
#f1 = open('./results_kinks/distance_matrix_model_number_3000_kinks_1_layer_256 neurons_300_2d_training_samples.csv', 'w')
writer = csv.writer(f1, delimiter=',')
#writer.writerows(numbers_of_kinks_model_all_np)
writer.writerows(distance_matrix_model_all_np[2])


#Where is the max value of number of kinks
#print(numbers_of_kinks_model_all_np.shape)
#print(distance_matrix_model_all_np.shape)
#print(np.max(numbers_of_kinks_model_all_np))
#print(np.where(numbers_of_kinks_model_all_np == np.amax(numbers_of_kinks_model_all_np)))

#Plots
#Plot the density of the numbers of kinks on each of the model
import seaborn as sns
sns.set(color_codes=True)
for j,i in enumerate(considered_model_nums):
    h = numbers_of_kinks_model_all_np[j]
    sns.kdeplot(h,bw=1,label =i)
# Plot formatting and save
plt.rcParams.update({'font.size': 12})
plt.legend(prop={'size': 6}, title = 'model number',ncol=2, loc='upper right')
plt.xlabel('numbers_of_kinks')
plt.ylabel('density')
#plt.savefig('./results_kinks/density_plot_kinks_some_epochs_r_0.1_bw_1_2d_data_300_training_samples.pdf')
plt.show()

#Plot the number of kinks of the samples 
for j,i in enumerate(considered_model_nums):
    plt.plot(numbers_of_kinks_model_all_np[j],linewidth =0.01, label = i)
plt.legend(prop={'size': 6}, title = 'model number',ncol=2, loc='upper left')
#plt.savefig('./results_kinks/plot_kinks_some_epochs_2d_data_300_training_samples_1.pdf')

#Quick rough check the input to ReLU values with weights and biases of a particular model
model_filepath_fully = './models_kinks/weights_model'+'_'+str(2000)+'.hdf5'
model_loaded_fully = load_model(model_filepath_fully)
weights = model_loaded_fully.layers[0].get_weights()[0]
biases = model_loaded_fully.layers[0].get_weights()[1]
#print('Shape of weights: ', weights.shape)
#print('Shape of weights: ',biases.shape)

#getting the output from layers
inp = model_loaded_fully.input                                           # input placeholder
outputs = [layer.output for layer in model_loaded_fully.layers]          # all layer outputs
x = outputs[0].op.inputs[0]
functor_k1 = K.function([inp, K.learning_phase()], [x])   # evaluation function

layer_outs = np.array(functor_k1([x_train_2d_seq, 0.]))
print('The input to ReLU values shape: ',np.array(layer_outs).shape)
print('The input to ReLU values: \n',np.array(layer_outs))
print('\nmax:',np.max(layer_outs))
print('\nmin:',np.min(layer_outs))

#Sanity check for the ReLU layer, printing values after the ReLU activation
functor_k2 = K.function([inp, K.learning_phase()], [outputs[0]])   # evaluation function
layer_outs = functor_k2([x_train_2d_seq, 0.])
print(np.array(layer_outs).shape)
print(np.array(layer_outs))