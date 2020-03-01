from keras import backend as K
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops
eps = 7./3 - 4./3 -1

def relu(matrix):
    return np.maximum(0,matrix)

def get_outputs(image, model):
    outputs = []
    for i in range(0,len(model.layers)):
        layer_output = K.function([model.layers[0].input],
                                      [model.layers[i].output])
        outputs.append(layer_output([image.reshape(tuple([1]+list(image.shape)))])[0])
    return outputs

def get_inputs(image, model):
    inputs = []
    for i in range(0,len(model.layers)):
        layer_input = K.function([model.layers[0].input],
                                      [model.layers[i].input])
        inputs.append(layer_input([image.reshape(tuple([1]+list(image.shape)))])[0])
    return inputs

def get_weights(model):
    weights = []
    for i in range(0,len(model.layers)):
        try:
            weights.append(model.layers[i].get_weights()[0])
        except: 
            weights.append(None)
            #print("error for weights in layer: ",i," , name: ",model.layers[i])
    return weights

def get_biases(model):
    biases = []
    for i in range(0,len(model.layers)):
        try:
            biases.append(model.layers[i].get_weights()[1])
        except: biases.append(None)
    return biases

def relprop_lin(layer, R, inputs, weights):
    Z = np.matmul(inputs[layer],weights[layer])+eps
    S = R/Z
    C = np.matmul(S,weights[layer].T)
    R = C*inputs[layer]
    return R

def relprop_lin_ab(layer, R, inputs, weights,biases, a=2, b=1):
    Z_p = np.matmul(inputs[layer],weights[layer]*(weights[layer]>=0))+biases[layer]*(biases[layer]>=0)+eps
    Z_n = np.matmul(inputs[layer],weights[layer]*(weights[layer]<0))-biases[layer]*(biases[layer]<0)-eps
    S_p = R/Z_p
    S_n = R/Z_n
    C_p = np.matmul(S_p,(weights[layer]*(weights[layer]>=0)).T)
    C_n = np.matmul(S_n,(weights[layer]*(weights[layer]<0)).T)
    R = (C_p*inputs[layer]*a-C_n*inputs[layer]*b)
    return R

def relprop_flatten(layer,R, inputs):
    return R.reshape(inputs[layer].shape)

def relprop_pooling(layer, R, inputs, outputs, model):
    pool_height, pool_width = model.layers[layer].pool_size
    stride_up, stride_side = model.layers[layer].strides
    placeholder = np.zeros(inputs[layer].shape)
    Z = outputs[layer]
    S = R/(Z+eps)
    for l in range(outputs[layer].shape[3]):
        for i in range(outputs[layer].shape[2]):
            for j in range(outputs[layer].shape[1]):
                placeholder[0,i*stride_side:(i*stride_side+pool_width),j*stride_up:(j*stride_up+pool_height),l]= S[0,i,j,l]*((inputs[layer][0,i*stride_side:(i*stride_side+pool_width),j*stride_up:(j*stride_up+pool_height),l]==outputs[layer][0,i,j,l])&(outputs[layer][0,i,j,l]!=0))
    R = placeholder*inputs[layer]
    return R

def relprop_pooling1(layer, R, inputs, outputs, model):
    placeholder = K.eval(gen_nn_ops.max_pool_grad_v2(inputs[layer], outputs[layer], R/(outputs[layer]+eps), (1,2,2,1), (1,2,2,1), padding='VALID'))
    R = placeholder*inputs[layer]
    return R

def relprop_conv2d(layer, R, inputs, weights, model):
    Z = K.eval(K.conv2d(tf.constant(inputs[layer]),tf.constant(weights[layer]),strides=(1,1), padding=model.layers[layer].padding))+eps
    S = R/Z
    C = K.eval(K.tf.nn.conv2d_backprop_input(inputs[layer].shape, weights[layer],S, (1,1,1,1),padding=model.layers[layer].padding.upper() ))
    R = C*inputs[layer]
    return R

def relprop_conv2d_first(layer, R, inputs, weights, model):
    X = inputs[layer]
    L = inputs[layer]*0 + -1
    H = inputs[layer]*0 + 1
    W_pos = np.maximum(0,weights[layer])
    W_neg = np.minimum(0,weights[layer])

    Z = K.eval(K.conv2d(tf.constant(inputs[layer]),tf.constant(weights[layer]),strides=(1,1), padding=model.layers[layer].padding))
    Z = Z - K.eval(K.conv2d(tf.constant(L),tf.constant(W_pos),strides=(1,1), padding=model.layers[layer].padding))
    Z = Z - K.eval(K.conv2d(tf.constant(H),tf.constant(W_neg),strides=(1,1), padding=model.layers[layer].padding))+eps
    S = R/Z
    C = K.eval(K.tf.nn.conv2d_backprop_input(inputs[layer].shape, weights[layer],S, (1,1,1,1),padding=model.layers[layer].padding.upper() ))
  
    R = C*inputs[layer] - C*L - C*H
    return R


def relprop_conv2d_ab(layer, R, inputs, weights, model,biases, a=2, b=1):
    Z_p = K.eval(K.conv2d(tf.constant(inputs[layer]),tf.constant(weights[layer]*(weights[layer]>=0)),strides=(1,1), padding=model.layers[layer].padding))+biases[layer]*(biases[layer]>=0)+eps
    Z_n = K.eval(K.conv2d(tf.constant(inputs[layer]),tf.constant(weights[layer]*(weights[layer]<0)),strides=(1,1), padding=model.layers[layer].padding))-biases[layer]*(biases[layer]>=0)-eps
    S_p = R/Z_p
    S_n = R/Z_n
    C_p = K.eval(K.tf.nn.conv2d_backprop_input(inputs[layer].shape,weights[layer]*(weights[layer]>=0),S_p, (1,1,1,1),padding=model.layers[layer].padding.upper() ))
    C_n = K.eval(K.tf.nn.conv2d_backprop_input(inputs[layer].shape,weights[layer]*(weights[layer]<0),S_n, (1,1,1,1),padding=model.layers[layer].padding.upper() ))
    R = (C_p*inputs[layer]*a-C_n*inputs[layer]*b)
    return R

def relprop(model, img, R, a=2, b=1, verbose = True):
    if verbose: 
        print("calculating LRP of ",str(model))
        print("###################")
        print("getting values")
    inputs = get_inputs(img, model)
    outputs = get_outputs(img, model)
    weights = get_weights(model)
    biases = get_biases(model)
    if verbose: print("propagating relevance regarding classification: ", np.argmax(R))
    for i in range(-1,-len(model.layers),-1):

        if (i-1 == -len(model.layers)):
            R = relprop_conv2d_first(i,R,inputs, weights, model)
            if verbose: print("In layer ",i," : ",model.layers[i]," check-value: ", np.sum(R))
            break

        if "Dense" in str(model.layers[i]):
            R = relprop_lin_ab(i, R, inputs, weights,biases, a, b)
            if verbose: print("In layer ",i," : ",model.layers[i]," check-value: ", np.sum(R))
        if "Flatten" in str(model.layers[i]):
            R = relprop_flatten(i , R, inputs)
            if verbose: print("In layer ",i," : ",model.layers[i]," check-value: ", np.sum(R))
        if "MaxPool" in str(model.layers[i]):
            R = relprop_pooling(i ,R , inputs, outputs, model)
            if verbose: print("In layer ",i," : ",model.layers[i]," check-value: ", np.sum(R))
        if "Conv2D" in str(model.layers[i]):
            R = relprop_conv2d_ab(i, R, inputs, weights, model,biases, a, b)
            if verbose: print("In layer ",i," : ",model.layers[i]," check-value: ", np.sum(R))
        if "Dropout" in str(model.layers[i]):
            None

    return R


def draw_superimposed(R, img):
    img_R = np.add.reduce(R.reshape(R.shape[1:]),-1)
    img_R = np.interp(img_R, (img_R.min(), img_R.max()), (0, +1))
    cmap = plt.get_cmap("jet")
    heatmap = cmap(img_R)
    heatmap = np.delete(heatmap, 3,2)
    plt.imshow(img*0.4+heatmap*0.6)