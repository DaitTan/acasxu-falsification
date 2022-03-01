import logging
import math

import numpy as np
import plotly.graph_objects as go
# from numpy.typing import NDArray
from acasxu_tf_keras.gen_tf_keras import build_dnn, read_acasxu_weights
import glob
import tensorflow as tf
import os
import numpy as np
from partx.interfaces.run_standalone import run_partx

acas_net = []
# NeuralNetwork_path = "acasxu_nnet/ACASXU_experimental_v2a_1_1.nnet"
for nnet_path in sorted(glob.glob("acasxu_tf_keras/acasxu_nnet/*.nnet")): 
# for nnet_path in sorted(glob.glob("acasxu_tf_keras/acasxu_nnet/ACASXU_experimental_v2a_1_9.nnet")):    
    net = build_dnn(read_acasxu_weights(nnet_path))
    for layer in net.layers:
        layer.trainable = False
        # layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
    acas_net.append(net)

ensemble_input = [model.input for model in acas_net]
ensemble_output = [model.output for model in acas_net]
# merge = concatenate(ensemble_outputs)
model = tf.keras.Model(inputs = ensemble_input, outputs = ensemble_output)
model.compile()
# model.summary()
tf.keras.models.save_model(model, os.path.basename("ensemble_model.h5"))


def test_property_7(X):
    # X[0] = (X[0] - 1.9791091e+04)/60261.0
    # X[1] = X[1] / 6.28318530718
    # X[2] = X[2] / 6.28318530718
    # X[3] = (X[3] - 650.0)/1100.
    # X[4] = (X[4] - 600.0)/1200.
    input = np.array([[X[0], X[1], X[2], X[3], X[4]]])
    # input = np.array([[X[0], X[1], X[2], 1000., 1000.]])
    model = tf.keras.models.load_model(os.path.basename("ensemble_model.h5"))
    x = model((input,)*45, training=False)
    outputs = np.squeeze(x)
    # outputs = (outputs )
    # print(outputs)
    # print(len(outputs.shape))
    if len(outputs.shape) == 1:
        term_1 = np.bitwise_and(outputs[3]>outputs[0], outputs[4]>outputs[0])
        term_2 = np.bitwise_and(outputs[3]>outputs[1], outputs[4]>outputs[1])
        term_3 = np.bitwise_and(outputs[3]>outputs[2], outputs[4]>outputs[2])
        
        mult = -1* sum([out])
        if mult == 0:
            mult = 1
        return  mult * (np.absolute(outputs[3]) + np.absolute(outputs[4]))
    else:
        term_1 = np.bitwise_and(outputs[:,3]>outputs[:,0], outputs[:,4]>outputs[:,0])
        term_2 = np.bitwise_and(outputs[:,3]>outputs[:,1], outputs[:,4]>outputs[:,1])
        term_3 = np.bitwise_and(outputs[:,3]>outputs[:,2], outputs[:,4]>outputs[:,2])
        out = np.invert(np.bitwise_or(term_1,np.bitwise_or(term_2, term_3)))
        # print(out)
        # print(-1* sum(out))
        
        return -1* sum(out) + 20


# pred = test_property_7([ 86.44793021 , -3.141592  ,   3.141592  , 910.04390116 ,395.94531244])
# print(pred)
# Options initialization

# Test function properties
test_function_dimension = 5
region_support = np.array([[[0.0,60760.], [-3.141592, 3.141592], [-3.141592, 3.141592], [100., 1200. ], [0., 1200. ]]])

# Budgets
initialization_budget = 50
max_budget = 2000
continued_sampling_budget = 100

# BO grid size : number_of_BO_samples * number_of_samples_gen_GP
number_of_BO_samples = [30]
R = 30
M = 10000
NGP = R*M

# Mostly not changes. change with caution
branching_factor = 2
nugget_mean = 0
nugget_std_dev = 0.001
alpha = [0.05]
delta = 0.001

# Other Parameters
number_of_macro_replications = 1
start_seed = 1002
fv_quantiles_for_gp = [0.5, 0.95, 0.99]


results_folder_name = "acas_xu_testing"
BENCHMARK_NAME = "property_7_N_1_9_exp_4"
results_at_confidence = 0.95
run_partx(BENCHMARK_NAME, test_property_7, test_function_dimension, region_support, 
              initialization_budget, max_budget, continued_sampling_budget, number_of_BO_samples, 
              NGP, M, R, branching_factor, nugget_mean, nugget_std_dev, alpha, delta,
              number_of_macro_replications, start_seed, fv_quantiles_for_gp, results_at_confidence, results_folder_name)
                
