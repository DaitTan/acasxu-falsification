#!/usr/bin/env python3
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from numpy import argmax
from scipy.stats import norm
from pathos.multiprocessing import ProcessingPool as Pool
from .calculate_robustness import calculate_robustness
from .sampling import lhs_sampling
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, RBF,WhiteKernel
from .optimizer_gpr import optimizer_lbfgs_b
from scipy.optimize import Bounds
from scipy.optimize import minimize, differential_evolution, basinhopping, dual_annealing
from .sampling import lhs_sampling
from .sampling import uniform_sampling
from sklearn.preprocessing import StandardScaler
from ..utilities.utils_partx import initial_theta_estimate
# import GPy

def surrogate(model, X:np.array):
    """Surrogate Model function

    Args:
        model ([type]): Gaussian process model
        X (np.array): Input points

    Returns:
        [type]: predicted values of points using gaussian process model
    """
	# catch any warning generated when making a prediction
    with catch_warnings():
        # ignore generated warnings
        simplefilter("ignore")
        return model.predict(X, return_std=True)


def acquisition(X: np.array, Xsamples: np.array, model):
    """Acquisition function

    Args:
        X (np.array): sample points 
        Xsamples (np.array): randomly sampled points for calculating surrogate model mean and std
        model ([type]): Gaussian process model

    Returns:
        [type]: Sample probabiility of each sample points
    """
    Xsamples = Xsamples.reshape(1,1,Xsamples.shape[0])
    # print(X.shape)
    # calculate the best surrogate score found so far
    yhat, _ = surrogate(model, X)
    
    # print(yhat)
    # print("****************")
    curr_best = min(yhat)

    # calculate mean and stdev via surrogate function
    mu, std = surrogate(model, Xsamples[0,:,:])
    mu = mu[:, 0]
    # print(mu.shape)
    # print(std.shape)
    # print(curr_best)
    # print(mu)
    # print(std)
    # print("*********************************")
    # print(f)
    pred_var = std
    # print(pred_var)
    # print("*********************************")
    if pred_var >0 :
        var_1 = curr_best-mu
        var_2 = var_1 / pred_var
        
        ei = ((var_1 * norm.cdf(var_2,loc=0,scale=1)) + (pred_var * norm.pdf(var_2,loc=0,scale=1)))
    else:
        ei = 0.0    
    # print(ei)
    # print(f)
    return ei

class MyBounds:
    def __init__(self, xmax, xmin ):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

def opt_acquisition(X: np.array, y: np.array, model, sbo:list ,test_function_dimension:int, region_support: np.array, rng) -> np.array:
    """Get the sample points

    Args:
        X (np.array): sample points 
        y (np.array): corresponding robustness values
        model ([type]): the GP models 
        sbo (list): sample points to construct the robustness values
        test_function_dimension (int): The dimensionality of the region. (Dimensionality of the test function)
        region_support (np.array): The bounds of the region within which the sampling is to be done.
                                    Region Bounds is M x N x O where;
                                        M = number of regions;
                                        N = test_function_dimension (Dimensionality of the test function);
                                        O = Lower and Upper bound. Should be of length 2;

    Returns:
        [np.array]: the new sample points by BO
        [np.array]: sbo - new samples for resuse
    """
    region_support = np.array(region_support.reshape((1,region_support.shape[0],region_support.shape[1])))
    lower_bound_theta = np.ndarray.flatten(region_support[0,:,0])
    upper_bound_theta = np.ndarray.flatten(region_support[0,:,1])
    bnds =  Bounds(lower_bound_theta, upper_bound_theta)
    fun = lambda x_: -1*acquisition(X,x_,model)
    # random_sample = uniform_sampling(1, region_support, test_function_dimension, rng)
    # random_uniform_scaled = input_scaler.transform(random_sample[0])
    # options = {'max_iter':1e10}
    # params = minimize(fun, np.ndarray.flatten(random_sample[:,0,:]), method = 'L-BFGS-B', bounds = bnds)
    # params_2 = differential_evolution(fun, bounds = bnds)
    # minimizer_kwargs = {"method":"L-BFGS-B"}
    # my_bounds = MyBounds(lower_bound_theta, upper_bound_theta)
    # params_3 = basinhopping(fun, np.ndarray.flatten(random_sample[:,0,:]), minimizer_kwargs = minimizer_kwargs, niter = 200, accept_test = my_bounds)
    # print(params)
    # print("********************")
    # print(params_2)
    # print("********************")
    # print(params_3)
    # print(f)
    # params = minimize(fun, np.ndarray.flatten(random_uniform_scaled), method = 'L-BFGS-B', bounds = bnds)
    params_2 = dual_annealing(fun, bounds = list(zip(lower_bound_theta, upper_bound_theta)), no_local_search = False)
    min_bo = params_2.x
    # print(input_scaler.inverse_transform(min_bo))
    # print(params)
    # print(f)
    # print(params)
    # print(f)
    # region_support = np.array(region_support.reshape((1,region_support.shape[0],region_support.shape[1])))
    # scores = acquisition(X, sbo, model)
    # ix = argmax(scores)
    # min_bo = sbo[0,ix,:]
    new_sbo = np.delete(sbo, 0, axis = 1)
    return np.array(min_bo), new_sbo
    



def bayesian_optimization(test_function, samples_in: np.array, corresponding_robustness: np.array, number_of_samples_to_generate: list, test_function_dimension:int, region_support:list, random_points_for_gp: list, rng) -> list:
    """Sample using Bayesian Optimization
    https://machinelearningmastery.com/what-is-bayesian-optimization/

    Args:
        samples_in (np.array): Sample points
        corresponding_robustness (np.array): Robustness values
        number_of_samples_to_generate (list): Number of samples to generate using BO
        test_function_dimension (int): The dimensionality of the region. (Dimensionality of the test function)
        region_support (np.array): The bounds of the region within which the sampling is to be done.
                                    Region Bounds is M x N x O where;
                                        M = number of regions;
                                        N = test_function_dimension (Dimensionality of the test function);
                                        O = Lower and Upper bound. Should be of length 2;
        random_points_for_gp (list): Random samlpes for SBO

    Returns:
        list: Old and new samples (np.array of shape M x N x O). Length of list is number of regions provided in samples_in
                        M = number of regions
                        N = number_of_samples
                        O = test_function_dimension (Dimensionality of the test function) )
        list: corresponding robustness
        list: samples from acquisition function for reuse in classification
    """

    samples_in_new = []
    corresponding_robustness_new = []
    sbo = random_points_for_gp
    for i in range(samples_in.shape[0]):
        X = samples_in[i,:,:]
        Y = corresponding_robustness[i,:].reshape((corresponding_robustness.shape[1],1))
        for j in range(number_of_samples_to_generate[i]):
            # input_scaler = StandardScaler()
            # X_scaled = input_scaler.fit_transform(X)
            # output_scalar = StandardScaler()
            # Y_scaled = output_scalar.fit_transform(Y)

            model = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            normalize_y=True,
            alpha=1e-6,
            n_restarts_optimizer=5,
            optimizer = optimizer_lbfgs_b
            )
            
            # model = GaussianProcessRegressor(
            # alpha=1e-6,
            # normalize_y=True,
            # n_restarts_optimizer=5,
            # optimizer = optimizer_lbfgs_b
            # )
            # print(f)
            # print(X_scaled.shape)
            model.fit(X, Y)
            # print(f)
            min_bo, sbo = opt_acquisition(X, Y, model, sbo, test_function_dimension, region_support[i,:,:], rng)
            actual = calculate_robustness(np.array(min_bo), test_function)
            print("***************************")
            print(min_bo)
            print(actual)
            print("***************************")
            X = np.vstack((X, np.array(min_bo)))
            
            Y = np.vstack((Y, np.array(actual)))
            
        samples_in_new.append(np.expand_dims(X, axis = 0))
        corresponding_robustness_new.append(np.transpose(Y))
    return samples_in_new, corresponding_robustness_new
    

# rng = np.random.default_rng(seed)
# region_support = np.array([[[-1, 1], [-1, 1]]])
# test_function_dimension = 2
# number_of_samples = 20

# x = lhs_sampling(number_of_samples, region_support, test_function_dimension, rng)
# y = calculate_robustness(x)

# x_new, y_new, s = bayesian_optimization(x, y, [10], test_function_dimension, region_support, 10, rng)
# return x_new, y_new,s



# def run_par(data):
#     num_samples, BO_samples, s = data
#     rng = np.random.default_rng(s)
#     region_support = np.array([[[-1, 1], [-1, 1]]])
#     test_function_dimension = 2
#     number_of_samples = num_samples

#     x = lhs_sampling(number_of_samples, region_support, test_function_dimension, rng)
#     y = calculate_robustness(x)

#     x_new, y_new, s = bayesian_optimization(x, y, BO_samples, test_function_dimension, region_support, 10, rng)
#     print(test_function.callCount)
#     return [test_function.callCount]

# inputs = []
# start_seed = 1
# a = [10,10,10,10]
# b = [[20],[21],[19],[22]]
# for q in range(4):
#     s =  start_seed + q
#     data = (a[q], b[q], s)
#     inputs.append(data)

# pool = Pool()
# results = list(pool.map(run_par, inputs))

# print(results)