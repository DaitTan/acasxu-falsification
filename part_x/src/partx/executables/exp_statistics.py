import numpy as np
import matplotlib.pyplot as plt
import pickle
from ..models.partx_options import partx_options
from ..numerical.classification import calculate_volume
from sklearn.gaussian_process import GaussianProcessRegressor
from ..numerical.optimizer_gpr import optimizer_lbfgs_b
from ..numerical.sampling import uniform_sampling, lhs_sampling
from scipy import stats
from ..numerical.calculate_robustness import calculate_robustness
from ..models.testFunction import callCounter
import pathlib
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, RBF,WhiteKernel
from ..utilities.utils_partx import initial_theta_estimate

def load_tree(tree_name):
    """Load the tree

    Args:
        tree_name ([type]): Load a tree for a particular replication

    Returns:
        [type]: tree
    """
    f = open(tree_name, "rb")
    ftree = pickle.load(f)
    return ftree

def falsification_volume(ftree, options):
    """Calculate Falsification Volume Using the classified and unclassified regions

    Args:
        ftree ([type]): ftree
        options ([type]): initialization options

    Returns:
        [type]: volumes of classified and unclassified regions
    """
    leaves = ftree.leaves()
    region_supports_classified = []
    region_supports_unclassified = []
    for x,i in enumerate(leaves):
        node_data = i.data
        if node_data.region_class == "-":
            region_supports_classified.append(node_data.region_support)
        if node_data.region_class == "r" or node_data.region_class == "r+" or node_data.region_class == "r-" or node_data.region_class == "-":
            region_supports_unclassified.append(node_data.region_support)

    falsified_volume_count_classified = len(region_supports_classified)
    region_supports_classified = np.reshape(np.array(region_supports_classified), (falsified_volume_count_classified,options.test_function_dimension, 2))
    volumes_classified = calculate_volume(region_supports_classified)

    falsified_volume_count_unclassified = len(region_supports_unclassified)
    region_supports_unclassified = np.reshape(np.array(region_supports_unclassified), (falsified_volume_count_unclassified,options.test_function_dimension, 2))
    volumes_unclassified = calculate_volume(region_supports_unclassified)
    return np.sum(volumes_classified), np.sum(volumes_unclassified)

def falsification_volume_using_gp(ftree, options, quantiles_at, rng):
    """Calculate falsification volume using GP 

    Args:
        ftree ([type]): [description]
        options ([type]): [description]
        quantiles_at ([type]): [description]
        rng ([type]): [description]

    Returns:
        [type]: [description]
    """
    leaves = ftree.leaves()
    region_supports = []
    falsification_volumes = []
    for iterate,temp_node_id in enumerate(leaves):
        
        node_data = temp_node_id.data
        X = node_data.samples_in[0]
        Y = np.transpose(node_data.samples_out)
        model = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            normalize_y=True,
            alpha=1e-6,
            n_restarts_optimizer=5,
            optimizer = optimizer_lbfgs_b
            )
            
        model.fit(X, Y)
        quantile_values_r= []
        for r in range(options.R):
            samples = uniform_sampling(options.M, node_data.region_support, options.test_function_dimension, rng)
            y_pred, sigma_st = model.predict(samples[0], return_std=True)
            quantile_values_m = []
            for x in range(options.M):
                quantiles_values_alp = []
                for alp in quantiles_at:
                    
                    quantiles_values = (stats.norm.ppf(alp,y_pred[x][0],sigma_st[x]))
                    # print(quantiles_values)
                    quantiles_values_alp.append(quantiles_values)
                quantile_values_m.append(quantiles_values_alp)
            quantile_values_r.extend(quantile_values_m)
        falsified_volume_region = ((np.array(quantile_values_r) < 0).sum(axis=0) / (options.R*options.M)) * calculate_volume(node_data.region_support)
        falsification_volumes.append(falsified_volume_region)
        # print("{} of {} done".format(iterate, len(leaves)))
    # print(np.sum(np.array(falsification_volumes),axis=0))
    return np.array(falsification_volumes)

def con_int(x, conf_at):
    """Calculate COnfidence interval

    Args:
        x ([type]): [description]
        conf_at ([type]): [description]

    Returns:
        [type]: [description]
    """
    mean, std = x.mean(), x.std(ddof=1)
    conf_intveral = stats.norm.interval(conf_at, loc=mean, scale=std)
    return conf_intveral