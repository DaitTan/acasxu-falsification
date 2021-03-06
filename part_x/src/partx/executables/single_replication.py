from ..numerical.classification import calculate_volume
from ..utilities.utils_partx import assign_budgets, branch_new_region_support, pointsInSubRegion, plotRegion
from ..models.testFunction import callCounter
from ..models.partx_node import partx_node
from ..models.partx_options import partx_options
import numpy as np
from ..numerical.classification import calculate_volume
import matplotlib.pyplot as plt
from ..numerical.budget_check import budget_check
from treelib import Tree
from ..numerical.calIntegral import calculate_mc_integral
import logging
import pickle
from .exp_statistics import falsification_volume_using_gp

def run_single_replication(inputs):
    replication_number, options, test_function, benchmark_result_directory = inputs

    seed = options.start_seed + replication_number
    BENCHMARK_NAME = options.BENCHMARK_NAME

    benchmark_result_log_files = benchmark_result_directory.joinpath(BENCHMARK_NAME + "_log_files")
    benchmark_result_log_files.mkdir(exist_ok=True)

    benchmark_result_pickle_files = benchmark_result_directory.joinpath(BENCHMARK_NAME + "_result_generating_files")
    benchmark_result_pickle_files.mkdir(exist_ok=True)

    callCounts = callCounter(test_function)
    
    log = logging.getLogger()
    log.setLevel(logging.INFO) 
    fh = logging.FileHandler(filename=benchmark_result_log_files.joinpath(BENCHMARK_NAME + "_" + str(replication_number) + ".log"))
    formatter = logging.Formatter(
                    fmt = '%(asctime)s :: %(message)s', datefmt = '%a, %d %b %Y %H:%M:%S'
                    )

    fh.setFormatter(formatter)
    log.addHandler(fh)
    log.info("Information about Replication {}".format(replication_number))
    log.info("Running {} Replication {} with seed {}".format(BENCHMARK_NAME, replication_number, seed))
    log.info("**************************************************")
    log.info("Options File:")
    options_results = vars(options)
    for key, value in options_results.items():
        log.info("{} : {}".format(key, value))
    log.info("**************************************************")
    log.info("Budget Used = {}".format(callCounts.callCount))
    log.info("Budget Available (Max Budget) = {}".format(options.max_budget))
    log.info("**************************************************")
    log.info("**************************************************")
    log.info("***************Replication Start******************")
    print("Started replication {}".format(replication_number))

    rng = np.random.default_rng(seed)

    if options.max_budget < options.initialization_budget+options.number_of_BO_samples[0]:
        log.info("Error: Cannot Initialize root node")
        raise Exception("(Max Budget) MUST NOT BE LESS THAN (Initialization_budget + number_of_BO_samples)")
# Sampling Initialization
    samples_in = np.array([[[]]])
    samples_out = np.array([[]])
    indices_branching_direction = np.arange(options.test_function_dimension)
    direction = rng.permutation(indices_branching_direction)
    
    direction_of_branch = 0



    # define root node
    root = partx_node(options.initial_region_support, samples_in, samples_out, direction_of_branch)
    root.generate_samples_points_grid(options, rng)
    samples_in, samples_out = root.samples_management_unclassified(options, callCounts, rng)
    region_class = root.calculate_and_classifiy(options,rng)

    id = 0
    remaining_regions_list = []
    classified_regions_list = []
    unidentified_regions_list = []
    if region_class == 'r' or region_class == 'r+' or region_class == 'r-':
        remaining_regions_list.append(id)
    elif region_class == '+' or region_class == '-':
        classified_regions_list.append(id)
    elif region_class == 'u':
        unidentified_regions_list.append(id)

    log.info("**************************************************")
    log.info("Classified regions = {}".format(len(classified_regions_list)))
    log.info("Unclassified regions = {}".format(len(remaining_regions_list)))
    log.info("Unidentified regions = {}".format(len(unidentified_regions_list)))
    log.info("Budget available for replication = {}".format(options.max_budget - callCounts.callCount))
    log.info("**************************************************")

    # Tree initialization using root
    ftree = Tree()
    ftree.create_node(id,id,data=root)

    # print(len(remaining_regions_list))
    while budget_check(options, callCounts.callCount, remaining_regions_list):
        # print(len(remaining_regions_list))
        tempQueue = remaining_regions_list.copy()
        remaining_regions_list = []

        for temp_node_id in tempQueue:
            node = ftree.get_node(temp_node_id)
            parent = node.identifier
            node = node.data
            new_bounds = branch_new_region_support(node.region_support, direction[node.direction_of_branch % options.test_function_dimension], options.uniform_partitioning, options.branching_factor, rng)
            points_division_samples_in, points_division_samples_out = pointsInSubRegion(node.samples_in, node.samples_out, new_bounds)
            for iterate in range(new_bounds.shape[0]):
                # print("parent = {} ------> id{}".format(parent,id))
                id = id+1
                new_region_supp = np.reshape(new_bounds[iterate], (1,new_bounds[iterate].shape[0],new_bounds[iterate].shape[1]))

                new_node = partx_node(new_region_supp, points_division_samples_in[iterate], points_division_samples_out[iterate], (node.direction_of_branch+1), node.region_class)
                new_node.generate_samples_points_grid(options, rng)
                samples_in, samples_out = new_node.samples_management_unclassified(options, callCounts, rng)
                region_class = new_node.calculate_and_classifiy(options,rng)
                ftree.create_node(id, id, parent = parent, data = new_node)
                
                if region_class == 'r' or region_class == 'r+' or region_class == 'r-':
                    remaining_regions_list.append(id)
                elif region_class == '+' or region_class == '-':
                    classified_regions_list.append(id)
                elif region_class == 'u':
                    unidentified_regions_list.append(id)
        log.info("**************************************************")
        log.info("Mid Classified regions = {}".format(len(classified_regions_list)))
        log.info("Mid Unclassified regions = {}".format(len(remaining_regions_list)))
        log.info("Mid Unidentified regions = {}".format(len(unidentified_regions_list)))
        
        if len(classified_regions_list) != 0:

            tempQueue = classified_regions_list.copy()
            classified_regions_list = []
            volumes = []

            for temp_node_id in tempQueue:
                node = ftree.get_node(temp_node_id)
                parent = node.identifier
                node = node.data
                cdf_sum = calculate_mc_integral(node.samples_in, node.samples_out, node.region_support, options.test_function_dimension, options.R, options.M, rng)
                # volumes.append((cdf_sum * calculate_volume(node.region_support)[0]) + abs(rng.normal(options.nugget_mean, options.nugget_std_dev))) # + nugget
                volumes.append((cdf_sum * calculate_volume(node.region_support)[0])) # + nugget

            if np.sum(volumes) != 0.0:
                volume_distribution = volumes/np.sum(volumes)
            else:
                volume_distribution = volumes
            
            n_cont_sampling_budget_assignment = assign_budgets(volume_distribution, options.continued_sampling_budget, rng)

            for iterate, temp_node_id in enumerate(tempQueue):
                if n_cont_sampling_budget_assignment[iterate] != 0:
                    node = ftree.get_node(temp_node_id)
                    parent = node.identifier
                    node = node.data
                    
                    samples_in, samples_out = node.samples_management_classified(options, callCounts, n_cont_sampling_budget_assignment[iterate], rng)

                    region_class = node.calculate_and_classifiy(options, rng)
                    
                    if region_class == 'r' or region_class == 'r+' or region_class == 'r-':
                        remaining_regions_list.append(parent)
                    elif region_class == '+' or region_class == '-':
                        classified_regions_list.append(parent)
                    elif region_class == 'u':
                        unidentified_regions_list.append(parent)
                else:
                    node = ftree.get_node(temp_node_id)
                    parent = node.identifier
                    region_class = node.data.region_class
                    
                    if region_class == '+' or region_class == '-':
                        classified_regions_list.append(parent)
                    elif region_class == 'u':
                        unidentified_regions_list.append(parent)
        log.info("Classified regions = {}".format(len(classified_regions_list)))
        log.info("Unclassified regions = {}".format(len(remaining_regions_list)))
        log.info("Unidentified regions = {}".format(len(unidentified_regions_list)))
        log.info("Budget available for replication = {}".format(options.max_budget - callCounts.callCount))
        log.info("**************************************************")
        # print(len(remaining_regions_list)!=0)
    budget_available = options.max_budget - callCounts.callCount
    if budget_available >= 0:
        log.info("**************************************************")
        log.info("In the last act for replication {}:".format(replication_number))
        log.info("Budget Used for replication {} = {}".format(replication_number, callCounts.callCount))
        log.info("Budget available for replication {} = {}".format(replication_number, budget_available))
        log.info("Classified regions = {}".format(len(classified_regions_list)))
        log.info("Unclassified regions = {}".format(len(remaining_regions_list)))
        log.info("Unidentified regions = {}".format(len(unidentified_regions_list)))
        log.info("**************************************************")

        tempQueue = classified_regions_list + remaining_regions_list
        if len(tempQueue) != 0:
            remaining_regions_list = []
            classified_regions_list = []
            volumes = []
            for temp_node_id in tempQueue:
                node = ftree.get_node(temp_node_id)
                parent = node.identifier
                node = node.data
                cdf_sum = calculate_mc_integral(node.samples_in, node.samples_out, node.region_support, options.test_function_dimension, options.R, options.M,rng)
                # volumes.append((cdf_sum * calculate_volume(node.region_support)[0]) #+ abs(rng.normal(options.nugget_mean, options.nugget_std_dev))) # + nugget
                volumes.append((cdf_sum * calculate_volume(node.region_support)[0]))

            volume_distribution = volumes/np.sum(volumes)
            n_cont_sampling_budget_assignment = assign_budgets(volume_distribution,budget_available,rng)

            for iterate, temp_node_id in enumerate(tempQueue):
                if n_cont_sampling_budget_assignment[iterate] != 0:
                    node = ftree.get_node(temp_node_id)
                    parent = node.identifier
                    node = node.data

                    node.generate_samples_points_grid(options, rng)
                    samples_in, samples_out = node.samples_management_classified(options, callCounts, n_cont_sampling_budget_assignment[iterate],rng)

                    region_class = node.calculate_and_classifiy(options,rng)
                    
                    if region_class == 'r' or region_class == 'r+' or region_class == 'r-':
                        remaining_regions_list.append(parent)
                    elif region_class == '+' or region_class == '-':
                        classified_regions_list.append(parent)
                    elif region_class == 'u':
                        unidentified_regions_list.append(parent)
                else:
                    node = ftree.get_node(temp_node_id)
                    parent = node.identifier
                    region_class = node.data.region_class
                    
                    if region_class == 'r' or region_class == 'r+' or region_class == 'r-':
                        remaining_regions_list.append(parent)
                    elif region_class == '+' or region_class == '-':
                        classified_regions_list.append(parent)
                    elif region_class == 'u':
                        unidentified_regions_list.append(parent)

    budget_available = options.max_budget - callCounts.callCount
    log.info("**************************************************")
    log.info("**************************************************")
    log.info("ENDING replication {}:".format(replication_number))
    log.info("Budget Used = {}".format(callCounts.callCount))
    log.info("Budget available = {}".format(budget_available))
    log.info("**************************************************")
    log.info("**************************************************")
    log.info("****************Replication END*******************")

    f = open(benchmark_result_pickle_files.joinpath(BENCHMARK_NAME+ "_" + str(replication_number) + ".pkl"), "wb")
    pickle.dump(ftree,f)
    f.close()

    print("Generating Result Files for replication Number {}".format(replication_number))
    falsification_volume_arrays = falsification_volume_using_gp(ftree, options, options.fv_quantiles_for_gp, rng)
            
    f = open(benchmark_result_pickle_files.joinpath(BENCHMARK_NAME + "_" + str(replication_number) + "_fal_val_gp.pkl"), "wb")
    pickle.dump(falsification_volume_arrays,f)
    f.close()

    f = open(benchmark_result_pickle_files.joinpath(BENCHMARK_NAME + "_" + str(replication_number) + "_point_history.pkl"), "wb")
    pickle.dump(callCounts.point_history,f)
    f.close()

    print("Ended replication {}".format(replication_number))
    log.removeHandler(fh)
    del log, fh

    return {
        'ftree': ftree,
        'classified_regions_list': classified_regions_list,
        'remaining_regions_list': remaining_regions_list,
        'unidentified_regions_list': unidentified_regions_list
    }