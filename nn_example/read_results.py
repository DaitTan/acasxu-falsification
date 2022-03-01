import pathlib
import pickle
from partx.models.partx_options import partx_options
from partx.executables.generate_statistics import generate_statistics
import matplotlib.pyplot as plt
import numpy as np
# from read_UR_results import read_UR_results
import logging
from partx.executables.exp_statistics import falsification_volume_using_gp
import re, csv, itertools
from partx.executables.exp_statistics import falsification_volume_using_gp


# BENCHMARK_NAMES = ["f16_alt2300_budget_5000",
#                    "f16_alt2330_budget_5000",
#                    "f16_alt2338_budget_5000",
#                    "f16_alt2338_2_budget_5000",
#                    "f16_alt2338_4_budget_5000",
#                    "f16_alt2338_5_budget_5000",
#                    "f16_alt2338_6_budget_5000",
#                    "f16_alt2350_budget_5000",
                #    "f16_alt2400_budget_5000"]

BENCHMARK_NAMES = ["property_7_N_1_9_exp_4"]

log = logging.getLogger()
log.setLevel(logging.INFO) 
fh = logging.FileHandler(filename = pathlib.Path().joinpath("AT_res_log.log"))
formatter = logging.Formatter(
                fmt = '%(message)s'
                )

fh.setFormatter(formatter)
log.addHandler(fh)
UR_BENCHMARK_NAME = "property_7_N_1_9_UR_exp_4"
UR_result_directory = pathlib.Path().joinpath('acas_xu_testing').joinpath(UR_BENCHMARK_NAME).joinpath(UR_BENCHMARK_NAME + "_result_generating_files")
f = open(UR_result_directory.joinpath(UR_BENCHMARK_NAME+"_0_uniform_random_results.pkl"), "rb")
UR_files = pickle.load(f)
f.close()
print(np.mean(UR_files['robustness']))
print(UR_files['samples'][0][np.argmin(UR_files['robustness'])])
print(UR_files['samples'][0][np.argmax(UR_files['robustness'])])

for BENCHMARK_NAME in BENCHMARK_NAMES:    
    
    result_directory = pathlib.Path().joinpath('acas_xu_testing').joinpath(BENCHMARK_NAME).joinpath(BENCHMARK_NAME + "_result_generating_files")
    print(result_directory)
    f = open(result_directory.joinpath(BENCHMARK_NAME + "_options.pkl"), "rb")
    options = pickle.load(f)
    f.close()

    
    print(vars(options))
    number_of_macro_replications = 1
    for i in range(number_of_macro_replications):
        f = open(result_directory.joinpath(BENCHMARK_NAME+"_" + str(i)+"_point_history.pkl"), "rb")
        point_history = pickle.load(f)
        f.close()
        print(np.array(point_history)[:,-1])
    
    # print(result_dictionary)
    # for i in range(number_of_macro_replications):
    #     f = open(result_directory.joinpath(BENCHMARK_NAME+ "_" + str(i) + ".pkl"), "rb")
    #     ftree = pickle.load(f)
    #     f.close()
    #     rng = np.random.default_rng(1000+i)
    #     falsification_volume_arrays = falsification_volume_using_gp(ftree, options, options.fv_quantiles_for_gp, rng)
            
    #     f = open(result_directory.joinpath(BENCHMARK_NAME + "_" + str(number_of_macro_replications) + "_fal_val_gp.pkl"), "wb")
    #     pickle.dump(falsification_volume_arrays,f)
    #     f.close()
    #     print(falsification_volume_arrays)
    #     # f = open(benchmark_result_pickle_files.joinpath(BENCHMARK_NAME + "_" + str(replication_number) + "_point_history.pkl"), "wb")
    #     # f = open(result_directory.joinpath(BENCHMARK_NAME + "_" + str(number_of_macro_replications) + "_point_history.pkl.pkl"), "wb")
    #     # pickle.dump(callCounts.point_history,f)
    #     # f.close()
    #     print("********************")

    quantiles_at = [0.999, 0.95, 0.99]
    
    confidence_at = 0.95
    result_dictionary = generate_statistics(BENCHMARK_NAME, number_of_macro_replications, quantiles_at, confidence_at,'acas_xu_testing')
    print("**************************")
    with open('dict.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in result_dictionary.items():
            writer.writerow([key, str(value)])
    print("**************************")
    #     falsification_volume_using_gp(ftree, options, quantiles_at, rng)

    # for key, value in result_dictionary.items():
    #     print("{} : {}".format(key, value))

    
    
    FR, mean_fv, std_error_fv, con_int_0, con_int_1, best_rob = read_UR_results(BENCHMARK_NAME, confidence_at, number_of_macro_replications)
    # result_dictionary['falsification_rate']
    log.info("{};{};{};{};{};{};{};{};{};{};{}".format(BENCHMARK_NAME,
                                result_dictionary['falsification_rate'],
                                result_dictionary['mean_fv_with_gp_quan0_5'],
                                result_dictionary['std_dev_fv_with_gp_quan0_5'],
                                result_dictionary['con_int_fv_with_gp_quan_0_5_confidence_0_95'][0],
                                result_dictionary['con_int_fv_with_gp_quan_0_5_confidence_0_95'][1],
                                result_dictionary['numpoints_fin_first_f_mean'],
                                result_dictionary['numpoints_fin_first_f_median'],
                                result_dictionary['numpoints_fin_first_f_min'],
                                result_dictionary['numpoints_fin_first_f_max'],
                                result_dictionary['best_robustness']))