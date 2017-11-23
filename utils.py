import pickle
import os

def save_result(result, exp_result_dir, data_name):
    with open(os.path.join(exp_result_dir, data_name) + '.obj', 'wb') as f:
        pickle.dump(result, f)

def load_result(exp_result_dir, data_name):
    with open(os.path.join(exp_result_dir, data_name) + '.obj', 'rb') as f:
        result = pickle.load(f)
    return result

def average_results(exp_result_dir, data_names):
    results = [load_result(exp_result_dir, data_name) for data_name in data_names]
    disc_method = results[0]['disc_method']
    avg_result = sum([result.drop('disc_method', 1) for result in results]) / len(results)
    avg_result['disc_method'] = disc_method
    return avg_result