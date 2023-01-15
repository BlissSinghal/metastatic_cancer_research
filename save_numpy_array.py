import numpy as np
import os
from pathlib import Path
import csv
def save_numpy_array(array, file_name):
    path = f"_results/Numpy_Arrays/{file_name}.npy"
    np.save(path, array)
def get_numpy_array(file_name):
    data = np.load(f"_results/Numpy_Arrays/{file_name}.npy", allow_pickle=True)
    return data
def get_array_csv_file(file_name):
    path = Path("_results/Numpy_Arrays")
    lines = (path/f"{file_name}.csv").read_text().splitlines()
    lin_model_data = []
    neighbor_model_data = []
    tree_model_data = []
    for line in lines:
        n_model, lin_model, tree_model = line.split(",")
        lin_model_data.append(float(lin_model))
        neighbor_model_data.append(float(n_model))
        tree_model_data.append(float(tree_model))
    #test_set = dict(parse_labels(lines[10001: 20001]))
    return neighbor_model_data, lin_model_data, tree_model_data
def split_array_diff_models(array):
    lin_model = []
    neighbor_model = []
    tree_model = []
    for index in range(len(array)-1):
        print(array[index][0])
        """
        neighbor_model.append(array[index][0])
        lin_model.append(array[index][1])
        tree_model.append(array[index][2])
        """
        
    return lin_model, tree_model, neighbor_model
def get_model_data(file_name):
   return get_array_csv_file(file_name)
    
def write_results(file_name, array):
    f = open(f"_results/Numpy_Arrays/{file_name}.csv", "w")
    writer = csv.writer(f)
    for row in array: 
        writer.writerow(row)
    f.close()

def get_dataset_array(file_name):
    path = Path("_results/Numpy_Arrays")
    lines = (path/f"{file_name}.csv").read_text().splitlines()
    final_results = []
    for line in lines:
        results = []
        for element in line.split(","):
            if element != "ga" and element != "pca":
                results.append(float(element))
            else:
                results.append(element)
        final_results.append(results)
    return final_results



    