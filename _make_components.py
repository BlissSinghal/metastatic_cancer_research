from pathlib import Path
import imageio as img
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import os
from os import listdir
import matplotlib.pyplot as plt
from itertools import islice
import csv
from sklearn import metrics
import hiplot as hip
from load_data import load_data
from get_training_data import get_training_data
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import genetic_algorithm as gen_algo
#from library.genetic_algorithm import run_genetic_algorithm


def test_function()->None:
    output_path = Path("_results/test_function.txt")
    output_path.parent.mkdir(parents= True, exist_ok = True)
    output_path.write_text("test function")



def get_vectors(image_array):
    dim1, dim2, dim3, dim4 = image_array.shape
    #dim1 represents nunber of images, so we want to keep that
    image_array = image_array.reshape(dim1, (dim2*dim3*dim4))
    return image_array

def get_pca(training_set, num_components):
    #principal component analysis reduces the dimensionality of arrays without losing information
    #transforms a large set of variables to a smaller one without losing info
    
    pca = PCA(n_components=num_components)
    pca.fit(training_set)
    
    #gets the transformed array of vectors and analyzes that and compares it to the analysis of principal components
    #note that most programs that I saw using pca to preprocess the data fed the transformed dataset into the ML model, not the pca.components dataset
    return pca

def get_transformed_vector(pca, transforming_set):
   
    dim_reduced_train = np.array(pca.transform(transforming_set))
    return dim_reduced_train

def get_principal_components(pca):
    return pca.components_

def reconstruct_image(principal_components, folder_name, mid_shape, file_names):
    principal_components = principal_components.reshape(len(principal_components), mid_shape, mid_shape, 3)
    principal_components -= principal_components.min()
    principal_components /= principal_components.max()
    principal_components *= 255
    principal_components = np.uint8(principal_components)
    
    file_names_1, file_names_2 = np.array_split(file_names, 2)
    for i in range(len(principal_components)):
        os.makedirs(f"_results/{folder_name}", exist_ok=True)
        file_name, _ = file_names_2[i].split(".")
        img.imwrite(f"_results/{folder_name}/{file_name}.png", principal_components[i])
  
def make_hiplot(transformed_train, y_vals):
    data = []
    #converting the transformed_train array and the y_vals array into dictionarys, with each dictionary representing one image
    for image_index in range(len(transformed_train)):
        dictionary = {"image": image_index}
        for component_index in range(len(transformed_train[image_index])):
            key = "Principal Component " + str(component_index)
            value = transformed_train[image_index][component_index]
            dictionary[key] = value
        dictionary["Cancer"] = y_vals[image_index]
        data.append(dictionary)
    hip.Experiment.from_iterable(data).display()


def make_subplots(transformed_train, folder_name, label, y_vals):
  
    #makes plots that look at two components of a given array of components

    num_images, principal_components = transformed_train.shape
    os.makedirs(folder_name, exist_ok=True)
    #iterating through each principal component
    for count in range(principal_components-1):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel(label + str(count+1))
        ax.set_ylabel(label + str(count))
        data = np.transpose(transformed_train)
        ax.scatter(data[count], data[count+1], c=y_vals, cmap = "PiYG")
        plt.savefig(f"_results/{folder_name}/{count}.png")
        plt.close()    

def get_variance_ratio(pca):
    return pca.explained_variance_ratio_


def reorder_labels(labels, image_name):
    y_vals = []
    for image in image_name: 
        index, end = image.split(".")
        y_vals.append(int(labels[index]))
    return y_vals

def predict_nearest_neighbors(data_x, data_y, x_predict, y_true):
    nn_model = KNeighborsClassifier(n_neighbors=2)
    nn_model.fit(data_x, data_y)
    y_preds = nn_model.predict(x_predict)
    accuracy = metrics.accuracy_score(y_true, y_preds)
    return accuracy

def predict_logistic_regression(data_x, data_y, predict_x, y_true):
    log_model = LogisticRegression()
    log_model.fit(data_x, data_y)
    y_preds = log_model.predict(predict_x)
    accuracy = metrics.accuracy_score(y_true, y_preds)
    return accuracy

def predict_decision_tree(data_x, data_y, predict_x, y_true):
    lin_model = DecisionTreeClassifier()
    lin_model.fit(data_x, data_y)
    y_preds = lin_model.predict(predict_x)
    accuracy = metrics.accuracy_score(y_true, y_preds)
    return accuracy

def make_components(fig=0, folder_name=0, data_size = 600, num_components = 15, num_neighbors = 2, iteration = 29, preprocessing_method = "pca", population_size = 29, testing_size= 1000):
    labels = load_data()
    image_names, array = get_training_data(data_size*2+testing_size)
    y_vals = np.array(reorder_labels(labels, image_names))
    y = y_vals[:data_size]
    y_vals_transformed_train = y_vals[data_size:]

    vectors = get_vectors(array)
    training_set_pca = vectors[:data_size]
    pca = get_pca(training_set_pca, num_components)

    transformed_train = get_transformed_vector(pca, vectors[data_size:])
    if preprocessing_method == "pca":
        components = get_principal_components(pca)
    else:
        components = gen_algo.run_genetic_algorithm(transformed_train, y_vals_transformed_train, population_size=population_size, max_iterations = iteration)

    #total number of possible components is around 27000
    y_train = y_vals_transformed_train[:(len(y_vals_transformed_train)-testing_size)]
    y_true = y_vals_transformed_train[(len(y_vals_transformed_train)-testing_size):]
    x_train = transformed_train[: len(transformed_train)-testing_size]
    x_test = transformed_train[len(transformed_train)-testing_size:]
       

    #print("Nearest neighbors score: " + str(predict_nearest_neighbors(x_train, y_train, x_test, y_true)))
    #print("Logistic regression score: " + str(predict_logistic_regression(x_train, y_train, x_test, y_true)))
    #print("Linear Regression score: " + str(predict_linear_regression(x_train, y_train, x_test, y_true)))
 
    #make_subplots(transformed_train, "train_small_plots", "Principal Component ", y_vals_transformed_train)
    #make_hiplot(transformed_train, y_vals_transformed_train) 
    #len_components = len(components)
 
    if fig != 0: 
        reconstruct_image(components, folder_name, 96, image_names)
    return [predict_nearest_neighbors(x_train, y_train, x_test, y_true), predict_logistic_regression(x_train, y_train, x_test, y_true), predict_decision_tree(x_train, y_train, x_test, y_true)]
    
 
#print("Genetic Algorithm: " + str(make_components(preprocessing_method="genetic algorithm")))
#print("PCA:" + str(make_components()))
