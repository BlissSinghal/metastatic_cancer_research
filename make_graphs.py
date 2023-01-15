import matplotlib.pyplot as plt
import make_component_images as comp_img
import _make_components as m_components
import _make_components_2 as m_components_2
import os
import numpy as np

def make_graphs_dataset_sizes(dataset_size, num_components = 25):
    neighbors_scores, log_scores, stoch_grad_scores, dataset_sizes = comp_img.make_component_images(dataset_size, num_components)
    make_graphs("Graphs", "Accuracy with Different Dataset Sizes", "Dataset Sizes", "dataset_sizes_2", dataset_sizes, neighbors_scores, log_scores, stoch_grad_scores)

def make_graphs_components(num_components):
    n_scores, log_scores, stoch_scores, components = get_accuracy_components(num_components)
    make_graphs("Graphs", "Accuracy with Different Component Sizes", "Components", "Components_2", components, n_scores, log_scores, stoch_scores)
def make_graphs(folder_name, title, x_label, image_name, x_vals, n_scores, l_scores, stoch_scores):
    path = "_results/" + folder_name
    os.makedirs(f"{path}", exist_ok=True)
    fig = plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    
    plt.ylabel("Accuracy Scores")
    plt.plot(x_vals, n_scores, color = "blue", label = "Nearest Neighbors")
    plt.plot(x_vals, l_scores, color = "red", label = "Logistic Regression")
    plt.plot(x_vals, stoch_scores, color = "green", label = "Decision Tree")
    plt.legend()
    plt.savefig(f"{path}/{image_name}_plot.png")
def make_bar_graph_ga(title, x_label, y_label, x_vals, scores1, scores2, scores3):
    path = "_results/Graphs"
    fig = plt.figure()
    colors = ["b", "r"]
    labels = ["Nearest Neighbors", "Logistic Regression", "Decision Tree"]
    pos = np.arange(len(labels))
    plt.title(title)
    plt.xticks(pos, labels)
    plt.ylim(0, 0.71)
    bar_width = 0.35
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.bar(pos, [scores1[0], scores2[0], scores3[0]], bar_width, color = colors[0], label = x_vals[0])
    plt.bar(pos+bar_width, [scores1[1], scores2[1], scores3[1]], bar_width, color = colors[1], label = x_vals[1])
    """
    plt.bar(pos, scores2[0], bar_width, color = colors[0])
    plt.bar(pos+bar_width, scores2[1], bar_width, color = colors[1])
    plt.bar(pos, scores3[0], bar_width, color = colors[0])
    plt.bar(pos+bar_width, scores3[1], bar_width, color = colors[1])
    """
    print(scores1)
    print(scores2)
    plt.legend()
    plt.savefig(f"{path}/{x_label}")
    
def make_bar_graph(title, x_label, y_label, scores):
    fig = plt.figure()
    colors = ["b", "r", "g"]
    labels = ["Nearest Neigbors", "Logistic Regression", "Decision Tree"]
    pos = np.arange(len(labels))
    plt.title(title)
    plt.xticks(pos, labels)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.bar(pos, scores, color = colors, label = labels)
    plt.savefig(f"_results/Graphs/most_optimized.png")
    
    
def _get_neighbors_scores(max_neighbors):
    neighbors_scores = []
    num_neighbors = []
    y, y_transformed_pca, transformed_train, pca = m_components_2.do_PCA(data_size = 6250, num_components= 250)
    while max_neighbors > 0:
        neighbors_score = m_components_2.make_components(y, y_transformed_pca, transformed_train, pca, data_size = 6250, num_components = 250, num_neighbors = max_neighbors, preprocessing_method = "ga")
        neighbors_scores.append(neighbors_score)
        num_neighbors.append(max_neighbors)
        print("Neighbors: " + str(max_neighbors))
        max_neighbors = int(max_neighbors/2)
    return neighbors_scores, num_neighbors

def make_neighbors_graph(max_neighbors):
    neighbors_scores, num_neighbors = _get_neighbors_scores(max_neighbors)
    print(neighbors_scores)
    fig = plt.figure()
    plt.title("Nearest Neighbors Accuracy with Different Numbers of Nearest Neighbors")
    plt.xlabel("Number of Nearest Neighbors")
    plt.ylabel("Accuracy Score")
    plt.plot(num_neighbors, neighbors_scores)
    path = "_results/Graphs/neighbors_results.png"
    plt.savefig(path)

def get_accuracy_components(num_components):
    n_scores = []
    log_scores = []
    stoch_scores = []
    components_array = []
    while num_components > 0:
        n_score, l_score, stoch_score = m_components.make_components(num_components = num_components)
        n_scores.append(n_score)
        log_scores.append(l_score)
        stoch_scores.append(stoch_score)
        components_array.append(num_components)
        num_components = int(num_components / 2)
    return n_scores, log_scores, stoch_scores, components_array
        



    

    

    
