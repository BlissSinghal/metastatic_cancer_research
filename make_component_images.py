import _make_components as m_component
import matplotlib.pyplot as plt
import shutil
def make_component_images(dataset_size, num_components):
    fig = plt.figure(figsize=(12, 8))
    neighbors_scores = []
    log_scores = []
    stoch_grad_scores = []
    path = "_results"
    dataset_sizes = []
    while dataset_size > 96:
        folder_name = "Dataset_" + str(dataset_size) + "_Component_Images"
        complete_path = f"{path}/{folder_name}"
        #clearing the folder first
        shutil.rmtree(complete_path)
        neighbors_score, log_score, stoch_grad_score = m_component.make_components(fig = fig, folder_name = folder_name, data_size = dataset_size, num_components = num_components, preprocessing_method = "pca")
        neighbors_scores.append(neighbors_score)
        log_scores.append(log_score)
        stoch_grad_scores.append(stoch_grad_score)
        dataset_sizes.append(dataset_size)
        dataset_size = int(dataset_size/2)
    return neighbors_scores, log_scores, stoch_grad_scores, dataset_sizes