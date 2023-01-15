import _make_components as m_components
import save_numpy_array as Save

def get_ga_result(pop_size, num_gens):
    neighbor_score, log_score, stoch_score = m_components.make_components(preprocessing_method = "genetic algorithms", iteration = num_gens, population_size = pop_size)
    return [neighbor_score, log_score, stoch_score]

def get_ga_results(max_pop_size, max_num_gens, file_name = "array"):
    final_results = []
    while max_pop_size > 1 and max_num_gens > 1:
        result = get_ga_result(max_pop_size, max_num_gens)
        print(result)
        results = []
        results.append([max_pop_size, max_num_gens])
        results.append(result)
        final_results.append(results)
        max_pop_size = int(max_pop_size/1.5)
        max_num_gens = int(max_num_gens/1.5)
    Save.save_numpy_array(final_results, file_name)
    return final_results
def get_ga_results_processing(file_name = "dataset_size_ga"):
    log_scores = []
    n_scores = []
    tree_scores = []
    x_vals = ["ga", "pca"]
    for process in x_vals:
        n_score, log_score, tree_score = get_data_ga(process)
        print(n_score)
        n_scores.append(n_score)
        log_scores.append(log_score)
        tree_scores.append(tree_score) 
    final_array = [x_vals, n_scores, log_scores, tree_scores]
    Save.save_numpy_array(final_array, file_name)
    Save.write_results(file_name, final_array)
    return final_array
def get_data_datasets_size(dataset_size):
    return m_components.make_components(data_size = dataset_size, preprocessing_method = "ga")
def get_data_ga(processing_method):
    return m_components.make_components(preprocessing_method = processing_method)
def get_data_num_components(num_components):
    return m_components.make_components(num_components= num_components, preprocessing_method = "ga")
def get_ga_results_components(dataset_size, decrement, file_name = "dataset_size_ga"):
    log_scores = []
    n_scores = []
    tree_scores = []
    dataset_sizes = []
    while dataset_size > 0:
        dataset_sizes.append(dataset_size)
        n_score, log_score, tree_score = get_data_num_components(dataset_size)
        print(n_score)
        n_scores.append(n_score)
        log_scores.append(log_score)
        tree_scores.append(tree_score)
        dataset_size -= decrement
    final_array = [dataset_sizes, n_scores, log_scores, tree_scores]
    Save.save_numpy_array(final_array, file_name)
    Save.write_results(file_name, final_array)
    return final_array