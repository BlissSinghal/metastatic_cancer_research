import matplotlib.pyplot as plt


import imageio
import os
from os import listdir

def get_fig_1(data_size=50000, num_components=50):
    print("finding what dataset size gives the best result")
    """
    num_components_table = 1000
    results = [["number of components: "], ["dataset size"]]
    """
    fig = plt.figure(figsize = (12, 8))
    rows = 5
    index = 1
    for count in range(rows): 
        """
        results[0].append(num_components)
        results[1].append(" ")
        """
        folder_name = "Dataset_" + str(data_size) + "_Component_Images"
        
        component_title = 1
        for image_name in listdir(f"_results/{folder_name}")[:num_components]:
            fig.add_subplot(rows, num_components, index)
            if (component_title == 1):
                plt.ylabel(f"{data_size} samples")
            plt.title(f"PC {component_title}")
            image = imageio.imread(f"_results/{folder_name}/{image_name}")
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])
            #plt.axis("off")
            index+=1
            component_title+=1
            print(index)
        data_size = int(data_size/4)
        
    os.makedirs(f"_results/reconstructed_images", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"_results/reconstructed_images/principal_components.png", dpi=400)
    #make_tables(results)
    print()