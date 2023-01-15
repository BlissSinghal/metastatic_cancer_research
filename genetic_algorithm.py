import random
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np

"""
this algorithm is an alternative to PCA and also gets the combonation of features in the dataset that gives the best accuracy when fed into a ML model
"""
def get_chromosome(vector):
    #gets a random chromosome with random genes
    #only used to generate inital population
    chromosome = []
    for feature in vector:
        gene = int(random.randint(0, 1))
        chromosome.append(gene)
    return chromosome

def get_gen_0(vectors, num_chromosomes= 25):
    #gets an initial population of chromosomes
    gen_0 = []
    #test if it counts it the correct number of times
    for _ in range(num_chromosomes):
        gen_0.append(get_chromosome(vectors[0]))
    return gen_0

def get_vectors_from_chromosome(chromosome, vectors):
    #gets only the features that have been selected to be fed into ML model from the chromosome of zeros and ones
    #if gene in the chromosome is a one, that corresponding feature should be fed into the ML model
    
    new_vectors = []
    for vector in vectors: 
        new_vector = []
        for feature_index in range(len(vector)):

            if chromosome[feature_index] == 1:
                new_vector.append(vector[feature_index])
        new_vectors.append(new_vector)
    return new_vectors

def get_accuracy(new_vectors, y_train, vectors_test, y_true):
    #gets the accuracy score of the given feature data being fed to a ML model
    
    log_model = LogisticRegression(solver="saga")
    log_model.fit(new_vectors, y_train)
    #TODO: Make vectors test just 5 features
    y_preds = log_model.predict(vectors_test)
    accuracy = metrics.accuracy_score(y_true, y_preds)
    return accuracy

def fitness(vectors_train, chromosome, y_train, vectors_test, y_true):
    #gets the fitness of the given chromosome
    return get_accuracy(get_vectors_from_chromosome(chromosome, vectors_train), y_train, get_vectors_from_chromosome(chromosome, vectors_test), y_true)

def split_data(vectors):
    #splits data so that half can be used to train the ML model and half can be used to get the accuracy score for getting the fitness of the chromosome
    vectors_train, vectors_test = np.array_split(vectors, 2)
    return [vectors_train, vectors_test]

def select_pair(population, fitness_func, vectors_train, y_vals, vectors_test, y_true):
    """
    selects the pair of solutions that will be the parents
    the better their fitness score, the more likely they will be selected as parents
    k = 2 means that we draw twice from our population because we need two parents
    """
    return random.choices(population, weights=[fitness_func(vectors_train, chromosome, y_vals, vectors_test, y_true) for chromosome in population], k=2)

def single_point_crossover(parent1, parent2):
    #randomly select a point at which to crossover
    #gets the children of the parent chromosomes
    #does single point crossover so the children have "DNA" from both parents
    crossover_point = random.randint(0, len(parent1)-1)
    child1 = parent1[:crossover_point]+parent2[crossover_point: ]
    child2 = parent2[:crossover_point] + parent1[crossover_point: ]
    return child1, child2

def mutation(chromosome, mut_probability, num = 1):
    for _ in range(num): 
        #gets a random gene from the chromosome to mutate
        index = random.randint(0, len(chromosome)-1)
        #accounts for the 1/4 chance of mutation
        if (random.random()< mut_probability):
            #switches the one to a zero and the zero to a one
            # if origninal value was zero, then 0-1 = -1 and abs(-1) is 1
            #if original value was one, then 1-0 = 0 and abs(0) is 0
           chromosome[index] = abs(chromosome[index]-1)
    return chromosome
            
def run_genetic_algorithm(vectors, y_vals, max_iterations=25, mutation_probability=0.25, fitness_function = fitness, population_size=50): 
    
    vectors_train, vectors_test = split_data(vectors)
    y_train, y_test = split_data(y_vals)
    generation = get_gen_0(vectors, population_size)
    
    
    for i in range(max_iterations-1):
        #gets the chromosomes order by their fitness score, from greatest to least
        
        key = lambda chromosome: fitness_function(vectors_train, chromosome, y_train, vectors_test, y_test)
      
        
        generation = sorted(generation, key = lambda chromosome: fitness_function(vectors_train, chromosome, y_train, vectors_test, y_test), reverse = True)
        #keep the top two chromosomes in the next generation so that we don't lose the best solutions
        #print(generation)
        print("Generation Number: " + str(max_iterations))
        print("Generation: " + str(generation))
        
        new_gen = generation[:2]
        #if one of the top two chromosomes in the generation has an accuracy of 100%, then the algorithm should stop running
        for chromosome in new_gen:
            if (int(fitness_function(vectors_train, chromosome, y_train, vectors_test, y_test)) == 1):
                return chromosome
        #gets the new generation
        #only loops for half the generation because we want the generation size to be the same, and each pair of parents produce two children
        for count in range(int(len(generation)/2)-1):
            #get the parents
            parents = select_pair(generation, fitness_function, vectors_train, y_train, vectors_test, y_test)
            #get the children
            child1, child2 = single_point_crossover(parents[0], parents[1])
            #apply the mutation function to each child
            child1 = mutation(child1, mutation_probability)
            child2 = mutation(child2, mutation_probability)
            # add these children to the new generation
            new_gen.append(child1)
            new_gen.append(child2)
        #sets the generation to the new generation so that this process can be done all over again
        generation = new_gen
        
    #re-sort the generation from best to worst
    generation = sorted(generation, key = lambda chromosome: fitness_function(vectors_train, chromosome, y_train, vectors_test, y_test),reverse = True) 
    #return the features with the best fitness 
    return get_vectors_from_chromosome(generation[0], vectors_train)
    

