import numpy as np
import matplotlib.pyplot as plt

from Knapsack import Knapsack

def main():
    # setting parameters
    weight_for_box = [20, 30, 60, 90, 50, 70, 30, 30, 70, 20, 20, 60] # weight for each item
    max_weight = 250 # max weight allowed
    value_for_box = [6, 5, 8, 7, 6, 9, 4, 5, 4, 9, 2, 1] # value for each item. we are trying to maximize total value
    alpha = 1.0

    geno_size = 12 # 12 box
    population_size = 200 # population size
    max_generations = 100

    # construct object
    knapsack = Knapsack(geno_size, population_size, weight_for_box, value_for_box, max_weight, alpha)

    # initialize population
    knapsack.initialize_population()

    # solve problem through GA
    best_results, best_of_all = knapsack.genetic_algorithm(max_generations) #run genetic algorithm

    # get best result
    best_score = best_results[-1][0]
    best_individual = best_results[-1][2]
    print("The solution is: {}".format(best_individual))
    print("The total weight is: {}".format(np.sum(np.array(best_individual) * np.array(weight_for_box))))
    print("The score is: {}".format(best_score))

    # plot
    best_scores = [x[0] for x in best_results]
    avg_scores = [x[1] for x in best_results]
    # fig, ax = plt.subplots()
    plt.plot(best_scores, label="Best Results Per Generation")
    plt.plot(avg_scores, label="Avg Results Per Generation")
    plt.legend()
    plt.title("Genetic Algorithm for Knapsack")
    # plt.show()
    plt.savefig("Performance.jpeg")


if __name__ == "__main__":
    main()
