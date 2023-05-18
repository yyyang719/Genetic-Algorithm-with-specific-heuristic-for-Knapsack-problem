# Genetic-Algorithms-for-Knapsack-problem
THE CREATIVE PART FOR THIS GENETIC ALGORITHM:\
Instead of using the rule that when total weight is greater than max_weight we set the total value (fitness score) to zero, this algorithm implements a specific heuristic: the fitness score = (the total value)/(the total weight/the max_weight_250), which makes sure that the more the over weight, the more the punish for its value.\
This fitness function will degrade its value as the individual becomes heavier. In this way, individuals that are slightly overweight have a chance to be picked up. After all, they might be a single mutation away from being a perfect solution. This fitness function will achieve that the individual with small overweight has small punish for its value, so that it helps the algorithm more because it also accounts for mutations and crossovers.

THE PROBLEM TO BE SOLVED:\
  There are 12 boxes:\
  Weights for each box: 20, 30, 60, 90, 50, 70, 30, 30, 70, 20, 20, 60.\
  Value for each box: 6, 5, 8, 7, 6, 9, 4, 5, 4, 9, 2, 1.\
  The goal is to fill the backpack to make it as valuable as possible without exceeding the maximum weight (250).

HOW TO DEFINE THE PROBLEM AS A GENETIC ALGORITHM:\
  genotype: one bit for each box from #1 to #12.

  chromosom: a list with length 12, each item for each box.

  phenotype: each item (box) in the chromosom is either 1 or 0. 0 means that no
             such box in the bag, 1 means that box is in the bag.

  population size: randomly initialize unique individual, 50, 100, and 200 had been tried.

  fitness function: 
  1. when the sum weight of boxes does not greater than max weight, the fitness score is the total value.
  2. when the sum is greater than max weight, the fitness score is: (the total value)/(the total weight/the max_weight_250).
  
  selection: culling 50% first, and then weighted random choice. 
            the individual with high fitness socre has larger probability to be chosen.

  genetic operators: one-point crossover and single-point mutation.

  genetic metrics: fitness graph.

  number of generations: 50, 100, 200 had been tried.
  
  PERFORMANCE:\
  ![Performance](https://github.com/yyyang719/Genetic-Algorithms-for-Knapsack-problem/assets/125943763/94deb059-46d7-42a4-a48c-069018a08b1d)
