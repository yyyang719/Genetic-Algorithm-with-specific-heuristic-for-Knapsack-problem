#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yuanyuan
"""
# genotype: one bit for each box from #1 to #12.
# chromosom: a list with length 12, each item for each box.
# phenotype: each item (box) in the chromosom is either 1 or 0. 0 means that no
#            such box in the bag, 1 means that box is in the bag.      
# population size: randomly initialize unique individual, 50, 100, and 200 had been tried.
# fitness function: 1. when the sum weight of boxes does not greater than max weight, 
#                      the fitness score is the total value.
#                   2. when the sum is greater than max weight, the fitness score is: 
#                      (the total value)/(the total weight/the max_weight_250).
#                      the more the over weight, the more the punish for its value.
# selection: culling 50% first, and then weighted random choice. 
#            the individual with high fitness socre has larger probability to be chosen.
# genetic operators: one-point crossover and single-point mutation.
# genetic metrics: fitness graph.
# number of generations: 50, 100, 200 had been tried.

import numpy as np
import matplotlib.pyplot as plt


class Knapsack:


    def __init__(self, geno_size, population_size, weights, values, max_weight):

        self.geno_size = geno_size # the size of genes (boxes) in the chromosom.
        self.population_size = population_size # population size for each generation.
        self.population = [] # population for each generation
        self.weights = weights # a list of weight for each box.
        self.values = values # a list of value for each box.
        self.max_weight = max_weight # maximum weight for the knapsack.


    def initialize_population(self):

        # randomly generate individuals
        population_hash = []
        while len(self.population) < self.population_size:
            individual = list(np.random.randint(2, size=self.geno_size))
            individual_hash = hash(tuple(individual))

            # make sure each individual is unique
            if individual_hash not in population_hash:
                self.population.append(individual)
                population_hash.append(individual_hash)


    def fitness_score(self, individual):

        # for weight <= max_weight, directly use value
        # for weight > max_weight, new_score = (the total value)/(the total weight/the max_weight_250)
        # so that we punish overweight selections while still keep its possiblities to be chosen
        # measure fitness for individual. individual is an arrary with geno_size 12.
        cur_weight = np.sum(np.array(individual) * self.weights)
        cur_score = np.sum(np.array(individual) * self.values)
        if cur_weight > self.max_weight:
            cur_score = int(cur_score / (cur_weight / self.max_weight))

        return cur_score


    def culling(self, population):

        # culling population by 50% for our knapsack problem.
        score_individual = [] #score_individual is a list of tuple.
        for i in population:
            # each tuple is with (fitness score, individual)
            score_individual.append((self.fitness_score(i), i))

        # sort list with the fitness score in each tuple, from low fitness score to high fitness score.
        score_individual.sort(key = lambda x : x[0])

        # keep the half (50%) with higher fitness scores.
        score_individual = score_individual[int(self.population_size/2):]

        # extract fitness score from the tuple, and then normalize the fitness score to weights.
        weights = np.array([x[0] for x in score_individual])
        weights = np.cumsum(weights / sum(weights)) # cumulative sum for each weight in the list.
        # a list of tuples, each tuple is (cumulative normalized weight, individual).
        weight_individual = [(x,y[1]) for x,y in zip(weights, score_individual)]

        return weight_individual 


    def weighted_random_choices(self, weight_individual):

        # randomly choose individual according to its weight.
        weights = np.array([x[0] for x in weight_individual])

        # choose the first item in weights that greater than random number.
        # for example, if our cumulative weights is [0.1, 0.3, 0.5, 0.6, 1] for 5 individuals,
        # if the random number is 0.075, which is smaller than the first weight,
        # then we will choose the first individual.
        # if the random number is 0.55, which is samller than 0.6 and greater than 0.5,
        # then we will choose the individual corresponding to 0.6.
        idx1 = np.where((np.random.random()<weights)==True)[0][0]
        parent1 = weight_individual[idx1][1]

        idx2 = np.where((np.random.random()<weights)==True)[0][0]
        # if we need to keep two parents being different.
        # while idx2 == idx1:
        #     idx2 = np.where((np.random.random()<weights)==True)[0][0]
        parent2 = weight_individual[idx2][1]

        return parent1, parent2


    def crossover(self, parent1, parent2):

        # randomly choose a crossover line.
        crossline = np.random.randint(1, self.geno_size) 

        child1 = parent1[:crossline] + parent2[crossline:]
        child2 = parent2[:crossline] + parent1[crossline:]

        return child1, child2


    def mutation(self, child):

        # randomly pick one bit to mutate (flip)
        idx = np.random.permutation(self.geno_size)[0]
        #idx = np.random.randint(0, self.geno_size)
        child[idx] = int(not child[idx]) # 1 to 0 or 0 to 1 

        return child


    def genetic_algorithm(self, max_generations):

        best_results = [] # best result of each generation, tuple (best_score, avg_score, best_individual)
        best_of_all = (0,[]) # track the best result until current generation, track through all the generations

        for current_generation in range(max_generations):
            if (current_generation+1) % 10 == 0:
                print("Generation {:03d}, best score so far {:03d}, avg score so far {:03.2f}...".format(current_generation+1, best_results[-1][0], best_results[-1][1]))

            next_generation = []
            
            for i in range (int(self.population_size/2)):

                # weighted randomly choose two parents from the culled population
                parent1, parent2 = self.weighted_random_choices(self.culling(self.population))

                # cross over to reproduce children
                child1, child2 = self.crossover(parent1, parent2)

                # one-single mutation
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)

                next_generation.append(child1)
                next_generation.append(child2)
                
            self.population = next_generation
            
            # calculate fitness score for each individual in the current population
            fitness_scores = [self.fitness_score(np.array(x)) for x in self.population]
            # find best individual within current generation
            best_score = max(fitness_scores)
            # calculate average score for the whole current population
            avg_score = sum(fitness_scores) / len(fitness_scores)
            best_individual = self.population[fitness_scores.index(best_score)]
            best_results.append((best_score, avg_score, best_individual))

            # update best of all
            if best_score > best_of_all[0]:
                best_of_all = (best_score, best_individual)

            # replace current worst with global best, keep the best individual to the last generation
            worst_idx = fitness_scores.index(min(fitness_scores))
            self.population[worst_idx] = best_of_all[1]

        return best_results, best_of_all
