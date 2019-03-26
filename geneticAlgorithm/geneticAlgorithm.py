from .individuals import Individuals, np
from multiprocessing.pool import  Pool
from multiprocessing import cpu_count


class GeneticAlgorithm:
    __mutation_rate = 0.05

    __k = 0.75

    def __init__(self, size_population, interval, type_individual, fitness_function):

        self.__interval = interval
        self.__population = self._generateRandomPopulation(size_population, interval, type_individual)

        self.fitness_function = fitness_function
        self._evaluatePopulation(self.__population)

        self.best_individuals = self._getBestIndividuals(self.__population)
        self.best_individuals.wasChosen = False

    def _generateRandomPopulation(self, size_population, interval, type_individual):
        if type_individual is float and np.array(interval).ndim == 2:
            population = list()

            for i in range(size_population):
                random_sample = list()
                for limit in interval:
                    random_sample.append(np.random.uniform(min(limit), max(limit)))

                population.append(Individuals.numpy(np.array(random_sample)))
        else:
            raise NotImplementedError

        return population

    def _selectParents(self, n_parents):
        parents = list()

        # torneiro  (Mitchell 1997)
        for i in range(n_parents):
            best, worse = self._selectRandomIndividuals(n_parents)

            if np.random.uniform(0, 1) < self.__k:  # k = 0.75
                parents.append(best)
            else:
                parents.append(worse)

        return parents

    def _selectRandomIndividuals(self, number_individuals):
        individuals_list = list()
        size_population = len(self.__population)

        for i in range(number_individuals):
            individuals_list.append(self.__population[np.random.randint(0, size_population)])

        best_individuals = self._getBestIndividuals(individuals_list)

        worse_individuals = self._getWorseIndividuals(individuals_list)

        best_individuals.wasChosen = False
        worse_individuals.wasChosen = False

        return best_individuals, worse_individuals

    def _evaluatePopulation(self, population):
        max_threads = int(cpu_count() -1 )       
        size = len(population)      
        
        # for i in range(1,max_threads+1):
        #     pool = Pool(max_threads)
        #     chromosome_list = [individuals.chromosome for individuals in population[ int((i -1)*size/3) :int(i*size/3) ]]
        #     scores = pool.map(self.fitness_function,chromosome_list)

           

        #     for score , individual  in zip( scores ,[individuals for individuals in population[ int((i -1)*size/3) :int(i*size/3) ]]):
        #         individual.score = score

        #     pool.close()
        #     pool.join()

        for individual in self.__population:
            individual.score = self.fitness_function(individual.chromosome)



    def _getBestIndividuals(self, population):
        best_individuals = population[0]
        best_score = float("inf")

        for individuals in population:

            if not individuals.wasChosen and individuals.score < best_score:
                best_individuals = individuals
                best_score = individuals.score

        best_individuals.wasChosen = True

        return best_individuals

    def _getWorseIndividuals(self, population):
        worse_indindividuals = population[0]
        worse_score = - float("inf")

        for individuals in population:
            if not individuals.wasChosen and individuals.score > worse_score:
                worse_indindividuals = individuals
                worse_score = individuals.score

        worse_indindividuals.wasChosen = True

        return worse_indindividuals

    def oneStep(self):
        parents = self._selectParents(int(len(self.__population) * 2 / 3))

        childrens = self._generateChildrens(parents)

        self._evaluatePopulation(childrens)

        best_children = self._getBestIndividuals(childrens)

        best_children.wasChosen = False

        self._replace(childrens)

        return best_children

    def execute(self):

        best_children = self.oneStep()

        if best_children.score < self.best_individuals.score:
            self.best_individuals = best_children

        return self.best_individuals

    def print(self):

        for individual in self.__population:
            print("score: {} , chormosome: {}".format(individual.score,individual.chromosome))


    def populationToCSV(self,fileName):

        save_list = np.array([ individual.chormosome  for individual in self.__population  ] )

        np.savetxt(fileName,save_list,delimiter=",")

            


    def _generateChildrens(self, parents):
        childrens = list()

        i = 0
        while i < len(parents) - 1:
            child_1, child_2 = self._Crossover(parents[i], parents[i + 1])
            childrens.extend([self._Mutate(child_1), self._Mutate(child_2)])
            i += 2

        return childrens

    def _replace(self, childrens):

        self.__population.extend(childrens)

        for i in range(len(childrens)):
            worse = self._getWorseIndividuals(self.__population)
            self.__population.remove(worse)

    def _Mutate(self, children):

        if np.random.uniform(0, 1) < self.__mutation_rate:
            self._mutating(children)

        return children

    def _Crossover(self, father, mother):
        increment = int(father.chromosome.size / 5)
        if increment == 0:
            increment = int(father.chromosome.size / 2)

        end_point = increment
        child_1_chromosome = np.zeros(shape=father.chromosome.size, dtype=father.chromosome.dtype)

        child_2_chromosome = np.zeros(shape=father.chromosome.size, dtype=father.chromosome.dtype)

        switch = True

        for i in range(father.chromosome.size):
            if switch:
                child_1_chromosome[i] = father.chromosome[i]
                child_2_chromosome[i] = mother.chromosome[i]
            else:
                child_1_chromosome[i] = mother.chromosome[i]
                child_2_chromosome[i] = father.chromosome[i]

            if i + 1 == end_point:
                end_point += increment
                switch = not switch

        return Individuals.numpy(child_1_chromosome), Individuals.numpy(child_2_chromosome)

    def _mutating(self, child):
        best, worse = self._selectRandomIndividuals(len(self.__population))

        for i in range(int(child.chromosome.size / 5 + 1)):
            self._mutateChromosome(child.chromosome)

    def _sumOrSubtract(self):
        if np.random.uniform(0, 1) < 0.5:
            return -1
        else:
            return 1

    def _mutateChromosome(self, chromosome):

        index = np.random.randint(0, chromosome.size)

        operation = self._sumOrSubtract()

        min_value =  np.min(self.__interval[index])
        max_value =  np.max(self.__interval[index]) 

        new_value = np.random.uniform(min_value,max_value) * operation + chromosome[index]

        while new_value > max(self.__interval[index]) or new_value < min(self.__interval[index]):
            operation = self._sumOrSubtract()
            new_value = np.random.uniform(min_value,max_value)
            print("new value {} , interval {}".format(new_value , self.__interval[index] ))

        chromosome[index] = new_value
