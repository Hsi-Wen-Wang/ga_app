from enum import Enum
from abc import ABC, abstractclassmethod
import random as rd
import csv
import copy
import time
from queue import Queue

class MachineProblem:
    def __init__(self, machine_filename, job_filename):
        self.machine_filename = machine_filename
        self.job_filename = job_filename
        self.machine_list = []
        self.job_list = Chromosome()

    def machineSet(self):
        with open (self.machine_filename, newline='') as csvfile:
            machine_data = list(csv.reader(csvfile))
            for i in range(2,len(machine_data)):
                machine_set = Machine()
                machine_set.machine_name = machine_data[i][0]
                for j in range(len(machine_data[0])):
                    process_do = Process()              
                    if machine_data[i][j] == '1' and machine_data[i][j+1] == '1' and machine_data[1][j] == ('A'):
                        process_do.typ = 'both'                    
                        process_do.ID = machine_data[0][j]
                        process_do.workTime = machine_data[i][j+2]
                        machine_set.processing.append(process_do)             
                    elif machine_data[i][j] == '1' and machine_data[i][j+1] != '1' and machine_data[1][j] == ('A'):
                        process_do.typ = 'A'                    
                        process_do.ID = machine_data[0][j]
                        process_do.workTime = machine_data[i][j+2]
                        machine_set.processing.append(process_do)
                    elif machine_data[i][j] == '1' and machine_data[i][j+1] != '1' and machine_data[i][j-1] != '1' and machine_data[1][j] == ('B'):
                        process_do.typ = 'B'                    
                        process_do.ID = machine_data[0][j]
                        process_do.workTime = machine_data[i][j+1]
                        machine_set.processing.append(process_do)
                self.machine_list.append(machine_set)
    
    def job_chromosome_set(self, order=[]):
        k = 0
        with open(self.job_filename, newline='') as csvfile:
            job_data = list(csv.reader(csvfile))
            for i in range(len(job_data)):
                for j in range(2, len(job_data[0])):
                    job_set = Gene()
                    job_set.typ = job_data[i][0]
                    job_set.order = i
                    job_set.ID = job_data[i][j]
                    job_set.process_machine = randMachine(self.machine_list, job_set)
                    job_set.current = k
                    self.job_list.gene.append(job_set)
                    k += 1
                tmp = Order()
                order.append(tmp)
                k = 0
        return self.job_list
        
class Gene:
    def __init__(self):
        self.ID = -1
        self.order = -1
        self.typ = 'None'
        self.process_machine = 'None'
        self.current = -1
        self.process = 0

class Chromosome:
    def __init__(self):
        self.gene = []
        self.fitness = 0
        self.rfitness = 0
        self.cfitness = 0
    def calculate_fitness(self):
        self.fitness = 2
class Population:
    def __init__(self):
        self.chromosome = []
        self.best_result = 1000
class Machine:
    def __init__(self):
        self.machine_name = 'None'
        self.time = 0
        self.waitQueue = []
        self.processing = []
        
class Process:
    def __init__(self):
        self.typ = 'Error'
        self.ID = '-1'
        self.workTime = -1
        self.cost = -1

class Order:
    def __init__(self):
        self.current = 0
        self.time = 0
        
        

def randMachine(machine_list=[], gene=Gene()):
    can_do_list = []
    result = 0
    for i in range(len(machine_list)):
        for j in range(len(machine_list[i].processing)):
            if machine_list[i].processing[j].typ == gene.typ and machine_list[i].processing[j].ID ==gene.ID:
                can_do_list.append(i)
            elif machine_list[i].processing[j].typ == 'both' and machine_list[i].processing[j].ID == gene.ID:
                can_do_list.append(i)
    if len(can_do_list) <= 0:
        print(f"Error,{gene.typ}{gene.ID} is wrong key, no machine can do!")
        return
    result = rd.randint(0, len(can_do_list)-1)
    return can_do_list[result]



class GeneticAl(MachineProblem):
    def __init__(self, population_size=300, crossover_rate=0.8, mutation_rate=0.1, iteration=20, machine_filename= 'machine_setting.csv', job_filename= 'data.csv'):
        super().__init__(machine_filename, job_filename)
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.iteration = iteration
        self.chromosome = Chromosome()
        self.population = Population()
        self.order = []
        self.timeRecord = [0 for x in range(self.population_size)]
        self.PmutationM = 1
        self.Pmutation = 1
        self.Pcrossover = 1
        self.generation = 0
        self.util = []
        self.result_filename = 'result.csv'
        self.avgFitnessQueue = []
        self.fitnessQueue = []

    def initialize(self):
        self.generation = 0
        self.population.chromosome.append(self.chromosome)
        
        for _ in range(self.population_size):
            copy_chromosome = copy.deepcopy(self.chromosome)
            rd.shuffle(copy_chromosome.gene)
            self.population.chromosome.append(copy_chromosome)

    def fitnessTimeCal(self):
        # start = time.time()
        tmp = -1
        run = 0

        for i in range(len(self.machine_list)):
            self.machine_list[i].waitQueue.clear()
            self.machine_list[i].time = 0

        for i in range(len(self.order)):
            self.order[i].current = 0
            self.order[i].time = 0

        for i in range(len(self.chromosome.gene)):
            tmp = self.chromosome.gene[i].process_machine
            self.machine_list[tmp].waitQueue.append(i)
        k = 1
        for timer in range(300):
            
            run = 0
            # for i in range(len(self.order)):
            #     if self.order[i].time != 0:
            #         self.order[i].time -= 1
            
            for i in range(len(self.machine_list)):
                if self.machine_list[i].time != 0:
                    self.machine_list[i].time -= 1
                if self.machine_list[i].time == 0 and len(self.machine_list[i].waitQueue) != 0:
                    if self.chromosome.gene[self.machine_list[i].waitQueue[0]].current == self.order[self.chromosome.gene[self.machine_list[i].waitQueue[0]].order].current:
                        if self.order[self.chromosome.gene[self.machine_list[i].waitQueue[0]].order].time == 0:
                            # print(f"第{k}次")
                            # k += 1
                            for j in range(len(self.machine_list[i].processing)):
                                # print(f"第{k}次")
                                # k += 1
                                # print(f'{self.machine_list[i].processing[j].typ} and {self.chromosome.gene[self.machine_list[i].waitQueue[0]].typ}')
                                if self.machine_list[i].processing[j].ID == self.chromosome.gene[self.machine_list[i].waitQueue[0]].ID and self.machine_list[i].processing[j].typ == self.chromosome.gene[self.machine_list[i].waitQueue[0]].typ:
                                    self.machine_list[i].time = int(self.machine_list[i].processing[j].workTime)
                                    self.order[self.chromosome.gene[self.machine_list[i].waitQueue[0]].order].time = int(self.machine_list[i].processing[j].workTime) + 1    
                                    break
                                    # print(f"第{k}次")
                                    # k += 1
                                elif self.machine_list[i].processing[j].ID == self.chromosome.gene[self.machine_list[i].waitQueue[0]].ID and self.machine_list[i].processing[j].typ == 'both':
                                    self.machine_list[i].time = int(self.machine_list[i].processing[j].workTime)
                                    self.order[self.chromosome.gene[self.machine_list[i].waitQueue[0]].order].time = int(self.machine_list[i].processing[j].workTime) + 1
                                    break
                                    # print(f"第{k}次")
                                    # k += 1
                            self.timeRecord[self.machine_list[i].waitQueue[0]] = timer

                            self.order[self.chromosome.gene[self.machine_list[i].waitQueue[0]].order].current += 1
                            self.machine_list[i].waitQueue.pop(0)
                
                if self.machine_list[i].time != 0:
                    run += 1
                if len(self.machine_list[i].waitQueue) != 0:
                    run += 1
            
            if run == 0:
                break
            # end = time.time()
            # print(f"fitness 執行時間:{end - start}")
        return timer - 1

        
    def evaluate(self):
        
        for popCount in range(self.population_size+1):
            # start = time.time()
            self.chromosome = self.population.chromosome[popCount]
            # end = time.time()
            # print(f"evaluate 執行時間:{end - start}")
            # start = time.time()
            self.population.chromosome[popCount].fitness = self.fitnessTimeCal()
            # end = time.time()
            # print(f"evaluate2 執行時間:{end - start}")
        
        


    def selection(self):
        # start = time.time()
        sum = 0
        p = 0
        newpopulation = Population()
        newpopulation = self.population
        for mem in range(self.population_size):
            sum += (1/self.population.chromosome[mem].fitness)

        for mem in range(self.population_size):
            self.population.chromosome[mem].rfitness = 1 / (sum * self.population.chromosome[mem].fitness)
        
        self.population.chromosome[0].cfitness = self.population.chromosome[0].rfitness

        for mem in range(1, self.population_size):
            self.population.chromosome[mem].cfitness = self.population.chromosome[mem - 1].cfitness + self.population.chromosome[mem].rfitness

        for i in range(self.population_size):
            p = rd.randrange(32768)%1000 / 1000.0
            if p < self.population.chromosome[0].cfitness:
                newpopulation.chromosome[i] = self.population.chromosome[0]

            else:
                for j in range(self.population_size):
                    if p >= self.population.chromosome[j].cfitness and self.population.chromosome[j+1].cfitness:
                        newpopulation.chromosome[i] = self.population.chromosome[j+1]

        self.population = newpopulation

        # end = time.time()
        # print(f"selection 執行時間:{end - start}")


    def elitist(self):
        # start = time.time()
        best_mem = 0
        worst_mem = 0
        best = self.population.chromosome[0].fitness
        worst = self.population.chromosome[0].fitness

        for i in range(self.population_size):
            if self.population.chromosome[i].fitness > self.population.chromosome[i+1].fitness:
                if self.population.chromosome[i].fitness >= worst:
                    worst = self.population.chromosome[i].fitness
                    worst_mem = i

                if self.population.chromosome[i+1].fitness <= best:
                    best = self.population.chromosome[i+1].fitness
                    best_mem = i + 1

            else:
                if self.population.chromosome[i].fitness <= best:
                    best = self.population.chromosome[i].fitness
                    best_mem = i
                if self.population.chromosome[i+1].fitness >= worst:
                    worst = self.population.chromosome[i+1].fitness
                    worst_mem = i + 1

        if best < self.population.chromosome[self.population_size].fitness:
            for i in range(len(self.population.chromosome[self.population_size].gene)):
                self.population.chromosome[self.population_size].gene[i] = self.population.chromosome[best_mem].gene[i]

            self.population.chromosome[self.population_size].fitness = self.population.chromosome[best_mem].fitness
        else:
            self.population.chromosome[worst_mem].gene = self.population.chromosome[self.population_size].gene

            self.population.chromosome[worst_mem].fitness = self.population.chromosome[self.population_size].fitness

        # end = time.time()
        # print(f"elitist 執行時間:{end - start}")

    
    def crossover(self):
        # start = time.time()
        for i in range(self.population_size):
            x = rd.randrange(32768)%1000/1000.0
            if x < self.Pmutation:
                firstPoint = rd.randint(0, len(self.population.chromosome[i].gene))
                secondPoint = rd.randint(0, len(self.population.chromosome[i].gene))

                if secondPoint < firstPoint:
                    temp = firstPoint
                    firstPoint = secondPoint
                    secondPoint = temp
                
                for j in range(firstPoint, secondPoint):
                    for k in range(len(self.population.chromosome[i].gene)):
                        if self.population.chromosome[i].gene[k].current == self.population.chromosome[self.population_size].gene[j].current and self.population.chromosome[i].gene[k].order == self.population.chromosome[self.population_size].gene[j].order:
                            del self.population.chromosome[i].gene[k]
                            break

                for j in range(firstPoint, secondPoint):
                    mid = len(self.population.chromosome[i].gene) / 2
                    mid = int(mid)
                    self.population.chromosome[i].gene.insert(mid, self.population.chromosome[self.population_size].gene[j])
        
        # end = time.time()
        # print(f"crossover 執行時間:{end - start}")

    def keepBest(self):
        # start = time.time()
        cur_best = 0
        self.population.chromosome[self.population_size].fitness = 9999

        for popCount in range(self.population_size):
            if self.population.chromosome[popCount].fitness < self.population.chromosome[self.population_size].fitness:
                cur_best = popCount
                self.population.chromosome[self.population_size].fitness = self.population.chromosome[popCount].fitness

        for i in range(len(self.population.chromosome[self.population_size].gene)):
            self.population.chromosome[self.population_size].gene[i] = self.population.chromosome[cur_best].gene[i]
        # end = time.time()
        # print(f"keepBest 執行時間:{end - start}")


    def mutate(self):
        # start = time.time()
        for i in range(self.population_size):
            x = rd.randrange(32768)%1000/1000.0
            if x < self.Pmutation:
                for j in range(10):
                    firstPoint = rd.randint(0, len(self.population.chromosome[i].gene) - 1)
                    secondPoint = rd.randint(0, len(self.population.chromosome[i].gene) - 1)
                    self.population.chromosome[i].gene[firstPoint], self.population.chromosome[i].gene[secondPoint] = self.population.chromosome[i].gene[secondPoint], self.population.chromosome[i].gene[firstPoint]

        for i in range(self.population_size):
            for j in range(len(self.population.chromosome[i].gene)):
                x = rd.randrange(32768)%1000/1000.0
                if x < self.PmutationM:
                    self.population.chromosome[i].gene[j].process_machine = randMachine(self.machine_list, self.population.chromosome[i].gene[j])

        # end = time.time()
        # print(f"mutate 執行時間:{end - start}")

    def report(self):
        # start = time.time()
        avg = .0

        if self.generation % 20 == 0:
            self.util.clear()
            maxProcess = 0
            self.chromosome = self.population.chromosome[self.population_size]
            self.fitnessTimeCal()

            for i in range (len(self.machine_list)):
                self.machine_list[i].waitQueue.clear()

            for i in range(len(self.population.chromosome[self.population_size].gene)):
                tmp = self.population.chromosome[self.population_size].gene[i].process_machine
                self.machine_list[tmp].waitQueue.append(i)
                if len(self.machine_list[tmp].waitQueue) > maxProcess:
                    maxProcess += 1

            with open(self.result_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                recordList = ['Machine    ']
                recordList.append('    ')
                endl = []

                for i in range(1, maxProcess):
                    recordList.append(f"Process {i},      ")
                    writer.writerow(recordList)
                writer.writerow(endl)
                for i in range(len(self.machine_list)):
                    utilTime = 0

                    machine_record = []
                    machine_record.append(self.machine_list[i].machine_name)
                    
                    for j in range(len(self.machine_list[i].waitQueue)):
                        machine_record.append(f"{self.population.chromosome[self.population_size].gene[self.machine_list[i].waitQueue[j]].typ}{self.population.chromosome[self.population_size].gene[self.machine_list[i].waitQueue[j]].order}: {self.population.chromosome[self.population_size].gene[self.machine_list[i].waitQueue[j]].current} OP{self.population.chromosome[self.population_size].gene[self.machine_list[i].waitQueue[j]].ID}  {self.timeRecord[self.machine_list[i].waitQueue[j]]}")

                        for k in range(len(self.machine_list[i].processing)):
                            if self.machine_list[i].processing[k].ID == self.population.chromosome[self.population_size].gene[self.machine_list[i].waitQueue[j]].ID:
                                utilTime += int(self.machine_list[i].processing[k].workTime)
                                break
                    writer.writerow(machine_record)

                    self.util.append(utilTime/self.population.chromosome[self.population_size].fitness)

        for i in range(self.population_size):
            avg += self.population.chromosome[i].fitness / self.population_size

        self.avgFitnessQueue.append(avg)
        self.fitnessQueue.append(self.population.chromosome[self.population_size].fitness)
        print(f"iter:{self.generation}:{self.population.chromosome[self.population_size].fitness} adverge:{avg}")
        # end = time.time()
        # print(f"report 執行時間:{end - start}")

    
    def iterate(self, show):
        start = time.time()
        self.generation += 1
        self.selection()
        end = time.time()
        print(f"selection 執行時間:{end - start}")
        start = time.time()
        self.crossover()
        end = time.time()
        print(f"crossover 執行時間:{end - start}")
        start = time.time()
        self.mutate()
        end = time.time()
        print(f"mutate 執行時間:{end - start}")

        if show == True:
            start = time.time()
            self.report()
            end = time.time()
            print(f"report 執行時間:{end - start}")
        start = time.time()
        self.evaluate()
        end = time.time()
        print(f"evalutate 執行時間:{end - start}")
        start = time.time()
        self.elitist()
        end = time.time()
        print(f"elitist 執行時間:{end - start}")
        # end = time.time()
        # print(f"iterate 執行時間:{end - start}")

    # def calculate(self, show)
    #     self.

    def ga(self):
        self.machineSet()
        self.chromosome = self.job_chromosome_set(self.order)

        self.population.chromosome.append(self.chromosome)
        self.initialize()
        

        self.evaluate()
        self.keepBest()
        # for i in range(len(self.population.chromosome)):
        #     print(self.population.chromosome[i].fitness)

        while self.generation < self.iteration:
            self.iterate(True)
        
        print("sucess")

        



if __name__ == '__main__':
    ga = GeneticAl()
    ga.ga()


    
    