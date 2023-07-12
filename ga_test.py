# from enum import Enum
# from abc import ABC, abstractclassmethod
import random as rd
import csv
import copy
import time
import threading
from ga_class import Machine,Process,Order
import numpy as np
from enum import Enum

def normalize(min, max, ori_list):
    nor_list = []
    for x in ori_list:
        nor_list.append((x-min)/(max-min))
    return nor_list

# def roulette(select_list):
#     sum_val = sum(select_list)
#     random_val = rd.random()
#     probability = 0
#     for i in range(len(select_list)):
#         probability += select_list[i] / sum_val
#         if probability >= random_val:
#             return i
#         else:
#             continue

class FitnessMode(Enum):
    TimeFitness = 1
    CostFitness = 2
    DateFitness = 3
    TimeNCostFitness = 4
    AllFitness = 5


class Problem:
    def __init__(self):
        self.machine_filename = 'test.csv'
        self.job_filename = 'data.csv'
        self.machine_list = []
        self.typeDict = {}
        self.idDict = {}
        self.numDict = {}
        # type_list : A01...
        self.job_type_list  = []
        # ID_list : 110...
        self.job_ID_list = []
        self.order_count = 0
        self.order = []

    def machineSet(self):
        with open (self.machine_filename, newline='') as csvfile:
            machine_data = list(csv.reader(csvfile))
            for i in range(2,len(machine_data)):
                machine_set = Machine()
                machine_set.machine_name = machine_data[i][0]
                machine_set.cost = machine_data[i][-1]
                for j in range(len(machine_data[0])-1):
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
    
    def jobOrderSet(self):
        with open(self.job_filename, newline='') as csvfile:
            job_data = list(csv.reader(csvfile))
            k = 0
            for i in range(len(job_data)):
                tmp_order = Order()
                for j in range(len(job_data[0])-2):
                    self.typeDict[(str(i+1)+str(j))] = job_data[i][0]
                    self.idDict[(str(i+1)+str(j))]=job_data[i][j+2]
                    self.numDict[(str(i+1)+str(j))] = k
                    tmp_order.count+=1
                    k+=1
                self.order.append(tmp_order)
        self.job_type_list  = list(self.typeDict.values())
        self.job_ID_list  = list(self.idDict.values())


    def randMachine(self, num):
        can_do_list = []
        result = 0
        for i in range(len(self.machine_list)):
            for j in range(len(self.machine_list[i].processing)):
                if self.machine_list[i].processing[j].typ == self.job_type_list[num][0] and self.machine_list[i].processing[j].ID == self.job_ID_list[num]:
                    can_do_list.append(i)
                elif self.machine_list[i].processing[j].typ == 'both' and self.machine_list[i].processing[j].ID == self.job_ID_list[num]:
                    can_do_list.append(i)
        if len(can_do_list) <= 0:
            print(f"Error,{self.job_type_list[num]}{self.job_ID_list[num]} is wrong key, no machine can do!")
            return
        result = rd.randint(0, len(can_do_list)-1)
        return can_do_list[result]


class GeneticAl(Problem):
    def __init__(self, pop_size=300, cross_rate=0.1, mutate_rate=0.1, iteration=2000):
        super().__init__()
#------------------------------------------
        self.gene = []
        self.population_op = []
        self.population_mc = []
        self.sort_population_op = []
        self.sort_population_mc = []
#------------------------------------------
        self.pop_size = pop_size
        self.crossover_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.iteration = iteration
        self.elitist_rate = 0.2
        self.elist_num = int(self.pop_size*self.elitist_rate)
#------------------------------------------
        self.fitTime = []
        self.fitCost = []
        self.nor_fitTime = []
        self.nor_fitCost = []
        self.fitness = []
#------------------------------------------
        self.record_fitness = []
        # self.record_costfit = []
        # self.record_allfit = []
#------------------------------------------
        self.best_chrom_op = []
        self.best_chrom_mc = []
        self.best_fitness = 0
#------------------------------------------

        self.count = []
        self.generate = 0
        self.fitness_mode = FitnessMode.AllFitness

    # 操作編碼，111222333444
    def opEncoder(self,chromosome_op):
        for i in range(len(self.order)):
            for _ in range(self.order[i].count):
                chromosome_op.append(i+1)
        return chromosome_op
    
    # 解碼，轉成10，11，12，20，21，22...
    def decoder(self,chromosome_op,chromosome_decode):
        for i in range(len(chromosome_op)):
            tmp = chromosome_op[i]
            chromosome_decode.append(f"{tmp}{self.order[tmp-1].num}")
            self.order[tmp-1].num += 1
        for i in range(len(self.order)):
            self.order[i].num = 0
        return chromosome_decode
    
    # 機器編碼，0，1，2，3...
    def mcEncoder(self, chromosome_mc,chromosome_decode):
        for i in range(len(chromosome_decode)):
            num = self.numDict[str(chromosome_decode[i])]
            chromosome_mc.append(self.randMachine(num))
        return chromosome_mc

    def initialize(self):
        for i in range(self.pop_size):
            chromosome_op = []
            chromosome_mc = []
            chromosome_decode = []

            chromosome_op = self.opEncoder(chromosome_op)
            rd.shuffle(chromosome_op)

            chromosome_decode = self.decoder(chromosome_op,chromosome_decode)
            chromosome_mc = self.mcEncoder(chromosome_mc, chromosome_decode)

            self.population_op.append(chromosome_op)
            self.population_mc.append(chromosome_mc)
        # print('init')

    def fitTimeCal(self, chromosome_op, chromosome_mc):
        
        run = True
        chromosome_decode = []
        chromosome_decode = self.decoder(chromosome_op,chromosome_decode)
        
        for i in range(len(self.machine_list)):
            self.machine_list[i].waitQueue.clear()
            self.machine_list[i].time = 0

        for i in range(len(chromosome_mc)):
            tmp = chromosome_mc[i]
            self.machine_list[tmp].waitQueue.append(chromosome_decode[i])

        for i in range(len(self.order)):
            self.order[i].current = 0
            self.order[i].begin.clear()
            self.order[i].end.clear()
            self.order[i].begin.append(0)
        process = 0

        while run:
            judge = 0
            for i in range(len(self.machine_list)):
                if len(self.machine_list[i].waitQueue) != 0:
                    num = int(self.machine_list[i].waitQueue[0][0]) - 1
                    if self.order[num].current == int(self.machine_list[i].waitQueue[0][1]):
                        for j in range(len(self.machine_list[i].processing)):
                            if self.machine_list[i].processing[j].typ == self.typeDict[str(self.machine_list[i].waitQueue[0])] and self.machine_list[i].processing[j].ID == self.idDict[str(self.machine_list[i].waitQueue[0])]:
                                if process == 0:
                                    self.order[num].end.append(int(self.machine_list[i].processing[j].workTime) + 1)
                                    self.machine_list[i].time = self.order[num].end[-1]
                                    self.order[num].current += 1
                                else:
                                    if len(self.order[num].end) != 0:
                                        if self.order[num].end[self.order[num].current - 1] >= self.machine_list[i].time:
                                            self.order[num].begin.append(self.order[num].end[self.order[num].current - 1])
                                            self.order[num].end.append(self.order[num].begin[self.order[num].current] + int(self.machine_list[i].processing[j].workTime) + 1)
                                            self.machine_list[i].time = self.order[num].end[-1]
                                            self.order[num].current += 1
                                        else:
                                            self.order[num].begin.append(self.machine_list[i].time)
                                            self.order[num].end.append(self.order[num].begin[self.order[num].current] + int(self.machine_list[i].processing[j].workTime) + 1)
                                            self.machine_list[i].time = self.order[num].end[-1]
                                            self.order[num].current += 1
                                    else:
                                        self.order[num].begin[0] = self.machine_list[i].time
                                        self.order[num].end.append(self.order[num].begin[self.order[num].current] + int(self.machine_list[i].processing[j].workTime) + 1)
                                        self.machine_list[i].time = self.order[num].end[-1]
                                        self.order[num].current += 1
                            elif self.machine_list[i].processing[j].ID == self.idDict[str(self.machine_list[i].waitQueue[0])] and self.machine_list[i].processing[j].typ == 'both':
                                if process == 0:
                                    self.order[num].end.append(int(self.machine_list[i].processing[j].workTime) + 1)
                                    self.machine_list[i].time = self.order[num].end[-1]
                                    self.order[num].current += 1
                                else:
                                    if len(self.order[num].end) != 0:
                                        if self.order[num].end[self.order[num].current - 1] >= self.machine_list[i].time:
                                            self.order[num].begin.append(self.order[num].end[self.order[num].current - 1])
                                            self.order[num].end.append(self.order[num].begin[self.order[num].current] + int(self.machine_list[i].processing[j].workTime) + 1)
                                            self.machine_list[i].time = self.order[num].end[-1]
                                            self.order[num].current += 1
                                        else:
                                            self.order[num].begin.append(self.machine_list[i].time)
                                            self.order[num].end.append(self.order[num].begin[self.order[num].current] + int(self.machine_list[i].processing[j].workTime) + 1)
                                            self.machine_list[i].time = self.order[num].end[-1]
                                            self.order[num].current += 1
                                    else:
                                        self.order[num].begin[0] = self.machine_list[i].time
                                        self.order[num].end.append(self.order[num].begin[self.order[num].current] + int(self.machine_list[i].processing[j].workTime) + 1)
                                        self.machine_list[i].time = self.order[num].end[-1]
                                        self.order[num].current += 1
                        process+=1
                        del self.machine_list[i].waitQueue[0]
                    judge += 1

            if judge == 0:
                run = False
        fit = np.max(self.order[0].end)
        # print("123")
        for i in range(len(self.order)):
            if np.max(self.order[i].end) > fit:
                fit = np.max(self.order[i].end)

        # self.fitTime.append(fit)
        # print('fitTime')
        
        return fit


    def fitMcCostCal(self, chromosome_mc):
        total_cost = 0
        for i in range(len(chromosome_mc)):
            total_cost += int(self.machine_list[chromosome_mc[i]].cost)
        return total_cost

    def fitnessCal(self):
        base = 1
        self.fitTime.clear()
        self.fitCost.clear()
        self.fitness.clear()
        if self.fitness_mode == FitnessMode.TimeFitness:
            for i in range(len(self.population_op)):
                self.fitTime.append(self.fitTimeCal(self.population_op[i], self.population_mc[i]))
            min_time = np.min(self.fitTime)
            max_time = np.max(self.fitTime)
            self.nor_fitTime = normalize(min_time, max_time, self.fitTime)
            self.fitness = [1/(x+base) for x in self.nor_fitCost]

        if self.fitness_mode == FitnessMode.CostFitness:
            for i in range(len(self.population_mc)):
                self.fitCost.append(self.fitMcCostCal(self.population_mc[i]))
            min_cost = np.min(self.fitCost)
            max_cost = np.max(self.fitCost)
            self.nor_fitCost = normalize(min_cost, max_cost, self.fitCost)
            self.fitness = [1/(x+base) for x in self.nor_fitCost]

        if self.fitness_mode == FitnessMode.AllFitness:
            for i in range(len(self.population_op)):
                self.fitTime.append(self.fitTimeCal(self.population_op[i], self.population_mc[i]))
                self.fitCost.append(self.fitMcCostCal(self.population_mc[i]))
            min_cost = np.min(self.fitCost)
            max_cost = np.max(self.fitCost)
            min_time = np.min(self.fitTime)
            max_time = np.max(self.fitTime)
            self.nor_fitCost = normalize(min_cost, max_cost, self.fitCost)
            self.nor_fitTime = normalize(min_time, max_time, self.fitTime)
            for i in range(len(self.nor_fitTime)):
                self.fitness.append(1/(self.nor_fitCost[i]+base)+1/(self.nor_fitTime[i]+base))

    def evaluate(self):
        pass

    def keepbest(self,max_index):
        if self.generate == 0:
            self.best_chrom_op = self.population_op[max_index]
            self.best_chrom_mc = self.population_mc[max_index]
            self.best_fitness =self.fitness[max_index]

        else:
            if self.best_fitness < self.fitness[max_index]:
                self.best_chrom_op = self.population_op[max_index]
                self.best_chrom_mc = self.population_mc[max_index]
                self.best_fitness = (self.fitness[max_index])

        if self.fitness_mode == FitnessMode.TimeFitness:
            self.record_fitness.append(self.fitTime[max_index])
        elif self.fitness_mode == FitnessMode.CostFitness:
            self.record_fitness.append(self.fitCost[max_index])
        elif self.fitness_mode == FitnessMode.AllFitness:
            self.record_fitness.append(self.fitness[max_index])

    def select(self):
        index_sort = np.argsort(self.fitness)
        newpopulation_op = []
        newpopulation_mc = []
        index_sort = index_sort[::-1]
        for i in range(self.elist_num):
            newpopulation_op.append(self.population_op[index_sort[i]])
            newpopulation_mc.append(self.population_mc[index_sort[i]])

        for _ in range(self.pop_size - self.elist_num):
            select_index = self.roulette()
            newpopulation_op.append(self.population_op[select_index])
            newpopulation_mc.append(self.population_mc[select_index])

        self.population_op = newpopulation_op
        self.population_mc = newpopulation_mc
        self.keepbest(0)

    def roulette(self):
        sum_fitness = sum(self.fitness)
        transition_probability = [fitness/sum_fitness for fitness in self.fitness]
        rand = rd.random()
        sum_prob = 0

        for i, prob in enumerate(transition_probability):
            sum_prob += prob
            if (sum_prob >= rand):
                return i

    
    def crossover(self):
        if len(self.population_mc)%2 == 0:
            cross_select_list = [x for x in range(len(self.population_mc))]
            rd.shuffle(cross_select_list)
        else:
            cross_select_list = [x for x in range(len(self.population_mc)-1)]
            rd.shuffle(cross_select_list)
        for num in range(0,len(cross_select_list),2):
            k = len(self.population_op[0])
            rate = rd.random()
            if rate < self.crossover_rate:
                cross1 = cross_select_list[num]
                cross2 = cross_select_list[num+1]
                crossover_point = rd.randint(0, k-2)
                chrom1_op_cp = copy.deepcopy(self.population_op[cross1])
                chrom1_mc_cp = copy.deepcopy(self.population_mc[cross1])
                chrom2_op_cp = copy.deepcopy(self.population_op[cross2])
                chrom2_mc_cp = copy.deepcopy(self.population_mc[cross2])

                child1_op = chrom1_op_cp[:crossover_point]
                child1_mc = chrom1_mc_cp[:crossover_point]
                child2_op = chrom2_op_cp[:crossover_point]
                child2_mc = chrom2_mc_cp[:crossover_point]

                chrom1_de =[]
                chrom1_de = self.decoder(chrom1_op_cp, chrom1_de)
                
                chrom2_de = []
                chrom2_de = self.decoder(chrom2_op_cp, chrom2_de)

                chrom1_cross_de = chrom1_de[:crossover_point]
                chrom2_cross_de = chrom2_de[:crossover_point]
                chrom1_index = []
                chrom2_index = []

                for i in range(len(chrom1_cross_de)):
                    chrom1_index.append(chrom1_de.index(chrom2_cross_de[i]))
                    chrom2_index.append(chrom2_de.index(chrom1_cross_de[i]))
                chrom1_index = sorted(chrom1_index)
                chrom2_index = sorted(chrom2_index)
                for k, index in enumerate(chrom1_index):
                    chrom1_op_cp.pop(index - k)
                    chrom1_mc_cp.pop(index - k)
                for k, index in enumerate(chrom2_index):
                    chrom2_op_cp.pop(index - k)
                    chrom2_mc_cp.pop(index - k)

                child1_op += chrom2_op_cp
                child1_mc += chrom2_mc_cp
                child2_op += chrom1_op_cp
                child2_mc += chrom1_mc_cp

                self.population_op.append(child1_op)
                self.population_mc.append(child1_mc)
                self.population_op.append(child2_op)
                self.population_mc.append(child2_mc)
            else:
                continue


    def mutate(self):
        for i in range(len(self.population_op)):
            rate = rd.random()
            if rate < self.mutate_rate:
                mutate_point = rd.randint(0,len(self.population_op[0])-1)
                chrom_de = []
                chrom_de = self.decoder(self.population_op[i], chrom_de)
                # print(len(chrom_de))
                num = self.numDict[str(chrom_de[mutate_point])]
                chrom_mc_new = copy.deepcopy(self.population_mc[i])
                chrom_mc_new[mutate_point] = self.randMachine(num)
                ori_fit = self.fitTimeCal(self.population_op[i],self.population_mc[i])
                new_fit = self.fitTimeCal(self.population_op[i], chrom_mc_new)
                if new_fit > ori_fit:
                    self.population_mc[i] = chrom_mc_new
                else:
                    continue
            else:
                continue


    def iterate(self):
        while self.generate < self.iteration:
            start = time.time()
            self.fitnessCal()
            self.select()
            print(len(self.population_op))
            self.crossover()
            self.mutate()
            end = time.time()
            
            print(f"iter:{self.generate}, fitness: {self.best_fitness}time = {end-start}")
            self.generate +=1
        print("complete!")


    def report(self):
        pass
    
    def GA(self):
        self.machineSet()
        self.jobOrderSet()
        self.initialize()
        self.iterate()
        # self.report()



if __name__ == '__main__':

    ga = GeneticAl()
    ga.GA()