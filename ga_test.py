# from enum import Enum
# from abc import ABC, abstractclassmethod
import random as rd
import csv
import copy
import time
import threading
from ga_class import Machine,Process,Order
import numpy as np

def normalize(min, max, ori_list):
    nor_list = []
    for x in ori_list:
        nor_list.append((x-min)/(max-min))
    return nor_list

def roulette(select_list):
    sum_val = sum(select_list)
    random_val = rd.random()
    probability = 0
    for i in range(len(select_list)):
        probability += select_list[i] / sum_val
        if probability >= random_val:
            return i
        else:
            continue

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
    def __init__(self, pop_size=300, cross_rate=0.8, mutate_rate=0.1, iteration=200):
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

        self.best_chrom_op = []
        self.best_chrom_mc = []
        self.best_fitness = []
#------------------------------------------
        self.count = []
        self.generate = 0

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

        self.fitTime.append(fit)
        # print('fitTime')
        
        return fit
    
    def keepbest(self,index_sort):
        max_index = index_sort[-1]
        if self.generate == 0:
            self.best_chrom_op = self.population_op[max_index]
            self.best_chrom_mc = self.population_mc[max_index]
            self.best_fitness.append(self.fitTime[max_index])
            self.best_fitness.append(self.fitCost[max_index])
            self.best_fitness.append(self.fitness[max_index])
        else:
            if self.best_fitness[2] < self.fitness[max_index]:
                self.best_chrom_op = self.population_op[max_index]
                self.best_chrom_mc = self.population_mc[max_index]
                self.best_fitness[0] = (self.fitTime[max_index])
                self.best_fitness[1] = (self.fitCost[max_index])
                self.best_fitness[2] = (self.fitness[max_index])
        # print('keepbest')


    def fitMcCostCal(self, chromosome_mc):
        total_cost = 0
        for i in range(len(chromosome_mc)):
            total_cost += int(self.machine_list[chromosome_mc[i]].cost)

        self.fitCost.append(total_cost)
        # print('fitCost')

        return total_cost


    def fitnessCal(self):
        min_cost = min(self.fitCost)
        max_cost = max(self.fitCost)
        min_time = min(self.fitTime)
        max_time = max(self.fitTime)
        self.nor_fitCost = normalize(min_cost, max_cost, self.fitCost)
        self.nor_fitTime = normalize(min_time, max_time, self.fitTime)
        base = 1
        for i in range(len(self.nor_fitTime)):
            self.fitness.append(1/(self.nor_fitCost[i]+base)+1/(self.nor_fitTime[i]+base))
        # print('fitness-------------')

    def evaluate(self):
        pass

    def select(self):
        index_sort = np.argsort(self.fitness)
        newpopulation_op = []
        newpopulation_mc = []

        k = 0
        for i in range(self.elist_num):
            newpopulation_op.append(self.population_op[index_sort[k-i]])
            newpopulation_mc.append(self.population_mc[index_sort[k-i]])

        for _ in range(self.pop_size - self.elist_num):
            select_index = roulette(self.fitness)
            newpopulation_op.append(self.population_op[select_index])
            newpopulation_mc.append(self.population_mc[select_index])
        self.keepbest(index_sort)

        self.fitCost.clear()
        self.fitness.clear()
        self.fitTime.clear()
        # print('select')
    
    def crossover(self):
        cross_select  = [x for x in range(self.pop_size)]
        rd.shuffle(cross_select)

        num = len(self.population_mc[0])
        for i in range(0,self.pop_size,2):
            rd_rate = rd.random()
            firstpoint = rd.randint(0,len(self.population_mc[0]))
            secondpoint = rd.randint(0,len(self.population_mc[0]))
            if rd_rate < self.crossover_rate:
                if firstpoint > secondpoint:
                    tmp = firstpoint
                    firstpoint = secondpoint
                    secondpoint = tmp

                parent1_op = copy.deepcopy(self.population_op[i])
                parent1_mc = copy.deepcopy(self.population_mc[i])

                parent1_de = []
                parent1_de = self.decoder(parent1_op, parent1_de)

                parent2_op = copy.deepcopy(self.population_op[i+1])
                parent2_mc = copy.deepcopy(self.population_mc[i+1])

                parent2_de = []
                parent2_de = self.decoder(parent2_op, parent2_de)

                de1_tmp = []
                de2_tmp = []
                op1_tmp = []
                op2_tmp = []
                mc1_tmp = []
                mc2_tmp = []

                cross1_index = []
                cross2_index = []

                for j in range(firstpoint, secondpoint):
                    de1_tmp.append(parent1_de[j])
                    de2_tmp.append(parent2_de[j])

                    op1_tmp.append(parent1_op[j])
                    op2_tmp.append(parent2_op[j])

                    mc1_tmp.append(parent1_mc[j])
                    mc2_tmp.append(parent2_mc[j])

                for j in range(secondpoint-firstpoint):
                    cross1_index.append(parent2_de.index(de1_tmp[j]))
                    cross2_index.append(parent1_de.index(de2_tmp[j]))

                cross1_index = sorted(cross1_index)
                cross2_index = sorted(cross2_index)
                    
                for k, index in enumerate(cross1_index):
                    index = index - k
                    parent2_op.pop(index)
                    parent2_mc.pop(index)
                for k, index in enumerate(cross2_index):
                    index = index - k
                    parent1_op.pop(index)
                    parent1_mc.pop(index)

                for j in range(len(op1_tmp)):
                    parent2_op.insert(secondpoint, op1_tmp[j])
                    parent1_op.insert(secondpoint, op2_tmp[j])
                    parent2_mc.insert(secondpoint, mc1_tmp[j])
                    parent1_mc.insert(secondpoint, mc1_tmp[j])
                
                self.population_op.append(parent2_op)
                self.population_op.append(parent1_op)

                self.population_mc.append(parent2_mc)
                self.population_mc.append(parent1_mc)
        # print('crossover')

    def mutate(self):
        for i in range(len(self.population_mc)):
            rate = rd.random()
            if rate < self.mutate_rate:
                chromosome_de = []
                chromosome_de = self.decoder(self.population_op[i], chromosome_de)
                index = rd.randint(0, len(self.population_mc[0])-1)
                num = self.numDict[str(chromosome_de[index])]
                mutate_chrome = copy.deepcopy(self.population_mc[i])
                mutate_chrome[index] = self.randMachine(num)
                ori_fit = self.fitTimeCal(self.population_op[i], self.population_mc[i])
                mutate_fit = self.fitTimeCal(self.population_op[i], mutate_chrome)
                if mutate_fit > ori_fit:
                    self.population_mc[i] = mutate_chrome
                else:
                    continue
            else:
                continue

    def iterate(self):
        while self.generate < self.iteration:
            start = time.time()
            print(len(self.population_mc))
            for i in range(len(self.population_mc)):
                self.fitTimeCal(self.population_op[i], self.population_mc[i])
                self.fitMcCostCal(self.population_mc[i])
            
            self.fitnessCal()
            
            self.select()
            # print(len(self.population_mc))
            
            # self.crossover()
            
            
            # self.mutate()
            end = time.time()
            print(f"iter:{self.generate}, time = {end-start}")
            self.generate +=1


    def report(self):
        pass
    
    def GA(self):
        
        self.machineSet()
        
        self.jobOrderSet()
        
        self.initialize()
        # start = time.time()
        # for i in range(len(self.population_mc)):
        #     self.fitTimeCal(self.population_op[i], self.population_mc[i])
        # end = time.time()
        # print(end-start)
        # print(self.fitTime)
        # print(np.min(self.fitTime))
        self.iterate()
        # print(self.best_chrom_op)
        # print(self.best_chrom_mc)
        # print(self.best_fitness)
        

if __name__ == '__main__':

    ga = GeneticAl()
    ga.GA()