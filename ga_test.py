# from enum import Enum
# from abc import ABC, abstractclassmethod
import random as rd
import csv
import copy
import time
import threading
from ga_class import Chromosome, Gene,Population,Machine,Process,Order
from collections import Counter

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
        # self.gene = Gene()
        # self.chromosome = Chromosome()
        # self.population = Population()
        self.gene = []
        self.chromosome_op = []
        self.chromosome_mc = []
        self.chromosome_decode = []
        self.population_op = []
        self.population_mc = []
#------------------------------------------
        self.pop_size = pop_size
        self.crossover_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.iteration = iteration
#------------------------------------------
        self.fitTime = []
        self.fitCost = []

#------------------------------------------
        self.count = []

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

    def fitTimeCal(self, chromosome_op, chromosome_mc):
        k = 0
        # run = 0
        for i in range(len(self.machine_list)):
            self.machine_list[i].waitQueue.clear()
            self.machine_list[i].time = 0

        for i in range(len(chromosome_mc)):
            tmp = chromosome_mc[i]
            self.machine_list[tmp].waitQueue.append(i)

        for i in range(len(self.order)):
            self.order[i].current = 1
            self.order[i].time = 0

        chromosome_decode = []
        chromosome_decode = self.decoder(chromosome_op,chromosome_decode)

        timer = 0
        
        for timer in range(300):
            run = 0
            for i in range(len(self.machine_list)):
                if self.machine_list[i].time == 0 and len(self.machine_list[i].waitQueue) != 0:
                    if int(chromosome_decode[self.machine_list[i].waitQueue[0]][1]) == self.order[int(chromosome_decode[self.machine_list[i].waitQueue[0]][0])-1].current:
                        if self.order[int(chromosome_decode[self.machine_list[i].waitQueue[0]][0])-1].time == 0:
                            for j in range(len(self.machine_list[i].processing)):
                                if self.machine_list[i].processing[j].ID == self.idDict[chromosome_decode[self.machine_list[i].waitQueue[0]]] and self.machine_list[i].processing[j].typ == self.typeDict[chromosome_decode[self.machine_list[i].waitQueue[0]]]:
                                    self.machine_list[i].time = int(self.machine_list[i].processing[j].workTime)
                                    self.order[int(chromosome_decode[self.machine_list[i].waitQueue[0]][0])-1].time = int(self.machine_list[i].processing[j].workTime) + 1
                                    break
                                elif self.machine_list[i].processing[j].ID == self.idDict[chromosome_decode[self.machine_list[i].waitQueue[0]]] and self.machine_list[i].processing[j].typ == 'both':
                                    self.machine_list[i].time = int(self.machine_list[i].processing[j].workTime)
                                    self.order[int(chromosome_decode[self.machine_list[i].waitQueue[0]][0])-1].time = int(self.machine_list[i].processing[j].workTime) + 1
                                    break
                                
                            self.order[int(chromosome_decode[self.machine_list[i].waitQueue[0]][0])-1].current += 1
                            self.machine_list[i].waitQueue.pop(0)
                
                if self.machine_list[i].time != 0:
                    run += 1
                if len(self.machine_list[i].waitQueue) != 0:
                    run += 1
            
                # if self.machine_list[i].time == 0 and len(self.machine_list[i].waitQueue) == 0 and i == len(self.machine_list)-1:
                #     timer = False

        #     k+=1
            if run == 0:
                print('run = 0')
                break
        
        return timer - 1

    def fitMcCostCal(self, chromosome_mc):
        total_cost = 0
        for i in range(len(chromosome_mc)):
            total_cost += int(self.machine_list[chromosome_mc[i]].cost)

        return total_cost


    def fitnessCal(self):
        pass

    def evaluate(self):
        pass

    def select(self):
        pass
    
    def crossover(self):
        firstpoint = rd.randint(0,len(self.population_mc[0]))
        secondpoint = rd.randint(0,len(self.population_mc[0]))
        for i in range(self.pop_size):
            

    def mutate(self):
        pass

    def report(self):
        pass


if __name__ == '__main__':
    a = GeneticAl()
    a.machineSet()
    a.jobOrderSet()
    # print(a.numDict['10'])
    a.initialize()
    # a.decoder()
    print(a.population_op[0])
    print(a.population_mc[0])
    start = time.time()
    fit = a.fitTimeCal(a.population_op[0],a.population_mc[0])
    
    fitcost = a.fitMcCostCal(a.population_mc[0])
    end = time.time()
    print(end - start)
    print(fit)
    print(fitcost)
    # print(a.chromosome_mc)

    # print(type(a.chromosome.gene[0].order_seq))
    # print(len(a.chromosome))
    
    # print(a.population[0][0][0])
    # print(a.population[0])
    # print(a.population[1][0][0])
    # print(a.population.chromosome[0].gene[0].order)
    # for i in range(len(a.population.chromosome[0].gene)):
    #     print(a.job_type_list[i])
    #     print(f"order:{a.population.chromosome[0].gene[i].order},order_seq:{a.population.chromosome[0].gene[i].order_seq}")


    
    