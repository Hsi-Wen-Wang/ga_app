from enum import Enum
from abc import ABC, abstractclassmethod
import random as rd
import csv
import copy as cp

class MachineProblem:
    def __init__(self, machine_list, order_list):
        self.machine_list = machine_list
        self.order_list = order_list

    def compute_chromosome(self):
        print('hi')

class BaseSet:
    def __init__(self, typ='None', ID=-1):
        self.typ = typ
        self.ID = ID

class Machine:
    def __init__(self):
        self.machine_name = 'None'
        self.processing = []
        
class Process:
    def __init__(self):
        self.typ = 'Error'
        self.ID = '-1'
        self.time = -1
        self.cost = -1

class Job(BaseSet):
    def __init__(self):
        super().__init__(typ='None', ID=-1)
        self.order = -1
        self.machine = -1


def machine_set(machine_list, machine_filename):
    with open (machine_filename, newline='') as csvfile:
        machine_data = list(csv.reader(csvfile))
        for i in range(2,len(machine_data)):
            machine_set = Machine()
            machine_set.machine_name = machine_data[i][0]
            for j in range(len(machine_data[0])):
                process_do = Process()              
                if machine_data[i][j] == '1' and machine_data[i][j+1] == '1' and machine_data[1][j] == ('A'):
                    process_do.typ = 'both'                    
                    process_do.ID = machine_data[0][j]
                    process_do.time = machine_data[i][j+2]
                    machine_set.processing.append(process_do)             
                elif machine_data[i][j] == '1' and machine_data[i][j+1] != '1' and machine_data[1][j] == ('A'):
                    process_do.typ = 'A'                    
                    process_do.ID = machine_data[0][j]
                    process_do.time = machine_data[i][j+2]
                    machine_set.processing.append(process_do)
                elif machine_data[i][j] == '1' and machine_data[i][j+1] != '1' and machine_data[i][j-1] != '1' and machine_data[1][j] == ('B'):
                    process_do.typ = 'B'                    
                    process_do.ID = machine_data[0][j]
                    process_do.time = machine_data[i][j+1]
                    machine_set.processing.append(process_do)
            machine_list.append(machine_set)

    return machine_list



if __name__ == '__main__':
    machine_list = []
    machine_filename = 'machine_setting.csv'
    machine_list = machine_set(machine_list, machine_filename)
    # print(len(machine_list))
    # for mc in machine_list:
    #     for i in range (len(mc.processing)):
    #         print(f"Machine name:{mc.machine_name}, Machine type:{mc.processing[i].typ}, Machine ID:{mc.processing[i].ID}, , Machine time:{mc.processing[i].time}")
    # job_list = []
    # job_filename = 'data.csv'
    # job_list = job_chromosome_set(job_list, job_filename, machine_list)
    # print(machine_list[18].processing[0].typ)
    print(len(machine_list))
    