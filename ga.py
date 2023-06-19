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

class Machine(BaseSet):
    def __init__(self):
        super().__init__(typ='None', ID=-1)
        self.machine_name = 'None'
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
                if machine_data[i][j] == '1' and machine_data[i][j+1] == '1' and machine_data[1][j] == ('A'):
                    machine_set.typ = 'both'                    
                    machine_set.ID = machine_data[0][j]
                    machine_set.time = machine_data[i][j+2]
                    machine_list.append(machine_set)                
                elif machine_data[i][j] == '1' and machine_data[i][j+1] != '1' and machine_data[1][j] == ('A'):
                    machine_set.typ = 'A'                    
                    machine_set.ID = machine_data[0][j]
                    machine_set.time = machine_data[i][j+2]
                    machine_list.append(machine_set)
                elif machine_data[i][j] == '1' and machine_data[i][j+1] != '1' and machine_data[i][j-1] != '1' and machine_data[1][j] == ('B'):
                    machine_set.typ = 'B'                    
                    machine_set.ID = machine_data[0][j]
                    machine_set.time = machine_data[i][j+1]
                    machine_list.append(machine_set)
    return machine_list

def job_set(job_list, jobfilename):
    with open(jobfilename, newline='') as csvfile:
        job_data = list(csv.reader(csvfile))
        for i in range(len(job_data)):
            for j in range(2, len(job_data[0])):
                job_set = Job()
                job_set.typ = job_data[i][0]
                job_set.order = job_data[i][1]
                job_set.ID = job_data[i][j]
                job_set.machine = randMachine(1,2)
                job_list.append(job_set)
    return job_list
def randMachine(a, b):
    return a+b

if __name__ == '__main__':
    machine_list = []
    machine_filename = 'machine_setting.csv'
    machine_list = machine_set(machine_list, machine_filename)
    
    job_list = []
    job_filename = 'data.csv'
    job_list = job_set(job_list, job_filename)
    print(len(job_list))
    print(job_list[6].order)