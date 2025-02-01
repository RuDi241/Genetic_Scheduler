import copy
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from functools import partial
import os
import sys

#input
MACHINES = 3
SEED = 42423
SURVIVAL_COEF = 0.2
MUTATION_COEF = 0.1
CHROMOSOMES = 10
GENERATIONS = 100

TIME_WEIGHT = 2
UNFINISHED_WEIGHT = 10
FINISHED_WEIGHT = 1
rnd.seed(SEED)

def read_file(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")
    with open(filename, 'r') as file:
        first_line = file.readline().strip()
        num_machines, num_parts = map(int, first_line.split())

        parts = []
        for _ in range(num_parts):
            num_operations = int(file.readline().strip())
            operations = []
            for __ in range(num_operations):
                machine, duration = map(int, file.readline().strip().split())
                operations.append((machine, duration))
            parts.append(operations)

    return num_machines, num_parts, parts

class Task:
    def __init__(self, part, duration, previous_task, machine):
        self.part = part
        self.duration = duration
        self.previous_task = previous_task
        self.is_done = 0
        self.machine = machine
        self.started_at = -1
        self.finished_at = -1
        if duration > 0:
            is_done = 0

    def run(self, time):
        if self.started_at == -1:
            self.started_at = time
            self.finished_at = time + self.duration - 1
        if not self.duration > 0:
            raise ValueError("run() was called for a finished task.")
        self.duration -= 1
        if self.duration == 0:
            self.is_done = 1

def generate_tasks(parts):
    tasks = [[] for _ in range(MACHINES)]
    for i, part in enumerate(parts):
        previous_task = None
        for operation in part:
            machine_index = operation[0]
            new_task = Task(i, operation[1], previous_task, machine_index) 
            tasks[machine_index].append(new_task)
            previous_task = new_task

    #tasks[machine][task]
    return tasks

def generate_chromosome(tasks):
    chromosome = []
    for machine, task_list in enumerate(tasks):
        #number of tasks for a particular machine
        n_tasks = len(task_list)
        sub_chromosome = list(range(n_tasks))
        rnd.shuffle(sub_chromosome)
        chromosome.append(sub_chromosome)
    
    return chromosome

def generate_G0(tasks, N):
    chromosomes = []
    for _ in range(N):
        chromosomes.append(generate_chromosome(tasks))
    
    return chromosomes

def subchromosome_repair(subchromosome):
    #sub = np.array(subchromosome)
    sub = subchromosome
    N = len(subchromosome)
    missing = list(set(range(N)) - set(sub))
    rnd.shuffle(missing)
    counts = {}
    duplicates = []
    for idx, val in enumerate(sub):
        if val in counts:
            counts[val].append(idx)
            if len(counts[val]) == 2:
                duplicates.append(val)
        else:
            counts[val] = [idx]
    idx = 0
    for dup in duplicates:
        for i in counts[dup][1:]:
            sub[i] = missing[idx]
            idx += 1
    return sub

def subchromosome_crossover(parent1, parent2):
    crossover_point = rnd.randint(0, len(parent1) )
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    child1 = subchromosome_repair(child1)
    child2 = subchromosome_repair(child2)
    return child1, child2

def crossover(p1, p2):
    #select subchromosome
    parent1 = copy.deepcopy(p1)
    parent2 = copy.deepcopy(p2)
    sub_index= rnd.randint(0, len(parent1) - 1)
    child_sub1, child_sub2 = subchromosome_crossover(parent1[sub_index], parent2[sub_index])

    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)
    child1[sub_index] = child_sub1
    child2[sub_index] = child_sub2
    if p1 != parent1:
        print("error")
    return child1, child2

def mutation(cr):
    #select subchromosome
    chromosome = copy.deepcopy(cr)
    sub_index= rnd.randint(0, len(chromosome) - 1)
    #select indexes to swap
    index1 = rnd.randint(0, len(chromosome[sub_index]) - 1)
    index2 = rnd.randint(0, len(chromosome[sub_index]) - 1)
    temp = chromosome[sub_index][index1]
    chromosome[sub_index][index1] = chromosome[sub_index][index2]
    chromosome[sub_index][index2] = temp
    return chromosome

def generate_G1(chromosomes):
    
    survival_index = int(SURVIVAL_COEF*len(chromosomes))
    overwrite_location = survival_index + 1
    while overwrite_location + 1 < len(chromosomes)*(1 - MUTATION_COEF): 
        select1 = rnd.randint(0,survival_index)
        select2 = rnd.randint(0, survival_index)
        child1, child2 = crossover(chromosomes[select1], chromosomes[select2])
        chromosomes[overwrite_location] = child1
        chromosomes[overwrite_location + 1] = child2
        overwrite_location += 2     

    while overwrite_location < len(chromosomes):
        select = rnd.randint(0, survival_index)
        chromosomes[overwrite_location] = mutation(chromosomes[select])
        overwrite_location += 1
    return chromosomes
    
def get_max_time(tasks):
    max_time = 0
    for task_list in tasks:
        for task in task_list:
            max_time += task.duration

    return max_time

class MachineClass:
    def __init__(self, task_list, id):
        self.task_list = task_list
        self.current_task = None
        self.is_done = 0
        self.id = id 
        if len(task_list) != 0:
            self.current_task = task_list[0]
            self.current_task_id = 0

    def work(self, time, parts_status):
        if self.is_done == 1:
            return 0
        if self.current_task == None:
            return 0
        if self.current_task.previous_task != None and self.current_task.previous_task.is_done == 0:
            return 0
        if parts_status[self.current_task.part] != self.id and parts_status[self.current_task.part] != -1:
            return 0
     
        parts_status[self.current_task.part] = self.id
        self.current_task.run(time)
        if self.current_task.is_done == 1:
            parts_status[self.current_task.part] = -2
            self.current_task_id += 1
            if self.current_task_id >= len(self.task_list):
                self.is_done = 1
                self.current_task = None
                return 1
            self.current_task = self.task_list[self.current_task_id]
        return 0

def order_list(my_list, order):
    result = [my_list[i] for i in order]
    return result

def simulation(chromosome, tasks, max_time, parts):
    tasks_copy = copy.deepcopy(tasks)
    total_time = 0

    machines = np.array([MachineClass(order_list(task_list,chromosome[id]), id) for id, task_list in enumerate(tasks_copy)])
    parts_status = np.ones(len(parts))*-1
    #free = -1
    #just_freed = -2
    #busy = machine_id (0 to n)
    machines_done = 0
    while total_time <= max_time:
        #free just freed parts
        for i in range(parts_status.shape[0]):
            if parts_status[i] == -2:
                parts_status[i] = -1
        for machine in machines:
            machines_done += machine.work(total_time, parts_status) 
        total_time += 1
        if machines_done == machines.shape[0]:
            break
    
    unfinished_tasks = np.sum([int(task.is_done == 0) for machine in tasks_copy for task in machine])
    total_tasks = np.sum([len(machine) for machine in tasks_copy])
    finished_tasks = total_tasks - unfinished_tasks

    if unfinished_tasks != 0:
        print("error")
    return unfinished_tasks, finished_tasks, total_time, tasks_copy

def fitness_function(chromosome, tasks, max_time, parts):
    unfinished_tasks, finished_tasks, total_time, tasks_copy = simulation(chromosome, tasks, max_time, parts)

    result = TIME_WEIGHT*(max_time - total_time) - UNFINISHED_WEIGHT*unfinished_tasks + FINISHED_WEIGHT*finished_tasks
    #if result == -35:
    #    print("35_error")
    return result

def sort_by_ff(chromosomes, tasks, max_time, parts):
    evaluation_function = partial(fitness_function, tasks= tasks, max_time= max_time, parts= parts)
    sorted_chromosomes = sorted(chromosomes, key=evaluation_function, reverse=True)
    #print([evaluation_function(i) for i in sorted_chromosomes])
    return sorted_chromosomes

def create_gantt_chart(tasks, max_time, io_dir):
    fig, ax = plt.subplots(figsize=(max(12,0.2*max_time), 8))
    colors = ['#%06X' % rnd.randint(0, 0xFFFFFF) for _ in range(sum(len(machine_tasks) for machine_tasks in tasks))]

    color_index = 0
    y_labels = []
    y_positions = []
    for machine_tasks in tasks:
        for task in machine_tasks:
            if task.started_at != -1 and task.finished_at != -1:
                y_label = f'P{task.part}, M{task.machine}'
                y_labels.append(y_label)
                y_positions.append(y_label)
                
                color = colors[color_index]
                ax.barh(y_label, task.finished_at - task.started_at + 1, 
                        left=task.started_at, height=0.4, align='center', color=color, edgecolor='black')
        color_index += 1

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)

    ax.grid(True)
    ax.set_xlabel('Time')
    ax.set_ylabel('Tasks (Part, Machine)')
    ax.set_title('Gantt Chart')

    max_time = max(task.finished_at for machine_tasks in tasks for task in machine_tasks if task.finished_at != -1) + 1
    plt.xticks(range(0, max_time + 1, 1))

    plt.tight_layout()
    plt.savefig(io_dir + '/gantt.png')
    plt.close()

def iterate(current_gen, tasks, max_time, parts):
    sorted = sort_by_ff(current_gen, tasks, max_time, parts)
    best_cr = copy.deepcopy(sorted[0])
    best_score = fitness_function(best_cr, tasks, max_time, parts)
    next_gen = generate_G1(sorted)
    sorted = sort_by_ff(next_gen, tasks, max_time, parts)
    bff = fitness_function(sorted[0], tasks, max_time, parts)
    bff_best_score = fitness_function(best_cr, tasks, max_time, parts)
    if best_cr == sorted[0] and best_score != bff:
        bff_best_score = fitness_function(best_cr, tasks, max_time, parts)
        best_score = fitness_function(sorted[0], tasks, max_time, parts)
        print('error')
    return next_gen, best_score

def plot_hill_climbing_figure(best_evaluations, io_dir):
    plt.figure()
    time = list(range(len(best_evaluations)))
    plt.plot(time, best_evaluations)
    plt.xlabel('Generations')
    plt.ylabel('Best score')
    plt.savefig(io_dir + r'/hill_climbing.png')
    plt.close()

def output_solution(filename, tasks):
    buffer = ''
    for task_list in tasks:
        for task in task_list:
            buffer += f'{task.machine} {task.part} {task.started_at} {task.finished_at}\n'
    with open(filename, 'w') as file:
        file.write(buffer)
    
if __name__ == '__main__':
    #io_dir = sys.argv[1]
    io_dir = './test_cases/test6'
    if not os.path.isdir(io_dir):
        raise FileNotFoundError(f"The directory does not exist.")
    MACHINES, num_parts, parts = read_file(io_dir + r'/input.txt')
    if MACHINES <= 0:
        print('There are 0 machines. Nothing to do.')
        sys.exit()
    if num_parts != len(parts):
        print('There is a mismatch between the stated and actual number of parts. Program has been terminated.')
        sys.exit()
    if not any(parts):
        print('One of the parts has no opertaions for it. Fix the typo or remove this part from the input file.')
        sys.exit()
    if num_parts <= 0 or len(parts) <= 0:
        print('There are 0 parts. Nothing to do.')
        sys.exit()
        
    tasks = generate_tasks(parts)
    current_gen = generate_G0(tasks, CHROMOSOMES)
    max_time = get_max_time(tasks)
    best_evaluations = []
    for _ in range(GENERATIONS):
        #print status
        percent_completed = (_+ 1) / GENERATIONS * 100
        print(f"Progress: {percent_completed:.2f}% completed", end='\r')
        #iterate
        current_gen, best_score= iterate(current_gen, tasks, max_time, parts)
        best_evaluations.append(best_score)

    #get the best solution
    sorted = sort_by_ff(current_gen, tasks, max_time, parts)
    unfinished_tasks, finished_tasks, total_time, tasks_copy = simulation(sorted[0], tasks, max_time, parts)
    output_solution(io_dir + "/output.txt", tasks_copy)
    create_gantt_chart(tasks_copy,max_time, io_dir)
    plot_hill_climbing_figure(best_evaluations, io_dir)