import random
import matplotlib.pyplot as plt
import csv

from vpython.no_notebook import stop_server
import math as math
import numpy as np
import copy as copy
# Jianfei Pan and Zhengdong Liu
  # N by N by N array of atoms

lengh_springs = 0.1 # in meters
spacing = lengh_springs
atom_radius = 0.05 * spacing

# constants for the bouncing and breathing cubes
gravity = np.array([0,0,-9.81])
dt = 0.001 # simulation timestep# damping = 0.999 # damping coefficient
Global_time = 0
friction_mu_s = 1 # static friction coefficient
friction_mu_k = 0.8 # dynamic friction coefficient
#spring_const = 1000 # spring constant of the edges
sprint_const_ground = 25000# spring constant of the ground
individual_mass = 0.1 # each mass, metrics in kg


class atom_new:
    def __init__(self):
        self.pos = np.array([0, 0, 0], dtype = 'float64')
        # atom.rotate(angle=a, axis=vec(b, c, d), origin=vector(0.05, 0.05, 0.05))
        self.vel = np.array([0, 0, 0], dtype = 'float64')  # velocity #.astype('float64')
        self.acc = np.array([0, 0, 0], dtype = 'float64')  # acceleration
        self.spring_force = np.array([0, 0, 0], dtype = 'float64')  # spring force
        self.restore_force = np.array([0, 0, 0], dtype = 'float64')
        self.mass = -1
        self.radius = -1
        self.index = -1  # len(self.atoms)
        self.visible = -1
class spring_new:
    def __init__(self):
        self.pos = -1
        self.axis = -1
        self.orig_axis = -1
        self.thickness = -1
        self.radius = -1
        self.spring_k = -1
        self.material = -1
        # breathing = math.sin(0.1*pi*Global_time) add a sinusoid to achieve the breathing of the cubes
        self.start = -1
        self.end = -1
        self.color = -1
        self.bounds = -1
class individual:
    def __init__(self):
        self.centroids = []
        self.shape = []
        self.age = 0
        self.fitness = 0
        self.size = []
        self.num_blocks = 0
class crystal:
    def __init__(self, atom_radius, spacing, mass, coord_array, spring_material): #coord_array contains an array of xs, ys, zs
        self.atoms = []
        self.springs = []
        self.springs_endpoints = []
        self.atoms_pos = {}
        self.make_atoms( atom_radius, spacing, mass, coord_array, spring_material)

        # Create (N+2)^3 atoms in a grid; the outermost atoms are fixed and invisible
    def make_atoms(self,  atom_radius, spacing, mass, coord_array, spring_material):
        for num_crystal in range(len(coord_array)):
            x_axis = int(coord_array[num_crystal][0])
            y_axis = int(coord_array[num_crystal][1])
            z_axis = int(coord_array[num_crystal][2])
            repeated_atoms = []
            repeated_atoms_index = []
            current_atoms = [] #should be 8 atoms
            index = 0
            for z in range(z_axis, z_axis + 2, 1):
                for y in range(y_axis, y_axis + 2, 1):
                    for x in range(x_axis, x_axis + 2, 1):
                        atom_pos = tuple(np.array([x, y, z]) * spacing)
                        if atom_pos in self.atoms_pos.keys():
                            repeated_atoms.append(self.atoms_pos[atom_pos])
                            repeated_atoms_index.append(index)
                            atom = self.atoms_pos[atom_pos]
                        else:
                            #atom = sphere()
                            atom = atom_new()
                            atom.pos = np.array([x, y, z]) * spacing
                            #atom.rotate(angle=a, axis=vec(b, c, d), origin=vector(0.05, 0.05, 0.05))
                            atom.vel = np.array([0, 0, 0])  # velocity
                            atom.acc = np.array([0, 0, 0])  # acceleration
                            atom.spring_force = np.array([0, 0, 0])  # spring force
                            atom.restore_force = np.array([0, 0, 0])
                            atom.mass = mass
                            atom.radius = atom_radius
                            #atom.color = color.black
                            atom.index = index #len(self.atoms)
                            self.atoms.append(atom)
                            self.atoms_pos[tuple(np.array([x, y, z])* spacing)] = atom
                        index += 1
                        current_atoms.append(atom)

            self.build_springs(current_atoms, repeated_atoms_index, spring_material)

    def build_springs(self, atom_list, repeated_atoms_index, spring_material):
        # four types of materis

        list_m = [0, 1, 2, 3, 4, 5, 6, 7]
        springs_list = []
        for i in list_m:
            if i + 1 < 8:
                for j in list_m[i + 1:]:
                    springs_list.append([atom_list[i], atom_list[j]])
        for each_pair in springs_list:
            if each_pair[0].index not in repeated_atoms_index or each_pair[1].index not in repeated_atoms_index:
                self.make_spring(each_pair[0], each_pair[1],spring_material)

    # # Create a grid of springs linking each atom to the adjacent atoms
    # in each dimension, or to invisible motionless atoms
    def make_spring(self, start, end, spring_material):
        #spring = cylinder()
        spring = spring_new()
        spring.pos = start.pos
        spring.axis = end.pos - start.pos
        spring.orig_axis = end.pos - start.pos
        spring.thickness = 0.06
        spring.radius = 0.1 * atom_radius
        spring.spring_k = spring_material[0]
        spring.material = []
        # breathing = math.sin(0.1*pi*Global_time) add a sinusoid to achieve the breathing of the cubes

        spring.start = start
        spring.end = end
        #spring.color = color.gray(0.5)
        spring.bounds = (start.pos, end.pos)
        self.springs.append(spring)
        self.springs_endpoints.append((start.pos, end.pos))

def get_center(all_atoms):
        atom_postion=np.array([0.0,0.0,0.0])
        for atom in all_atoms:
            atom_postion += atom.pos/len(all_atoms)
        #print(atom_postion)
        return atom_postion
def initialization(num_centroids, num_pop, materials_types, max_num = 7):
    population_initial = []
    for i in range(num_pop):
        # initialize random shapes
        new_sample = individual()
        shape, size, _ = random_config(max_num) # size is range_array = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        new_sample.size = copy.deepcopy(size)
        new_sample.shape = copy.deepcopy(shape) # coordinates array
        new_sample.num_blocks = max_num # number of blocks
        # max_num is the number of blocks for the initial structure
        xmin = size[0][0]
        xmax = size[0][1]
        ymin = size[1][0]  # np.amin(vertices_array, axis=0)  # find minimum along the column direction
        ymax = size[1][1]
        zmin = size[2][0]
        zmax = size[2][1]  # np.amax(vertices_array, axis=0)
        # initialize the table storing the parameters
        x = np.linspace(xmin, xmax, 3)
        y = np.linspace(ymin, ymax, 3)
        z = np.linspace(zmin, zmax, 3)

        # initialize random centroids (initially evenly distributed)
        for j in range(num_centroids):
            centroid = []
            x_index = np.random.randint(0, 2)
            y_index = np.random.randint(0, 2)
            z_index = np.random.randint(0, 2)
            centroid.append(np.random.uniform(x[x_index], x[x_index+1]))   # generate random x
            centroid.append(np.random.uniform(y[y_index], y[y_index+1]))   # generate random y
            centroid.append(np.random.uniform(z[z_index], z[z_index+1]))   # generate random z
            rand_type = np.random.randint(4)
            centroid.append(materials_types[rand_type])
            new_sample.centroids.append(centroid)
        population_initial.append(new_sample)
    return population_initial
def random_config(max_num):
    robot = {}
    block = tuple([0, 0, 0])
    robot[block] = None

    while np.array(list(robot.keys())).shape[0] != max_num:
        rand_block = np.random.randint(0, 3)
        existing = np.array(list(robot.keys()))
        existing = existing.reshape(-1, 3)
        block = existing[np.argmax(existing[:, rand_block])]
        new_block = list(block)
        new_block = random_extensions(new_block)
        robot[tuple(new_block)] = block    # block is the parent block.
        #print('tuple(new_block)',i, '  ', tuple(new_block))

    list_masses = np.array(list(robot.keys()))
    x_max = 0.1 * np.amax(list_masses[:, 0])
    x_min = 0.1 * np.amin(list_masses[:, 0]) - 0.1
    y_max = 0.1 * np.amax(list_masses[:, 1])
    y_min = 0.1 * np.amin(list_masses[:, 1])
    z_max = 0.1 * np.amax(list_masses[:, 2])
    z_min = 0.1 * np.amin(list_masses[:, 2])
    range_array = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    shape = []
    for i in list(robot.keys()):
        shape.append(list(i))
    return shape, range_array, robot
def mutation(population, materials_types, num_centroids):
    mutated_pop = copy.deepcopy(population)
    # mutation begins
    for i in range(len(mutated_pop)):
        object_size = mutated_pop[i].size
        mutated_centroid = np.random.randint(num_centroids)
        index = np.random.randint(4)
        if index == 3:  # mutate material types
            rand_type = np.random.randint(4)
            current_material = mutated_pop[i].centroids[mutated_centroid][3]
            current_material_index = materials_types.index(current_material)
            while rand_type == current_material_index:
                rand_type = np.random.randint(4)
                print('excuting while')

            mutated_pop[i].centroids[mutated_centroid][index] = materials_types[rand_type]
        else:  # mutate positions
            # if np.random.randint(2) == 0:
            mutated_pop[i].centroids[mutated_centroid][index] = np.random.uniform(object_size[index][0],
                                                                                  object_size[index][1])
    return mutated_pop
def random_extensions(new_block):
    rand = np.random.randint(0, 2)
    rd_dir = np.random.randint(0, 3)
    if rand and new_block[rd_dir] > 0:
        new_block[rd_dir] -= 1
    else:
        new_block[rd_dir] += 1
    return new_block
def add_block(coords_array):
    coords_array = np.array(coords_array)
    rand_dim = np.random.randint(0, 3)
    chosen_block = coords_array[np.argmax(coords_array[:, rand_dim])]
    extended_block = random_extensions(list(chosen_block))
    i = 0
    while (extended_block in coords_array) and i < 8: # try 8 random attempts
        rand_dim = np.random.randint(0, 3)
        chosen_block = coords_array[np.argmax(coords_array[:, rand_dim])]
        extended_block = random_extensions(list(chosen_block))
        i += 1
        #print('extending')
    if extended_block in coords_array:
        max_dim = np.random.randint(0, 3)
        chosen_block = coords_array[np.argmax(coords_array[:, rand_dim])]
        extended_block = list(chosen_block)
        extended_block[max_dim] += 1

    shape = []
    for i in coords_array:
        shape.append(list(i))
    shape.append(extended_block)
    return shape
def mutation_config(population):
    mutated_pop = copy.deepcopy(population)
    # mutation begins
    for i in range(len(mutated_pop)):
        mutated_pop[i].shape = list(mutated_pop[i].shape)
        if mutated_pop[i].num_blocks >= 12:
            # randomly remove one cube
            mutated_pop[i].shape.remove(mutated_pop[i].shape[np.random.randint(0, mutated_pop[i].num_blocks)])
            mutated_pop[i].num_blocks -= 1
        elif mutated_pop[i].num_blocks <= 6:
            # randomly add one cube
            mutated_pop[i].shape = add_block(mutated_pop[i].shape)
            mutated_pop[i].num_blocks += 1
        else:
            is_adding = np.random.randint(0, 2)
            if is_adding:
                # add a block
                mutated_pop[i].num_blocks += 1
                mutated_pop[i].shape = add_block(mutated_pop[i].shape)
            else:
                # remove a block
                mutated_pop[i].num_blocks -= 1
                mutated_pop[i].shape.remove(mutated_pop[i].shape[np.random.randint(0, mutated_pop[i].num_blocks)])
    return mutated_pop
def crossover_2_point(population, num_centroid):
    crossovered_pop = copy.deepcopy(population)
    num_pop = len(population)
    point1 = num_centroid//3
    point2 = 2*point1
    np.random.shuffle(crossovered_pop) # this shuffles along the first dimension
    for i in range(num_pop//2):
        if (2*i + 1) <= (num_pop-1):
            crossovered_pop[2 * i].centroids[point1:point2], crossovered_pop[2*i + 1].centroids[point1:point2] = crossovered_pop[2*i + 1].centroids[point1:point2], crossovered_pop[2 * i].centroids[point1:point2]
    return crossovered_pop

def evolve(c, individual, time):
    Fg = individual_mass * gravity  # The floor is at the place of z = 0
    count = 0
    position_n = np.array([0, 0, 0])
    Global_time = 0
    while Global_time < time:
        # sleep(0)
        count += 1
        # Fc = vector(0, 0, 0)  # upward restoring force due to collision
        # print('len(c.atoms)', len(c.atoms))
        for atom in c.atoms:
            if atom.pos[2] < -0.01:
                # firstly added up x and y directions forces
                Fp = np.array([atom.spring_force[0], atom.spring_force[1], 0])
                # Fp_axis = Fp.norm()
                if np.linalg.norm(Fp) != 0: Fp_axis = Fp / np.linalg.norm(Fp)
                Fn = np.array([0, 0, atom.spring_force[2] + Fg[2]])
                if np.linalg.norm(Fp) < np.linalg.norm(Fn) * friction_mu_s:
                    atom.restore_force = np.array([0, 0, abs(-0.01 - atom.pos[2]) * sprint_const_ground]) - Fp
                else:
                    atom.restore_force = np.array([0, 0, abs(-0.01 - atom.pos[2]) * sprint_const_ground]) - Fp_axis * np.linalg.norm(Fn) * friction_mu_k
            else:
                atom.restore_force = np.array([0, 0, 0])

        for atom in c.atoms:
            # else:
            # atom.acc = vector(0, 0.1, 0)
            if atom.visible:
                # Tally all forces on each mass(atom)
                #print(atom.restore_force)
                Ft = atom.spring_force + Fg + atom.restore_force
                atom.acc = Ft / individual_mass
                atom.vel = atom.vel.astype('float64')
                atom.vel += atom.acc * dt
                # atom.vel *= 0.999  # add damping by multiplying a small constant
                atom.pos += atom.vel * dt
        for atom in c.atoms:
            atom.spring_force = np.array([0, 0, 0])

        for spring in c.springs:
            spring.pos = spring.start.pos
            spring.axis = spring.end.pos - spring.start.pos
            L = np.linalg.norm(spring.end.pos - spring.start.pos)
            if L == 0: L = 0.0001
            norm_axis = spring.axis / L
            spring.length = L
            if Global_time == 0:
                spring.orig_len = np.linalg.norm(spring.end.pos - spring.start.pos)
            #print(spring.material)
            #     spring.length = mag(end.pos - start.pos) # this fixed and presents the original length
            #     spring.ori_length = mag(end.pos - start.pos) #*math.sin(0.000001*Global_time) # 0.1 and 0.001 are arbitrary
            spring.ori_length = spring.orig_len * (1 + spring.material[1] * math.sin(2 * np.pi * Global_time + spring.material[2]))

            # If spring is extended,  (orig_len*0.8+orig_len*0.1 * math.sin(0.0001* Global_time + material_type[2]))
            spring.start.spring_force = spring.start.spring_force.astype('float64')
            spring.end.spring_force = spring.end.spring_force.astype('float64')
            spring.start.spring_force += norm_axis * (spring.spring_k * (L - spring.ori_length))
            spring.end.spring_force += -1 * norm_axis * (spring.spring_k * (L - spring.ori_length))
            # spring.pos = spring.start.pos
        Global_time += dt
        vel_total = np.sqrt(get_center(c.atoms)[1]**2+get_center(c.atoms)[0]**2)/3
        individual.fitness = vel_total
    return vel_total
def evaluation(centroids, c):
    for spring in c.springs:
        x = (spring.start.pos[0] + spring.end.pos[0]) / 2
        y = (spring.start.pos[1] + spring.end.pos[1]) / 2
        z = (spring.start.pos[2] + spring.end.pos[2]) / 2
        dist = []
        for k in range(len(centroids)):
            current_centroid = centroids[k]
            current_x = current_centroid[0]
            current_y = current_centroid[1]
            current_z = current_centroid[2]
            this_dist=((x - current_x) ** 2 + (y - current_y) ** 2 + (z - current_z) ** 2)**0.5
            if this_dist < 0.05:
                dist.append(this_dist)
                break
            else:
                dist.append(this_dist)
        min_value = min(dist)
        min_index = dist.index(min_value)
        spring.material = centroids[min_index][3]
        #print('spring.material[0]', spring.material[0])
        spring.spring_k = spring.material[0]
    return None
def sift(list, low, high):  # 排一个数
    i = low
    j = 2 * i + 1
    tmp = list[low]  # 堆顶
    while j <= high:  # 只要j节点有数，就一直循环
        if j + 1 <= high and list[j + 1] < list[j]:
            j = j + 1
        if list[j] < tmp:
            list[i] = list[j]
            i = j
            j = 2 * i + 1
        else:
            list[i] = tmp
            break
    else:
        list[i] = tmp
def heap_sort(list):
    n = len(list)
    for i in range((n - 2) // 2, -1, -1):
        # i 表示建堆的时候调整
        sift(list, i, n - 1)
    for i in range(n - 1, -1, -1):
        list[0], list[i] = list[i], list[0]
        sift(list, 0, i - 1)
    return list
def rank_selection(population, percent, vel_final):
    length = int(percent * len(population))
    #print(length)
    rank_vel_index = np.argsort(-np.array(vel_final))
    #print('rankk_index', rank_vel_index)
    selected_pop = []
    #print('vel_final', vel_final)
    #print('len population', len(population))
    rank_vel = heap_sort(vel_final)
    for i in range(length):
        selected_pop.append(population[rank_vel_index[i]])
    return rank_vel, selected_pop
def diversity(population):
    diversity_val = 0
    #print(len(population))
    for i in range(len(population) - 1):
        individual_pop = population[i].centroids
        for j in range(i + 1, len(population)):
            compare_pop = population[j].centroids
            for k in range(len(individual_pop)):
                check = 0
                for z in range(len(compare_pop)):
                    if individual_pop[k] == compare_pop[z]:
                        check = 0
                    if individual_pop[k] != compare_pop[z]:
                        check += 1
                if check == 0:
                    diversity_val += 0
                elif check == len(compare_pop):
                    diversity_val += 1
    final_diversity = diversity_val / ((len(population) * (len(population) - 1)) * 2)
    return final_diversity
def get_center_new(coor_array):
    cube_Center = []
    center = 0
    for i in range(len(coor_array)):
        this_cube = coor_array[i]

        x = np.array(this_cube)[0] - 0.5
        y = np.array(this_cube)[1] + 0.5
        z = np.array(this_cube)[2] + 0.5
        cube_center_each = np.array([x, y, z])
        # print('cube_center_each', cube_center_each)

        cube_Center.append(list(cube_center_each))
        center += cube_center_each / len(coor_array)
    return cube_Center, center
def calibrate(center1, center2, part1, part2):
    cali_factor = np.array([(center1[0] - center2[0]), (center1[1] - center2[1]), (center1[2] - center2[2])])
    #print('cali_factor', cali_factor)
    for i in range(len(part1)):
        part1[i] = list(np.array(part1[i]) - cali_factor)
    for i in range(len(part2)):
        part2[i] = list(np.array(part2[i]) + cali_factor)
    return part1, part2
def assign_Crosscube(shape):
    coor_array = shape
    cube_center, center = get_center_new(coor_array)
    # print(center)
    dist = []
    for i in range(len(cube_center)):
        current_cube = cube_center[i]
        x = current_cube[0]
        y = current_cube[1]
        z = current_cube[2]
        dist_indi = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
        dist.append(dist_indi)
    min_dist = min(dist)
    index = dist.index(min_dist)
    selected = np.array(cube_center[index])
    x = int(selected[0] + 0.5)
    y = int(selected[1] - 0.5)
    z = int(selected[2] - 0.5)
    selected_cube = [x, y, z]
    #print('selected_cube', selected_cube)
    return selected_cube
def get_parts(selected_cube, coor_array):
    parts = []
    for i in range(len(coor_array)):
        this_cube = coor_array[i]
        # print('this_cube',this_cube)
        if this_cube[0] >= selected_cube[0] and this_cube[1] >= selected_cube[1] and this_cube[2] >= selected_cube[2]:
            parts.append(this_cube)
    return parts
def swap_parts(shape1,shape2,part1,part2,first_cube,second_cube):
    for i in range(len(part1)):
        if part1[i] in shape1:
            shape1.remove(part1[i])
    for i in range(len(part2)):
        if part2[i] in shape2:
            shape2.remove(part2[i])
    # print('after removal')
    # print(shape1)
    # print(shape2)
    part1, part2 = calibrate(first_cube, second_cube, part1, part2)
    for i in range(len(part1)):
        shape2.append(part1[i])
    for i in range(len(part2)):
        shape1.append(part2[i])
    # print('after addiation')
    # print(shape1)
    # print(shape2)
    return shape1 , shape2
def cross_shape(mutated_population):
    np.random.shuffle(mutated_population)

    size = int(len(mutated_population) / 2)
    # crossed_pop=[0]*len(mutated_population)
    for i in range(size):
        pop_to_cross = copy.copy(mutated_population)
        # print('pop_to_cross', pop_to_cross)
        first_robot = mutated_population[i].shape
        first_cube = assign_Crosscube(first_robot)
        part1 = get_parts(first_cube, first_robot)
        # print('part1',part1)

        # second robot
        second_robot = mutated_population[i + 1].shape
        second_cube = assign_Crosscube(second_robot)
        part2 = get_parts(second_cube, second_robot)
        new_shape1, new_shape2 = swap_parts(first_robot, second_robot, part1, part2, first_cube, second_cube)

        mutated_population[i].shape = new_shape1
        mutated_population[i].num_blocks = len(new_shape1)
        mutated_population[i].size = update_size(new_shape1)
        mutated_population[i + size].shape = new_shape2
        mutated_population[i + size].num_blocks = len(new_shape2)
        mutated_population[i + size].size = update_size(new_shape2)
    return mutated_population
def update_size(shape1):
    shape1=np.array(shape1)
    x_max = 0.1 * np.amax(shape1[:, 0])
    x_min = 0.1 * np.amin(shape1[:, 0]) - 0.1
    y_max = 0.1 * np.amax(shape1[:, 1])
    y_min = 0.1 * np.amin(shape1[:, 1])
    z_max = 0.1 * np.amax(shape1[:, 2])
    z_min = 0.1 * np.amin(shape1[:, 2])
    range_array = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    return range_array
def populate_centroids(population, num_centroids):
    for i in range(len(population)):
        size = population[i].size
        # max_num is the number of blocks for the initial structure
        xmin = size[0][0]
        xmax = size[0][1]
        ymin = size[1][0]  # np.amin(vertices_array, axis=0)  # find minimum along the column direction
        ymax = size[1][1]
        zmin = size[2][0]
        zmax = size[2][1]  # np.amax(vertices_array, axis=0)
        x = np.linspace(xmin, xmax, 3)
        y = np.linspace(ymin, ymax, 3)
        z = np.linspace(zmin, zmax, 3)
        centroids = []
        for j in range(num_centroids):
            centroid = []
            x_index = np.random.randint(0, 2)
            y_index = np.random.randint(0, 2)
            z_index = np.random.randint(0, 2)
            centroid.append(np.random.uniform(x[x_index], x[x_index + 1]))  # generate random x
            centroid.append(np.random.uniform(y[y_index], y[y_index + 1]))  # generate random y
            centroid.append(np.random.uniform(z[z_index], z[z_index + 1]))  # generate random z
            rand_type = np.random.randint(4)
            centroid.append(materials_types[rand_type])
            centroids.append(centroid)
        population[i].centroids = copy.deepcopy(centroids)
    return population

def pareto_fit_age_selection(population, fitness, pareto_front, num_pop):  # percent_pop):
    selected_pop = []
    pareto_front_psedo = []
    population_ranked = copy.deepcopy(population)
    if pareto_front == []:
        oldest = []
        age = 0
        oldest_fitness = 0
        for i in range(len(population)):
            if population[i].age >= age:
                oldest = population[i]
                age = population[i].age
                oldest_fitness = fitness[i]
        oldest.fitness = oldest_fitness
        #population[np.argmax(fitness)][num_centroids + 1] = np.amax(fitness)
        pareto_front.append(oldest)
        pareto_front.append(population[np.argmax(fitness)])  # set up the initial pareto front

    for i in range(len(population)):
        for j in range(len(pareto_front)):
            if fitness[i] > pareto_front[j].fitness and population[i].age > pareto_front[j].age:  # fitness and age
                population[i].fitness = fitness[i]
                pareto_front_psedo.append(population[i])
                pareto_front.remove(pareto_front[j])
                break
            elif fitness[i] > pareto_front[j].fitness or population[i].age > pareto_front[j].age:  # fitness or age
                is_append = True
                for r in range(j, len(pareto_front)):
                    if fitness[i] > pareto_front[r].fitness and population[i].age > pareto_front[r].age:
                        is_append = False
                        break
                if is_append:
                    population[i].fitness = fitness[i]
                    pareto_front_psedo.append(population[i])
                break
    # select within the selected
    if pareto_front_psedo:
        tobe_removed = []
        for i in range(len(pareto_front_psedo)):
            for j in range(len(pareto_front_psedo)):
                if pareto_front_psedo[j].fitness > pareto_front_psedo[i].fitness and \
                        pareto_front_psedo[j].age > pareto_front_psedo[i].age:  # 6 is fitness and 5 is age
                    if pareto_front_psedo[i] not in tobe_removed:
                        tobe_removed.append(pareto_front_psedo[i])
                    break
        selected_pop = copy.deepcopy(tobe_removed)
        for removed_item in tobe_removed:
            pareto_front_psedo.remove(removed_item)
        pareto_front.extend(pareto_front_psedo)

    if selected_pop and len(selected_pop) >= num_pop:
        for front_state in selected_pop:
            front_state.age += dt
        each_diversity = 0
        returned_pop = []
        selected_pop = selected_pop + pareto_front
        for asd in range(5):
            np.random.shuffle(selected_pop)
            if diversity(selected_pop[0:num_pop]) > each_diversity:
                returned_pop = selected_pop[0:num_pop]
                each_diversity = diversity(selected_pop[0:num_pop])
        return returned_pop

    else:
        num_pop_needed = num_pop - len(
            selected_pop)  # output population in the pareto plot and the population with top fitness values
        percent_pop = num_pop_needed / len(population_ranked)
        _, selected_p = rank_selection(population_ranked, percent_pop, fitness)
        for selected_state in selected_p:
            print(selected_state)
            selected_state.age += dt
        selected_p.extend(selected_pop)
        return selected_p

def initialization_centroids(sample, num_pop, materials_types):
    population_initial = []
    for i in range(num_pop):
        # initialize random shapes
        new_sample = individual()
        size = sample.size
        new_sample.size = copy.deepcopy(size)
        new_sample.shape = copy.deepcopy(sample.shape) # coordinates array
        new_sample.num_blocks = sample.num_blocks # number of blocks
        # max_num is the number of blocks for the initial structure
        xmin = size[0][0]
        xmax = size[0][1]
        ymin = size[1][0]  # np.amin(vertices_array, axis=0)  # find minimum along the column direction
        ymax = size[1][1]
        zmin = size[2][0]
        zmax = size[2][1]  # np.amax(vertices_array, axis=0)
        # initialize the table storing the parameters
        x = np.linspace(xmin, xmax, 3)
        y = np.linspace(ymin, ymax, 3)
        z = np.linspace(zmin, zmax, 3)

        # initialize random centroids (initially evenly distributed)
        for j in range(num_centroids):
            centroid = []
            x_index = np.random.randint(0, 2)
            y_index = np.random.randint(0, 2)
            z_index = np.random.randint(0, 2)
            centroid.append(np.random.uniform(x[x_index], x[x_index+1]))   # generate random x
            centroid.append(np.random.uniform(y[y_index], y[y_index+1]))   # generate random y
            centroid.append(np.random.uniform(z[z_index], z[z_index+1]))   # generate random z
            rand_type = np.random.randint(4)
            centroid.append(materials_types[rand_type])
            new_sample.centroids.append(centroid)
        population_initial.append(new_sample)
    return population_initial

def evol_centroids_pop(one_individual, num_pop, num_eval, materials_types, pareto_front, num_centroids):
    population = []
    population.append(one_individual)
    for i in range(num_pop-1):
        population.extend(mutation([one_individual], materials_types, num_centroids))

    for i in range(num_eval):
        population_mutate = mutation(population, materials_types, num_centroids)
        crossed_pop = crossover_2_point(population_mutate, num_centroids)
        new_pop = crossed_pop + population
        # evolve centroids for 5 times before evolving the shapes again
        vel_final = []
        for j in range(len(new_pop)):
            print('the', j, 'th population centroid')
            spring_material = materials_types[np.random.randint(0, 4)]  # assign a random material type
            current_individual = new_pop[j]  # get current population
            coord_array = current_individual.shape
            # print('coord_array.....;.;.;', coord_array)
            c = crystal(atom_radius, spacing, individual_mass, coord_array, spring_material)
            evaluation(current_individual.centroids, c)
            vel_individual = evolve(c, current_individual, time)
            vel_final.append(vel_individual)
        population = pareto_fit_age_selection(new_pop, vel_final, pareto_front, num_pop)

        final_vel_p = []
        for ind in range(len(population)):
            final_vel_p.append(population[ind].fitness)
        current_best = np.amax(final_vel_p)
        best_individual = population[np.argmax(final_vel_p)]
    return current_best, best_individual
def evol_centroids_crossed(one_individual, num_pop, num_eval, materials_types, pareto_front, num_centroids):
    population = initialization_centroids(one_individual, num_pop, materials_types)
    for i in range(num_eval):
        population_mutate = mutation(population, materials_types, num_centroids)
        crossed_pop = crossover_2_point(population_mutate, num_centroids)
        new_pop = crossed_pop + population
        # evolve centroids for 5 times before evolving the shapes again
        vel_final = []
        for j in range(len(new_pop)):
            print('the', j, 'th population centroid')
            spring_material = materials_types[np.random.randint(0, 4)]  # assign a random material type
            current_individual = new_pop[j]  # get current population
            coord_array = current_individual.shape
            # print('coord_array.....;.;.;', coord_array)
            c = crystal(atom_radius, spacing, individual_mass, coord_array, spring_material)
            evaluation(current_individual.centroids, c)
            vel_individual = evolve(c, current_individual, time)
            vel_final.append(vel_individual)
        population = pareto_fit_age_selection(new_pop, vel_final, pareto_front, num_pop)

        final_vel_p = []
        for ind in range(len(population)):
            final_vel_p.append(population[ind].fitness)
        current_best = np.amax(final_vel_p)
        best_individual = population[np.argmax(final_vel_p)]

    return current_best, best_individual
def diversity_shape(population):
    count=0
    for i in range(len(population)-1):
        this_shape=population[i].shape
        for j in range(i + 1, len(population)-1):
            compare_shape=population[j].shape
            for k in range(len(this_shape)):
                if this_shape[k] not in compare_shape:
                    count+=1
    return count/len(population)
f = open('C:/Users/18462/Desktop/ME/MECE 4510ea/ass3/3c/fitness_hc10.csv', 'w')

writer_f = csv.writer(f)

# Make the ground contact
thk = 0.001
side = 10
s3 = 2*side + thk

materials_types = [[1000, 0, 0], [20000, 0, 0], [5000, 0.25, 0], [5000, 0.25, np.pi]]



num_pop_cen = 3
num_centroids = 5


vertices_array = [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 2, 1], [0, 2, 0]]
tissue = [1000, 0, 0]
bone = [20000, 0, 0]
muscle2 = [5000, 0.5, 0]
muscle1 = [5000, 0.5, np.pi]
#coord_array = [[2, 0, 0], [1,0,0],[1,1,0],[1,2,0]]
#range_array = [[-0.1, 0.2], [0, 0.3], [0, 0.1]]

best_centroids = []
best_shape = []
best_vel = []
fitness = []
ini_best = 0
current_best = 0
diversity_plot = [1]
# evaluation starts
best_indi = []
vel_t = []
x_gen = []

percent = 0.5 # selection pressure
pareto_front = []
diversity_value=[]

num_pop = 1
num_eval = 3 # centroids
num_evl = 50
time = 3
spring_material = materials_types[np.random.randint(0, 4)]

population_ini = initialization(num_centroids, num_pop, materials_types, max_num=7)
c = crystal(atom_radius, spacing, population_ini[0], population_ini[0].centroids, spring_material)
evaluation(population_ini[0].centroids, c)
vel_ini = evolve(c, population_ini[0], time)

for i in range(num_evl):
    print('population_ini', population_ini)
    population_mutate_c = mutation_config(population_ini)
    population_mutate_c = populate_centroids(population_mutate_c, num_centroids)

    c = crystal(atom_radius, spacing, population_mutate_c[0], population_mutate_c[0].centroids, spring_material)
    evaluation(population_mutate_c[0].centroids, c)
    vel_m = evolve(c, population_mutate_c[0], time)
    # #vel_m = 0
    # population_mutate_s = 0
    # for j in range(num_eval):
    #     population_mutate = mutation(population_mutate_c, materials_types, num_centroids)
    #     c = crystal(atom_radius, spacing, population_mutate[0], population_mutate[0].centroids, spring_material)
    #     evaluation(population_mutate[0].centroids, c)
    #     vel_mutate_c = evolve(c, population_mutate[0], time)
    #     if vel_mutate_c > 10:
    #         vel_mutate_c = 0
    #     if vel_mutate_c > vel_m:
    #         vel_m = vel_mutate_c
    #         population_mutate_s = copy.deepcopy(population_mutate)

        #print('vel_mutate=', vel_mutate_c)

    if vel_m > vel_ini:
        fitness.append(vel_m)
        vel_ini = vel_m
        #population_ini = copy.deepcopy(population_mutate_s)
        population_ini = copy.deepcopy(population_mutate_c)
    elif vel_m <= vel_ini:
        fitness.append(vel_ini)
    print('fitness=',fitness)

xcoor = np.linspace(1, num_evl, num_evl)
xdiv = np.linspace(1, num_evl, int(num_evl / 10) + 1)
print(diversity_plot)

writer_f.writerow(fitness)

plt.figure(1)
plt.plot(xcoor, fitness)
plt.ylabel('fitness')
plt.xlabel('No. of generations')
plt.title('Fitness vs No. generations')
plt.show()
f.close()






