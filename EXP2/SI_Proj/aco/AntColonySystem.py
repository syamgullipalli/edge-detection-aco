__author__ = 'syam'

import numpy as np
import random
import logging
import cv2
import os
import time

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# Seed value for pseudo random numbers
SEED = 1

MAIN_DIRECTORY = 'RESULTS'
SUB_DIRECTORY_IMG = 'RESULTS/IMG'
SUB_DIRECTORY_PLOT  = 'RESULTS/PLOT'


class PartitionBasedAcs:
    """Ant Colony System based on partitioned problem
    """

    # Constructor
    def __init__(self, **kwargs):
        self.nants = kwargs['nants']  # Number of ants
        self.tauinit = kwargs['tauinit']  # Initial pheromone value
        self.alpha = kwargs['alpha']  # Pheromone coefficient
        self.beta = kwargs['beta']  # Heuristic coefficient
        self.rho = kwargs['rho']  # Pheromone evaporation coefficient
        self.phi = kwargs['phi']  # Pheromone decay coefficient
        self.q0 = kwargs['q0']  # Degree of exploration
        self.iter = kwargs['iter']  # Number of iterations
        self.cons = kwargs['cons']  # Number of constructions
        self.hor = kwargs['hor']  # Number of horizontal partitions
        self.ver = kwargs['ver']  # Number of vertical partitions
        self.heuristic_matrix = kwargs['heu']  # Heuristic information
        self.memory = kwargs['mem']  # Ant's memory (Number of positions of last visited pixels)
        self.debug = kwargs['debug']

        # Max value in heuristic matrix (Used for normalization in heuristic matrix)
        self.V_max = np.max(self.heuristic_matrix)

        # Set logging
        log.propagate = self.debug

        # Initialize pheromone matrix
        self.pheromone_matrix = np.ndarray(shape=self.heuristic_matrix.shape, dtype=float)
        self.pheromone_matrix.fill(self.tauinit)  # Set all initial pheromone value to tauinit
        log.debug("Initialized pheromone matrix")

        # TODO: Move
        self.image_matrix = np.zeros(shape=self.heuristic_matrix.shape, dtype=float)

        # All the visited positions are maintained
        self.all_visited_positions = []

        # Boundaries of image
        self.boundary = {
            'min': (0, 0),
            'max': (self.heuristic_matrix.shape[0] - 1, self.heuristic_matrix.shape[1] - 1)
        }

        # Pixels in each segment
        self.segment = {
            'hor': ((self.heuristic_matrix.shape[0]) / self.hor),
            'ver': ((self.heuristic_matrix.shape[1]) / self.ver)
        }

        # Create ants data structure
        self.ants = np.empty(shape=(self.hor, self.ver, self.nants),
                             dtype=[('position', 'i8', 2),
                                    ('boundary', 'i8', (2, 2)),
                                    ('visited', object),
                                    ('pheromone', np.float64, 1)
                             ]
        )

        # Set SEED for random
        random.seed(SEED)

        # Ant initial positions on max values of heuristics
        # positions = self.nMaxPos(array=self.heuristic_matrix, count=self.nants)

        # i = 0

        # Image matrix
        img_mat = np.loadtxt(fname="./gray_values.txt", dtype=np.uint8)

        # Ant init positions
        self.ants_init_positions = []

        # Set initial ant positions and boundaries (ant position is a random coordinate within boundary)
        for (index_x, index_y, index_z), ant in np.ndenumerate(self.ants):
            self.ants['boundary'][index_x][index_y][index_z] = [
                [index_x * self.segment['hor'],
                 index_y * self.segment['ver']
                ],
                [index_x * self.segment['hor'] + self.segment['hor'] - 1,
                 index_y * self.segment['ver'] + self.segment['ver'] - 1
                ]
            ]

            # Position ants on random positions
            # self.ants['position'][index_x][index_y][index_z] = [
            # random.randint(ant['boundary'][0][0], ant['boundary'][1][0] - 1),
            #     random.randint(ant['boundary'][0][1],ant['boundary'][1][1] - 1)]

            # Position ants on highest values of heuristics
            array = self.__arrayWithBoundary(array=self.heuristic_matrix, boundary=ant['boundary'])
            positions = self.__nMaxPos(array=array, count=self.nants)

            self.ants['position'][index_x][index_y][index_z] = list(
                map(sum, zip(ant['boundary'][0], positions[index_z])))
            # self.ants['position'][index_x][index_y][index_z] = list(positions[index_z])
            # i += 1

            # Mark initial positions as visited for each ant
            self.ants['visited'][index_x][index_y][index_z] = [tuple(self.ants['position'][index_x][index_y][index_z])]

            # Save ants init positions
            self.ants_init_positions.append(tuple(self.ants['position'][index_x][index_y][index_z]))

            # Initialize the pheromone deposited by each ant (initially zero)
            self.ants['pheromone'][index_x][index_y][index_z] = 0.0

            # Mark initial ant positions
            cv2.circle(img_mat, tuple(ant['position']), 2, (0, 0, 0))

            # Draw partitions on image
            cv2.rectangle(img_mat, tuple(ant['boundary'][0]), tuple(ant['boundary'][1]), (0, 0, 0))

        # Save the index of max heuristic value
        self.max_heuristic_index = self.nants - 1

        # Heuristic positions in decreasing heuristic value
        sorted_array = np.argsort(np.ravel(self.heuristic_matrix))[::-1]
        unravel_indices = (np.unravel_index(i, self.heuristic_matrix.shape) for i in sorted_array)
        self.heuristic_sorted_indices = [j for j in unravel_indices]

        log.debug("Initialized ants")


        # Create directory to save results
        if not os.path.exists(MAIN_DIRECTORY):
            os.mkdir(MAIN_DIRECTORY)
        if not os.path.exists(SUB_DIRECTORY_IMG):
            os.mkdir(SUB_DIRECTORY_IMG)
        if not os.path.exists(SUB_DIRECTORY_PLOT):
            os.mkdir(SUB_DIRECTORY_PLOT)

        # Display initialized image
        cv2.imshow("Init positions", img_mat)
        cv2.imwrite(os.path.join(SUB_DIRECTORY_IMG, "Initialization.png"), img_mat)
        cv2.waitKey(500)


    def __arrayWithBoundary(self, array, boundary):
        return array[boundary[0][0]:boundary[1][0]+1, boundary[0][1]:boundary[1][1]+1]


    def __chooseAndMove(self, ant, index_x, index_y, index_z):
        # Get next position to be visited
        new_pos = self.__selectNextPixel(ant=ant, index_x=index_x, index_y=index_y, index_z=index_z)

        # Add new position to all visited positions if it is not visited previously by any ant
        if new_pos not in self.all_visited_positions:
            self.all_visited_positions.append(tuple(new_pos))

        # Update ant current position and visited positions
        self.ants['position'][index_x][index_y][index_z] = list(new_pos)
        self.ants['visited'][index_x][index_y][index_z].append(tuple(new_pos))

        # Let the ant remember only last number(memory) of visited positions, and remove the remaining
        # self.ants['visited'][index_x][index_y][index_z] = self.ants['visited'][index_x][index_y][index_z][-self.memory:]


    def __current_time_milli(self):
        return int(round(time.time() * 1000))


    def __daemonActions(self, iter):
        self.__displayResults(iter=iter)
        # Ant Exchange Procedure
        self.__modifyAntPositions(iter=iter)
        self.__resetAntPheromone()


    def __displayResults(self, iter):
        # Save pheromone matrix to a text file
        np.savetxt('pheromone_matrix.txt', self.pheromone_matrix, fmt='%5f', newline='\n')

        # Min, and max values in pheromone matrix
        ph_min, ph_max = self.pheromone_matrix.min(), self.pheromone_matrix.max()

        # Difference between min and max
        diff = abs(ph_max - ph_min)

        # On each visited position apply the gray value in image matrix based on the amount of pheromone
        for position in self.all_visited_positions:
            self.image_matrix[position[0]][position[1]] = int(
                round(abs((self.__pheromone(position=position) - ph_min) / diff) * 255))

        # Save image matrix to a text file
        np.savetxt('image_matrix.txt', self.image_matrix, fmt='%d', newline='\n')

        # Load image matrix (uint8 format which is supported by opencv)
        img_mat = np.loadtxt(fname="./image_matrix.txt", dtype=np.uint8)

        # Thresholding
        org, img_mat = cv2.threshold(img_mat, 0, 255, cv2.THRESH_BINARY_INV)

        # Save image in the given directory
        cv2.imwrite(os.path.join(SUB_DIRECTORY_IMG, "Iteration" + str(iter) + ".png"), img_mat)

        # Display image
        cv2.imshow('Edge', img_mat)
        cv2.waitKey(500)

    def __heuristic(self, position):
        return float(self.heuristic_matrix[position[0]][position[1]]) / self.V_max


    def __modifyAntPositions(self, iter):
        avg_pheromone = np.average(self.ants['pheromone'])
        log.debug("Avg. Pheromone: "+str(avg_pheromone))
        np.savetxt(os.path.join(SUB_DIRECTORY_PLOT, "Iteration" + str(iter) + ".txt"), np.hstack(self.ants['pheromone']), fmt='%f', newline='\n')

        # # For each ant
        # for (index_x, index_y, index_z), ant in np.ndenumerate(self.ants):
        #     # If pheromone deposited is less than average pheromone deposited
        #     if(ant['pheromone'] < avg_pheromone):
        #         self.__moveToNewPosition(index_x=index_x, index_y=index_y, index_z=index_z)


    def __moveToNewPosition(self, index_x, index_y, index_z):
        # Empty the visited positions
        del self.ants['visited'][index_x][index_y][index_z][:]
        # Increase max heuristic index
        self.max_heuristic_index += 1
        # Move the ant to a new position where the heuristic value is high (using self.max_heuristic_index)
        self.ants['position'][index_x][index_y][index_z] = list(self.heuristic_sorted_indices[self.max_heuristic_index])
        # Mark initial positions as visited for each ant
        self.ants['visited'][index_x][index_y][index_z] = [tuple(self.ants['position'][index_x][index_y][index_z])]


    def __nMaxPos(self, array, count):
        ravel_indices = np.argsort(np.ravel(array))[-count:]
        unravel_indices = (np.unravel_index(i, array.shape) for i in ravel_indices)
        return [i for i in unravel_indices]


    def __pheromone(self, position):
        return self.pheromone_matrix[position[0]][position[1]]

    def __randomPosition(self, *boundary):
        return random.randint(boundary[0][0], boundary[1][0] - 1), random.randint(boundary[0][1], boundary[1][1] - 1)

    def __resetAntPheromone(self):
        self.ants['pheromone'] = 0.0


    def run(self):
        log.debug("Running algorithm ...")
        elapsed_time_array = [] # To store elapsed time for each iteration

        for iter in range(0, self.iter):  # For number of iterations
            log.debug("Iteration: " + str(iter + 1))
            strt_time_millis = self.__current_time_milli()  # Calculate iteration start time
            log.debug("START: "+str(strt_time_millis))
            for cons in range(0, self.cons):  # For number of construction steps
                # index_x and index_y represent partition, index_z represents ant
                for (index_x, index_y, index_z), ant in np.ndenumerate(self.ants):  # For all ants
                    # Choose and move to the next position
                    self.__chooseAndMove(ant=ant, index_x=index_x, index_y=index_y, index_z=index_z)

                    # Update local pheromone
                    self.__updateLocalPheromone(position=ant['position'], index_x=index_x, index_y=index_y, index_z=index_z)
            # Update global pheromone
            self.__updateGlobalPheromone()

            # Timing analysis
            end_time_millis = self.__current_time_milli() # Calculate iteration end time
            log.debug("END: "+str(end_time_millis))
            elapsed_time = end_time_millis - strt_time_millis
            elapsed_time_array.append(elapsed_time)
            log.debug("ELAPSED TIME: "+str(elapsed_time)) # Debug elapsed time

            # Do daemon actions
            self.__daemonActions(iter + 1)
        log.debug("FINISHED running ACS")

        # Save elapsed times to file
        np.savetxt(os.path.join(SUB_DIRECTORY_PLOT, "elapsed_time.txt"), elapsed_time_array, fmt='%d', newline='\n')
        
        cv2.waitKey(0)


    def __saveAntPheromone(self, pheromone, index_x, index_y, index_z):
        self.ants['pheromone'][index_x][index_y][index_z] += pheromone


    def __selectNextPixel(self, ant, index_x, index_y, index_z):
        # positions of allowed neighbors
        unvisited_neighbors = self.__unvisitedNeighbors(ant)

        # Random probability
        q = random.random()

        # Calculate numerator for each unvisited neighbor
        numerators = [
            pow(self.__pheromone(position=neighbor), self.alpha) * pow(self.__heuristic(position=neighbor), self.beta)
            for neighbor in unvisited_neighbors]

        # Apply ACS (pseudo random proportional rule)
        try:
            if (q <= self.q0):  # Exploration
                return unvisited_neighbors[np.argmax(numerators)]
            else:  # Exploitation (transition probability)
                denominator = sum(numerators)
                p_values = [float(num) / denominator for num in numerators]
                return unvisited_neighbors[np.argmax(p_values)]
        except ValueError:
            log.debug("Empty sequence to be handled")
            # Let the ant forget the previously visited positions
            # del self.ants['visited'][index_x][index_y][index_z][:]

            # return random position within ant boundary
            # return self.__randomPosition(*ant['boundary'])

            return self.__moveToNewPosition(index_x=index_x, index_y=index_y, index_z=index_z)


    def __unvisitedNeighbors(self, ant):
        curr_pos = ant['position']  # Ant current position
        boundary_strt_pos = ant['boundary'][0]  # Ant boundary start position
        boundary_stop_pos = ant['boundary'][1]  # Ant boundary end position

        x_index_min = -1
        x_index_max = 1

        y_index_min = -1
        y_index_max = 1

        # If current pixel is on boundary, modify index according to the boundary
        # (Restrict ants to move only within their boundaries)
        # if(curr_pos[0] <= boundary_strt_pos[0]):
        #     x_index_min = 0
        #
        # if(curr_pos[1] <= boundary_strt_pos[1]):
        #     y_index_min = 0
        #
        # if(curr_pos[0] >= boundary_stop_pos[0]):
        #     x_index_max = 0
        #
        # if(curr_pos[1] >= boundary_stop_pos[1]):
        #     y_index_max = 0

        # If current position not on the corner or edge of the image
        # (Let the ants cross their boundaries)
        if (curr_pos[0] <= self.boundary['min'][0]):
            x_index_min = 0

        if (curr_pos[1] <= self.boundary['min'][1]):
            y_index_min = 0

        if (curr_pos[0] >= self.boundary['max'][0]):
            x_index_max = 0

        if (curr_pos[1] >= self.boundary['max'][1]):
            y_index_max = 0

        # Indexes of allowed neighbors
        neighbor_index = [(i, j)
                          for i in xrange(x_index_min, x_index_max + 1)
                          for j in xrange(y_index_min, y_index_max + 1)
                          if not (i == 0 and j == 0)]

        # Add each index to the current position (gives allowed neighbors positions), and return
        return [tuple(map(sum, zip(curr_pos, index)))
                for index in neighbor_index
                if tuple(map(sum, zip(curr_pos, index))) not in ant['visited'][
                                                                -self.memory:]]  # If check to make sure that the allowed position not in ant's memory


    def __updateGlobalPheromone(self):
        visited_positions = set(
            [w for x in self.ants['visited'] for y in x for z in y for w in z])  # Gives set of all positions where pheromone is deposited on last iteration

        # TODO: Exponential increase (Modification expected)
        heuristic_values = [
            [
                [
                    [
                        self.__heuristic(position=position)
                        for position in ind_tour
                    ]
                    for ind_tour in all_visited_tours
                ]
                for all_visited_tours in ant['visited']
            ]
            for ant in self.ants
        ]

        delta_tau = [
            [
                [
                    np.average(heu_ant)
                    for heu_ant in ant
                ]
                for ant in n_ants
            ]
            for n_ants in heuristic_values
        ]

        for position in visited_positions:
            self.pheromone_matrix[position[0]][position[1]] = (1 - self.rho) * self.__pheromone(position=position)

            delta_tau_total = 0

            for (index_x, index_y, index_z), ant in np.ndenumerate(self.ants):
                if (position in ant['visited']):
                    delta_tau_ant = delta_tau[index_x][index_y][index_z]
                    delta_tau_total += delta_tau_ant

                    # Let the ant remember the amount of pheromone updated by the ant of index_x, index_y, index_z
                    self.__saveAntPheromone(pheromone=delta_tau_ant, index_x=index_x, index_y=index_y, index_z=index_z)

            self.pheromone_matrix[position[0]][position[1]] += self.rho * delta_tau_total

        # Positions that are not in any ant's memory (forgot positions)
        forgot_positions = set(self.all_visited_positions).difference(visited_positions)

        # Decrease the pheromone on forgot positions if it is more than tau init
        for position in forgot_positions:
            temp_pheromone_decay = (1 - self.rho) * self.__pheromone(position=position)
            if temp_pheromone_decay > self.tauinit:
                self.pheromone_matrix[position[0]][position[1]] = temp_pheromone_decay
            else:
                self.pheromone_matrix[position[0]][position[1]] = self.tauinit
                self.all_visited_positions.remove(tuple(position))

    def __updateLocalPheromone(self, position, index_x, index_y, index_z):
        self.pheromone_matrix[position[0]][position[1]] = pheromone = (1 - self.phi) * self.__pheromone(position=position) + \
                                                          self.phi * self.tauinit

        # Let the ant remember the amount of pheromone updated by the ant of index_x, index_y, index_z
        self.__saveAntPheromone(pheromone=pheromone, index_x=index_x, index_y=index_y, index_z=index_z)