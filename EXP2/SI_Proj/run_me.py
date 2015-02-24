__author__ = 'syam'

import logging

from parser.CommandParser import ParseCommand
from parser.ImageParser import ParseImage
from aco.AntColonySystem import PartitionBasedAcs

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

class EdgeDetector(object):
    """Edge detector using Ant Colony System
    """

    def __init__(self):
        pass

    def run(self):
        # Parse command line options
        cmdParse = ParseCommand()
        args = cmdParse.parseOpt()

        # Set logging if DEBUG parameter is passed
        log.propagate = args.debug

        # Parse image
        imgParseParams = {      # Image parsing parameters
            'file_path' : args.image,
            'debug'     : args.debug
        }
        imgParse = ParseImage(**imgParseParams)
        heuristic_matrix = imgParse.parseIntensity()

        # Set parameters
        log.debug("Setting parameters for ACS")
        params = {
            'nants'   : args.ants,    # Number of ants (1)
            'tauinit' : args.tauini,  # Initial pheromone value (2)
            'alpha'   : args.alpha,   # Pheromone coefficient (3)
            'beta'    : args.beta,    # Heuristic coefficient (4)
            'rho'     : args.rho,     # Pheromone evaporation coefficient (5)
            'phi'     : args.phi,     # Pheromone decay coefficient (6)
            'q0'      : args.q0,      # Degree of exploration (7)
            'iter'    : args.iter,    # Number of iterations (8)
            'cons'    : args.cons,    # Number of constructions (9)
            'hor'     : args.hor,     # Number of horizontal partitions (10)
            'ver'     : args.ver,     # Number of vertical partitions (11)
            'heu'     : heuristic_matrix, # Heuristic information (12)
            'mem'     : args.mem,     # Ant's memory - Number of positions of last visited pixels (13)
            'debug'   : args.debug    # Print debug messages
        }

        #Run ACS Algorithm
        acs = PartitionBasedAcs(**params)
        acs.run()



if(__name__=="__main__"):
    edge = EdgeDetector()
    edge.run()