__author__ = 'syam'

import argparse

class ParseCommand:
    """Parse command line arguments"""

    def __init__(self):
        pass

    def parseOpt(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument("-i", "--image", metavar="IMAGE",
                        help="Path to an image to be processed", default="./standard_test_images/256/lena_color.tif", type=str)

        parser.add_argument("-a", "--alpha", metavar="ALPHA",
                        help="Pheromone coefficient", default=1, type=float)

        parser.add_argument("-b", "--beta", metavar="BETA",
                        help="Heuristic coefficient", default=2, type=float)

        parser.add_argument("-n", "--ants", metavar="NUM_ANTS",
                        help="Number of ants", default=8, type=int)

        parser.add_argument("-r", "--rho", metavar="RHO",
                        help="Pheromone evaporation coefficient (0, 1]", default=0.1, type=float)

        parser.add_argument("-p", "--phi", metavar="PHI",
                            help="Pheromone decay coefficient (0, 1]", default=0.05, type=float)

        parser.add_argument("-k", "--cons", metavar="NUM_CONSTRUCTIONS",
                        help="Number of construction steps", default=5, type=int)

        parser.add_argument("-l", "--iter", metavar="NUM_ITERATIONS",
                            help="Number of iterations", default=5, type=int)

        parser.add_argument("-q0", "--q0", metavar="Q0",
                            help="Degree of exploration (0, 1]", default=0.4, type=float)

        parser.add_argument("-t", "--tauini", metavar="TAU_INIT",
                            help="Initial value of pheromone matrix", default=0.01, type=float)

        parser.add_argument("-hor", "--hor", metavar="NUM_HOR_PAR",
                            help="Number of horizontal partitions", default=1, type=int)

        parser.add_argument("-ver", "--ver", metavar="NUM_VER_PAR",
                            help="Number of vertical partitions", default=1, type=int)

        parser.add_argument("-m", "--mem", metavar="MEMORY",
                            help="Ant's memory (Number of positions of last visited pixels)", default=8, type=int)

        parser.add_argument("-d", "--debug",
                            help="Print debug messages on console", action="store_true", default=False)

        args = parser.parse_args()

        return args
