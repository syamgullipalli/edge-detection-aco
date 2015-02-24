###Directory Structure

EXP1, EXP2, EXP3 directories are three different experiments. Each experiment directory has a project named SI_Proj. PLOT directory contains the plots comparing the three experiments. PAPER directory contains the paper (paper.pdf) explaining the experiments and document source files.


###Installation requirements:

+ Python 2.7
+ NumPy module
+ OpenCV2 module

###How to run:

On terminatl

*go-to-the-path/EXP#/SI_Proj$* __python run_me.py --help__

__--help__ gives list of arguments taht could be passed.

###Command issued for experiments:

The results presented in the paper are produced by the following command (same for all three experiments)

__python run_me.py --debug --hor 1 --ver 1 --tauini 0.1 --iter 10 --cons 40  --ants 512 --q0 0.7  --alpha 1 --beta 1  --rho 0.1 --phi 0.05 --image ./standard_test_images/256/lena_color.tif__