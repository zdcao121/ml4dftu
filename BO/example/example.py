'''
Author: zdcao
Date: 2022-06-30 16:29:13
LastEditors: zdcao
LastEditTime: 2022-07-04 23:00:50
FilePath: \code\example.py
Description: 

Copyright (c) 2022 by zdcao/Xiang-Northlife, All Rights Reserved. 
'''

from core import *
import os
import argparse

# Command to run VASP executable.
AIMS_RUN_COMMAND = 'bash /home/users/zdcao/learning_hubbard_U/aimsRun.sh'
VASP_RUN_COMMAND = 'bash /home/users/zdcao/learning_hubbard_U/vaspRun_std.sh'
# Define the name for output file.
OUTFILENAME = 'run.out'
AIMS_SPECIES_DIR = '/home/users/zdcao/bin/fhi-aims/species_defaults/defaults_2020'
VASP_PP_PATH = '/home/users/zdcao/learning_hubbard_U/vasp_aims_code'
################################################



def parse_argument():
	"""
	kappa: The parameter to control exploration and exploitation.
		   exploitation 0 <-- kappa --> 10 exploration

	alpha1: Weight coefficient of band gap in the objective function.

	alpha2: Weight coefficient of delta band in the objective function.
	
	threshold: Convergence threshold of Bayesian optimization process.

	auto_kpoint: True for automatically generating the bandpath in the k space
				 False for generating the bandpath from the input.json data
				 default value is False
	"""
	parser = argparse.ArgumentParser(description='params')
	parser.add_argument('--which_u', dest='which_u',nargs='+', type=int, default=(1,1))
	parser.add_argument('--br', dest='br', nargs='+', type=int, default=(5,5))
	parser.add_argument('--kappa', dest='kappa', type=float, default=5)
	parser.add_argument('--alpha1', dest='alpha1', type=float, default=0.25)
	parser.add_argument('--alpha2', dest='alpha2', type=float, default=0.75)
	parser.add_argument('--threshold', dest='threshold', type=float, default=0.0001)
	parser.add_argument('--urange', dest='urange',nargs='+', type=int, default=(-10,10))
	parser.add_argument('--import_kpath', dest='import_kpath', type=bool, default=False)
	parser.add_argument('--auto_kpoint', dest='auto_kpoint', type=bool, default=False)
	parser.add_argument('--elements', dest='elements',nargs='+', type=str, default=('In', 'As'))
	parser.add_argument('--iteration', dest='iter', type=int, default=50)


	return parser.parse_args()


#===========================================================================================

def main():
	args = parse_argument()
	k = args.kappa
	a1 = args.alpha1
	a2 = args.alpha2
	which_u = tuple(args.which_u)
	urange = tuple(args.urange)
	br = tuple(args.br)
	import_kpath = args.import_kpath
	elements = args.elements
	iteration = args.iter
	auto_kpoint=args.auto_kpoint

	os.environ['AIMS_SPECIES_DIR'] = AIMS_SPECIES_DIR
	os.environ['VASP_PP_PATH'] = VASP_PP_PATH

	is_converged = calculate_aims(AIMS_RUN_COMMAND, OUTFILENAME, method='hse', auto_kpoint=auto_kpoint, import_kpath=import_kpath)		# AIMS HSE calculation
	
	header = []
	for i, u in enumerate(which_u):
		header.append('U_ele_%s' % str(i+1))
	
	if os.path.exists('./u_tmp.txt'):
		os.remove('./u_tmp.txt')
		
	with open('./u_tmp.txt', 'w+') as f:
		f.write('%s band_gap delta_band \n' % (' '.join(header)))

	obj = 0 
	threshold = args.threshold
	obj_list = []
	for i in range(iteration):
		calculate(command=VASP_RUN_COMMAND, outfilename=OUTFILENAME, method='dftu', auto_kpoint=auto_kpoint, import_kpath=import_kpath)
		db = delta_band(bandrange=br, path='./')
		db.deltaBand_aims()	   # aims 
			
		bayesianOpt = bayesOpt_DFTU(path='./', opt_u_index=which_u, u_range=urange, kappa=k, a1=a1, a2=a2, elements=elements)
		obj_next = bayesianOpt.bo()
		if abs(obj_next - obj) <= threshold:
			print("Optimization has been finished!")
			break
		obj = obj_next 

	bayesianOpt.plot() 
	print(bayesianOpt.optimal) 

	os.system('mv ./u_tmp.txt ./u_kappa_%s_a1_%s_a2_%s.txt' %(k, a1, a2))	  

if __name__ == "__main__":
	main()
	

