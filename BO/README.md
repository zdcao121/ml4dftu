# BayesianOpt4dftu (FHI-aims version) #

The VASP version of `BayesianOpt4dftu` is available at: https://github.com/maituoy/BayesianOpt4dftu.

**[FHI-aims](https://fhi-aims.org/)** is an all-electron electronic structure code based on numeric atom-centered orbitals. It enables first-principles simulations with very high numerical accuracy for production calculations, with excellent scalability up to very large system sizes (thousands of atoms) and up to very large, massively parallel supercomputers (ten thousand CPU cores).

In the FHI-aims version of `BayesianOpt4dftu`, We have adopted FHI-aims as a substitution for VASP in our hybrid functional calculations (HSE), attributing to the superior computational efficiency.

## Requirements ##

1. Python 3.6+
2. NumPy
3. Pandas
4. ASE (https://wiki.fysik.dtu.dk/ase/)
5. pymatgen (https://pymatgen.org/)
6. bayesian-optimization https://github.com/fmfn/BayesianOptimization
7. Vienna Ab initio Simulation Package (VASP) https://www.vasp.at/
8. Vaspvis (https://github.com/DerekDardzinski/vaspvis)
9. FHI-aims (https://fhi-aims.org/)
10. aimstools (https://github.com/romankempt/aimstools)


## Set up the input file (input.json) before running the code 

The meanings of the parameters in the `input.json` file can be found in the original repo: https://github.com/maituoy/BayesianOpt4dftu. Note that the current version of the code does not support directly modifying the computational parameters of FHI-aims from the input.json file.

## Usage
To run the code in your own cluster, you need to modify the environment settings in the `example.py` of the selected calculation based on the specs of your system and VASP binary. The file structure for implementing this algorithm is organized as follows:
```
|---dftu
|    |---band
|    |---scf
|---hse                    
|    |---bandXYYY.dat       
|---core.py
|---example.py              
|---input.json   
```

#### 1. Setting environments
  
  Set the running command for VASP executable
  
      VASP_RUN_COMMAND = 'srun -n 54 vasp_std'

  Set the running command for FHI-aims executable

    AIMS_RUN_COMMAND = 'srun --mpi=pmi2 aims.210716_2.scalapack.mpi.x'
      
  Define the VASP output file name.
      
      OUTFILENAME = 'vasp.out'
      
  Define the path direct to the VASP pesudoopotential. (P.S. It should be the directory containing the `potpaw_PBE` folder)
      
      VASP_PP_PATH = '/PATH/TO/THE/PESUDOPOTENTIAL/'
      
  Define the path direct to the FHI-aims numeric atom-centered orbitals. 

      AIMS_SPECIES_DIR = '/PATH/defaults_2020'


#### 2. Arguments options

  `--which_u` defines which element you would like to optimize the U for. For a unary substance, it has to be `(1,)`. For compounds with over 2 elements, you can set each element to 0 or 1 to switch off/on the optimization for that element. For InAs, when optimizing for both In and As, it will be `(1,1)`.
  
  `--br` defines band range you would like to include in your Δband. It is a tuple of two integers, which define the number of valence bands and conduction bands from the Fermi level.
  
  `--kappa` controls the exploration and exploitation when acquisition function sample the next points. Exploitation 0 <-- kappa --> 10 Exploration
  
  `--alpha1` and `alpha2` are the weight coefficients of Δgap and Δband.
  
  `--threshold` defines at what accuracy would you stop the BO process.
  
  `--urange` defines the U parameter range for optimization, currently it's not supported to define different U ranges for different elements.
  
  `--import_kpath` provides an external listing of high symmetry k-points in case some special k coordinates are not present in the ase library. The utilization of this parameter has been discontinued in the most recent version of the code. I recommend utilizing the automated generation of high symmetry k-points feature (`auto_kpoint`).
  
  `--elements` defines the elements in your system. It is set for plotting the BO results. If it's a unary substance, it has to be (ele,).
  
  `--iteration` defines the maximum steps that BO will be performed with.

  `--auto_kpoint` defines whether the automatic generation of high symmetry k-points feature is enabled or not. If true, the high symmetry k-points will be generated by HighSymmKpath module in the pymatgen library.

#### 3. Running the code

  After setting up all these stuff, you can simply run the calculation by
  
  `cd PATH/`
  
  `python example.py --arg1 XX --arg2 XX ...`
  
#### 4. Outputs

  Once the threshold or the maximum iterations is reached, you will get two output files
  
  `u_xx.txt` file consists of the U parameters, DFT+U band gap, and the Δband at each step.
  
  `1D_xxx.png` or `2D_xxx.png` plots showing you the Gaussian process predicted mean and the acquisition function.

  `u_temp_dict.txt` file consists of the U parameters, DFT+U band gap, HSE band gap, and the Δband at each step.
  
  Optimal U values will be output at the end of entire process based on the interpolation from the predicted mean space. You can also pick up the Us that give you largest objective value from the u.txt file.

## Citation
Please cite the following work if you use this code.

[1] Z. Cao, G. Cai, F. Xie, H. Jia, W. Liu, Y. Wang, F. Liu, X. Ren, S. Meng, and M. Liu, Predicting Structure-Dependent Hubbard U Parameters for Assessing Hybrid Functional-Level Exchange via Machine Learning, arXiv preprint at https://arxiv.org/abs/2302.09507 (2023).







