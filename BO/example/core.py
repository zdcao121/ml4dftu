import os
import json
from tkinter.tix import Tree
import bayes_opt
import subprocess
import numpy as np
import pandas as pd
import pymatgen as mg
import xml.etree.ElementTree as ET

from ase import Atoms, Atom
from ase.calculators.vasp.vasp import Vasp
from ase.dft.kpoints import *

from pymatgen.io.vasp.inputs import Incar, Kpoints, Potcar, Poscar
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure, Molecule
from pymatgen.io.vasp.outputs import BSVasprun, Vasprun
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from bayes_opt import UtilityFunction
from bayes_opt import BayesianOptimization
from string import ascii_lowercase

from BayesOpt4dftu.special_kpath import kpath_dict

from vaspvis import Band
from vaspvis.utils import BandGap
from vaspvis.utils import get_bandgap_old as get_bandgap

from matplotlib import pyplot as plt
from matplotlib import cm, gridspec

from ase.calculators.aims import Aims
from ase.io.aims import write_aims
from aimstools.postprocessing.output_reader import FHIAimsControlReader
from aimstools.postprocessing.output_reader import FHIAimsOutputReader


class vasp_init(object):
    def __init__(self, input_path):
        with open(input_path, 'r') as f:
            self.input_dict = json.load(f)
        self.struct_info = self.input_dict['structure_info']
        self.general_flags = self.input_dict['general_flags']
        self.atoms = None

    def init_atoms(self):
        lattice_param = self.struct_info['lattice_param']
        cell = np.array(self.struct_info['cell'])
        self.atoms = Atoms(cell=cell*lattice_param)
        for atom in self.struct_info['atoms']:
            if len(atom) == 3:  # 有初始磁矩
                if type(atom[2]) != list:
                    self.atoms.append(Atom(atom[0], atom[1], magmom=atom[2]))
                elif len(atom[2]) == 3:
                    self.atoms.append(Atom(atom[0], atom[1], magmom=atom[2][2]))
            else:  # 没有设置磁矩
                self.atoms.append(Atom(atom[0], atom[1]))

        return self.atoms

    def modify_poscar(self, path='./'):
        with open(path + '/POSCAR', 'r') as f:
            poscar = f.readlines()
            poscar[7] = 'Direct\n'
            f.close()

        with open(path + '/POSCAR', 'w') as d:
            d.writelines(poscar)
            d.close()
    
    def modify_incar(self, path='./'):     # TODO: 修改INCAR的磁矩格式。不知道为啥vasp 6.1.0版本不太对
        with open(path + '/INCAR', 'r') as f:
            incar = f.readlines()
            f.close()

        for i, line in enumerate(incar):
            if "MAGMOM" in line:
                line_begin = line.split()[0:2]
                line_ = ""
                for ii in line_begin:
                    line_ = line_ + ii + " "
                initial_magmom = self.atoms.get_initial_magnetic_moments()
                for j in range(len(initial_magmom)):
                    for k in range(len(initial_magmom[0])):
                        magmom = initial_magmom[j][k]
                        line_ = line_ + str(magmom) + " "
                line_ = line_ + "\n"
                incar[i] = line_
                break
        
        with open(path + '/INCAR', 'w') as d:
            d.writelines(incar)
            d.close()

    def kpt4pbeband(self, path, import_kpath):
        if import_kpath:
            special_kpoints = kpath_dict
        else:
            special_kpoints = get_special_points(self.atoms.cell)

        num_kpts = self.struct_info['num_kpts']
        labels = self.struct_info['kpath']
        kptset = list()
        lbs = list()
        if labels[0] in special_kpoints.keys():
            kptset.append(special_kpoints[labels[0]])
            lbs.append(labels[0])

        for i in range(1, len(labels)-1):
            if labels[i] in special_kpoints.keys():
                kptset.append(special_kpoints[labels[i]])
                lbs.append(labels[i])
                kptset.append(special_kpoints[labels[i]])
                lbs.append(labels[i])
        if labels[-1] in special_kpoints.keys():
            kptset.append(special_kpoints[labels[-1]])
            lbs.append(labels[-1])

        # Hardcoded for EuS and EuTe since one of the k-point is not in the special kpoints list.
        if 'EuS' in self.atoms.symbols or 'EuTe' in self.atoms.symbols:
            kptset[0] = np.array([0.5, 0.5, 1])

        kpt = Kpoints(comment='band', kpts=kptset, num_kpts=num_kpts,
                      style='Line_mode', coord_type="Reciprocal", labels=lbs)
        kpt.write_file(path+'/KPOINTS')

    def kpt4hseband(self, path, import_kpath):
        ibz = open(path+'/IBZKPT', 'r')
        num_kpts = self.struct_info['num_kpts']
        labels = self.struct_info['kpath']
        ibzlist = ibz.readlines()
        ibzlist[1] = str(num_kpts*(len(labels)-1) +
                         int(ibzlist[1].split('\n')[0])) + '\n'
        if import_kpath:
            special_kpoints = kpath_dict
        else:
            special_kpoints = get_special_points(self.atoms.cell)
        for i in range(len(labels)-1):
            k_head = special_kpoints[labels[i]]
            k_tail = special_kpoints[labels[i+1]]
            increment = (k_tail-k_head)/(num_kpts-1)
            ibzlist.append(' '.join(map(str, k_head)) +
                           ' 0 ' + labels[i] + '\n')
            for j in range(1, num_kpts-1):
                k_next = k_head + increment*j
                ibzlist.append(' '.join(map(str, k_next)) + ' 0\n')
            ibzlist.append(' '.join(map(str, k_tail)) +
                           ' 0 ' + labels[i+1] + '\n')
        with open(path+'/KPOINTS', 'w') as f:
            f.writelines(ibzlist)

    def generate_input(self, directory, step, xc, auto_kpoint=False, import_kpath=False): # 这里auto_kpoint暂时只支持dftu计算
        flags = {}
        flags.update(self.general_flags)
        flags.update(self.input_dict[step])
        if step == 'scf':
            if xc == 'pbe':
                flags.update(self.input_dict[xc])
            # calc = Vasp(self.atoms, directory=directory,
            #                 kpts=self.struct_info['kgrid_'+xc], gamma=True, **flags)
            calc = Vasp(self.atoms, directory=directory,
                             gamma=True, **flags)
            calc.write_input(self.atoms)
            if str(self.atoms.symbols) in ['Ni2O2']:
                mom_list = {'Ni': 2, 'Mn': 5, 'Co': 3, 'Fe': 4}
                s = str(self.atoms.symbols[0])
                incar_scf = Incar.from_file(directory+'/INCAR')
                incar_scf['MAGMOM'] = '%s -%s 0 0' % (mom_list[s], mom_list[s])
                incar_scf.write_file(directory+'/INCAR')

            self.modify_poscar(path=directory)
            # self.modify_incar(path=directory)   # for test
        elif step == 'band':
            flags.update(self.input_dict[xc])
            calc = Vasp(self.atoms, directory=directory, gamma=True, **flags)
            calc.write_input(self.atoms)
            self.modify_poscar(path=directory)
            # self.modify_incar(path=directory)    # for test
            if xc == 'pbe':
                if auto_kpoint:
                    self.auto_generate_kpts(path=directory)
                else:  
                    self.kpt4pbeband(directory, import_kpath)
            elif xc == 'hse':
                self.kpt4hseband(directory, import_kpath)

    def auto_generate_kpts(self, path="./"):

        poscar = Poscar.from_file(path + "/POSCAR")
        struct = poscar.structure

        spg_analyzer = SpacegroupAnalyzer(struct)
        primitive_standard_structure = spg_analyzer.get_primitive_standard_structure(international_monoclinic=False)


        kpath = HighSymmKpath(primitive_standard_structure)
        kpts = Kpoints.automatic_linemode(divisions=self.struct_info['num_kpts'], ibz=kpath)
        kpts.write_file(path + "/KPOINTS")

class delta_band(object):
    def __init__(self, bandrange=(5,5), path='./', iteration=1, interpolate=False):  #SOC为自旋轨道耦合
        self.path = path
        self.br_vb = bandrange[0]
        self.br_cb = bandrange[1]
        self.interpolate = interpolate
        self.vasprun_hse = os.path.join(path, 'hse/band/vasprun.xml')
        self.kpoints_hse = os.path.join(path, 'hse/band/KPOINTS')
        self.vasprun_dftu = os.path.join(path, 'dftu/band/vasprun.xml')
        self.kpoints_dftu = os.path.join(path, 'dftu/band/KPOINTS')
        self.iteration = iteration
        self.path_aims = os.path.join(path, 'hse/')       # aims 计算结果的路径
        self.ibands = self.get_ibands()  # 能带路径数量

    def get_ibands(self):
        kpoints = Kpoints.from_file(self.kpoints_dftu)
        kpoints_dict = kpoints.as_dict()
        ibands = len(kpoints_dict['labels'])/2
        ibands = int(ibands)
        return ibands 

    def readInfo(self, filepath):
        tree = ET.parse(filepath)
        root = tree.getroot()
        ispin = int(root.findall(
            './parameters/separator/.[@name="electronic"]/separator/.[@name="electronic spin"]/i/.[@name="ISPIN"]')[0].text)
        nbands = int(root.findall(
            './parameters/separator/.[@name="electronic"]/i/.[@name="NBANDS"]')[0].text)
        nkpts = len(root.findall('./kpoints/varray/.[@name="kpointlist"]/v'))

        return ispin, nbands, nkpts

    def access_eigen(self, b, interpolate=False):
        wave_vectors = b._get_k_distance()
        eigenvalues = b.eigenvalues

        if interpolate:
            _, eigenvalues_interp = b._get_interpolated_data(
                wave_vectors=wave_vectors,
                data=eigenvalues
            )

        if interpolate:
            return eigenvalues_interp
        else:
            return eigenvalues

    def locate_and_shift_bands(self, eigenvalues):
        band_mean = eigenvalues.mean(axis=1)

        below_index = np.where(band_mean < 0)[0]
        above_index = np.where(band_mean >= 0)[0]

        vbm = np.max(eigenvalues[below_index])
        cbm = np.min(eigenvalues[above_index])

        if cbm < vbm:
            vbm = 0.0
            cbm = 0.0

        valence_bands = eigenvalues[below_index[-self.br_vb:]]
        conduction_bands = eigenvalues[above_index[:self.br_cb]]

        valence_bands -= vbm
        conduction_bands -= cbm

        shifted_bands = np.r_[conduction_bands, valence_bands]

        return vbm, cbm, shifted_bands #####

    def deltaBand(self):
        ispin_hse, nbands_hse, nkpts_hse = self.readInfo(self.vasprun_hse)
        ispin_dftu, nbands_dftu, nkpts_dftu = self.readInfo(self.vasprun_dftu)

        
        if nbands_hse != nbands_dftu:
            raise Exception('The band number of HSE and GGA+U are not match!')

        kpoints = [line for line in open(self.kpoints_hse) if line.strip()]
        kpts_diff = 0
        for ii, line in enumerate(kpoints[3:]):
            if line.split()[3] != '0':
                kpts_diff += 1

        if nkpts_hse - kpts_diff != nkpts_dftu:
            raise Exception(
                'The kpoints number of HSE and GGA+U are not match!')

        new_n = 500

        if ispin_hse == 1 and ispin_dftu == 1:
            band_hse = Band(
                folder=os.path.join(self.path, 'hse/band'),
                spin='up',
                interpolate=self.interpolate,
                new_n=new_n,
                projected=False,
            )
            band_dftu = Band(
                folder=os.path.join(self.path, 'dftu/band'),
                spin='up',
                interpolate=self.interpolate,
                new_n=new_n,
                projected=False,
            )

            eigenvalues_hse = self.access_eigen(band_hse, interpolate=self.interpolate)
            eigenvalues_dftu = self.access_eigen(band_dftu, interpolate=self.interpolate)

            _, _, shifted_hse = self.locate_and_shift_bands(eigenvalues_hse)
            _, _, shifted_dftu = self.locate_and_shift_bands(eigenvalues_dftu)

            n = shifted_hse.shape[0] * shifted_hse.shape[1]
            delta_band = sum((1/n)*sum((shifted_hse - shifted_dftu)**2))**(1/2)

            bg = BandGap(folder=os.path.join(self.path, 'dftu/band'), method=1, spin='both',).bg

            incar = Incar.from_file('./dftu/band/INCAR')
            u = incar['LDAUU']
            u.append(bg)
            u.append(delta_band)
            output = ' '.join(str(x) for x in u)

            with open('u_tmp.txt', 'a') as f:
                f.write(output + '\n')
                f.close

            return delta_band

        elif ispin_hse == 2 and ispin_dftu == 2:
            band_hse_up = Band(
                folder=os.path.join(self.path, 'hse/band'),
                spin='up',
                interpolate=self.interpolate,
                new_n=new_n,
                projected=False,
            )

            band_dftu_up = Band(
                folder=os.path.join(self.path, 'dftu/band'),
                spin='up',
                interpolate=self.interpolate,
                new_n=new_n,
                projected=False
            )

            band_hse_down = Band(
                folder=os.path.join(self.path, 'hse/band'),
                spin='down',
                interpolate=self.interpolate,
                new_n=new_n,
                projected=False,
            )

            band_dftu_down = Band(
                folder=os.path.join(self.path, 'dftu/band'),
                spin='down',
                interpolate=self.interpolate,
                new_n=new_n,
                projected=False,
            )

            eigenvalues_hse_up = self.access_eigen(band_hse_up, interpolate=self.interpolate)
            eigenvalues_dftu_up = self.access_eigen(band_dftu_up, interpolate=self.interpolate)

            _, _, shifted_hse_up = self.locate_and_shift_bands(eigenvalues_hse_up)
            _, _, shifted_dftu_up = self.locate_and_shift_bands(eigenvalues_dftu_up)

            n_up = shifted_hse_up.shape[0] * shifted_hse_up.shape[1]
            delta_band_up = sum((1/n_up)*sum((shifted_hse_up - shifted_dftu_up)**2))**(1/2)

            eigenvalues_hse_down = self.access_eigen(band_hse_down, interpolate=self.interpolate)
            eigenvalues_dftu_down = self.access_eigen(band_dftu_down, interpolate=self.interpolate)

            _, _, shifted_hse_down = self.locate_and_shift_bands(eigenvalues_hse_down)
            _, _, shifted_dftu_down = self.locate_and_shift_bands(eigenvalues_dftu_down)

            n_down = shifted_hse_down.shape[0] * shifted_hse_down.shape[1]
            delta_band_down = sum((1/n_down)*sum((shifted_hse_down - shifted_dftu_down)**2))**(1/2)

            delta_band = np.mean([delta_band_up, delta_band_down])

            bg = BandGap(folder=os.path.join(self.path, 'dftu/band'), method=1, spin='both',).bg

            incar = Incar.from_file('./dftu/band/INCAR')
            u = incar['LDAUU']

            u.append(bg)
            u.append(delta_band)
            output = ' '.join(str(x) for x in u)

            with open('u_tmp.txt', 'a') as f:
                f.write(output + '\n')
                f.close

            return delta_band
        else:
            raise Exception('The spin number of HSE and GGA+U are not match!')
    
    # # get FHI-aims band data  TODO: aims的SOC能带输出文件读取需要改一下
    def get_aims_hse_data(self, path_aims, energy_offset, ispin, ibands):
        band_energies_full = []
        for spin in range(1, ispin + 1):
            band_energies_temp = []
            for iband in range(1, ibands + 1):
                # fname = "band%i%03i.out"%(spin, iband)
                if iband <= 9:
                    fname = f"{path_aims}/band{spin}00{iband}.out"
                else: 
                    fname = f"{path_aims}/band{spin}0{iband}.out"
                idx = []
                kvec = []
                band_energies = []
                band_occupations = []
                for line in open(fname):
                    words = line.split()
                    idx += [int(words[0])]
                    kvec += [list(map(float, words[1:4]))]
                    band_occupations += [list(map(float, words[4::2]))]
                    band_energies += [list(map(float, words[5::2]))]
                    # Apply energy offset if specified to all band energies just read in
                    band_energies[-1] = [x - energy_offset for x in band_energies[-1]]

                # # 能量已减去费米能级
                band_energies = np.asarray(band_energies)  # 同一行对应着同一k点位置上的能量分布，同一列对应着同一条能带数据
                # print(band_energies.shape)
                # print(band_energies)

                band_energies_trans = np.transpose(band_energies)  # 转秩为同一行代表同一条能带，同一列对应同一k点位置的能量
                # print(band_energies_trans)
                # print('*'*80)

                band_energies_temp.append(band_energies_trans)

            # 对矩阵按列进行拼接，每一行形成一条设定k path下完整的能带
            band_energies_tot = band_energies_temp[0]
            for i in range(1, ibands):
                band_energies_tot = np.c_[band_energies_tot, band_energies_temp[i]]
            # print(band_energies_tot)
            # print(band_energies_tot.shape)

            # 将按列拼接的矩阵依据自旋上下，添加到band_energies_full列表中
            band_energies_full.append(band_energies_tot)

        return band_energies_full
    
    def deltaBand_aims(self):   #TODO: 完善deltaBand的aims版本
        ispin_dftu, nbands_dftu, nkpts_dftu = self.readInfo(self.vasprun_dftu)

        # new_n (int): New number of k-points in between each high symmetry point.
        new_n = 500

        if ispin_dftu == 2:        # TODO：开启SOC后OUTCAR中ISPIN会变成1
            band_dftu_up = Band(
                folder=os.path.join(self.path, 'dftu/band'),
                spin='up',
                interpolate=self.interpolate,
                new_n=new_n,
                projected=False)
                # bandgap=True,
                # printbg=False)

            band_dftu_down = Band(
                folder=os.path.join(self.path, 'dftu/band'),
                spin='down',
                interpolate=self.interpolate,
                new_n=new_n,
                projected=False)

            band_energies_full = self.get_aims_hse_data(self.path_aims, energy_offset=0, ispin=2, ibands=self.ibands)

            # up
            eigenvalues_dftu_up = self.access_eigen(band_dftu_up, interpolate=self.interpolate)
            dftu_vbm_up, dftu_cbm_up, shifted_bands_dftu_up = self.locate_and_shift_bands(eigenvalues_dftu_up)

            aims_vbm_up, aims_cbm_up, shifted_bands_aims_up = self.locate_and_shift_bands(band_energies_full[0])

            n_up = shifted_bands_dftu_up.shape[0] * shifted_bands_dftu_up.shape[1]
            delta_band_up = sum((1/n_up)*sum((shifted_bands_aims_up - shifted_bands_dftu_up)**2))**(1/2)

            # down
            eigenvalues_dftu_down = self.access_eigen(band_dftu_down, interpolate=self.interpolate)
            dftu_vbm_down, dftu_cbm_down, shifted_bands_dftu_down = self.locate_and_shift_bands(eigenvalues_dftu_down)

            aims_vbm_down, aims_cbm_down, shifted_bands_aims_down = self.locate_and_shift_bands(band_energies_full[1])

            n_down = shifted_bands_dftu_down.shape[0] * shifted_bands_dftu_down.shape[1]
            delta_band_down = sum((1/n_down)*sum((shifted_bands_aims_down - shifted_bands_dftu_down)**2))**(1/2)

            delta_band = np.mean([delta_band_up, delta_band_down])

            aims_vbm = max(aims_vbm_up, aims_vbm_down)
            aims_cbm = min(aims_cbm_up, aims_cbm_down)

        elif ispin_dftu == 1:           # SOC 会自动ISPIN=1，好像可以不用额外添加SOC属性
            band_dftu = Band(
                folder=os.path.join(self.path, 'dftu/band'),
                spin='up',
                interpolate=self.interpolate,
                new_n=new_n,
                projected=False,
            )                #TODO: 需要测试ispin_dftu=1情况

            band_energies_full = self.get_aims_hse_data(self.path_aims, energy_offset=0, ispin=1, ibands=self.ibands)

            eigenvalues_dftu = self.access_eigen(band_dftu, interpolate=self.interpolate)
            dftu_vbm, dftu_cbm, shifted_bands_dftu = self.locate_and_shift_bands(eigenvalues_dftu)

            aims_vbm, aims_cbm, shifted_bands_aims = self.locate_and_shift_bands(band_energies_full[0])

            n = shifted_bands_dftu.shape[0] * shifted_bands_dftu.shape[1]
            delta_band = sum((1/n)*sum((shifted_bands_aims - shifted_bands_dftu)**2))**(1/2)

        else:
            raise Exception('The spin number of HSE and GGA+U are not match!')
        
        bg = get_bandgap(
            folder=os.path.join(self.path, 'dftu/band'),
            printbg=False,
            method=1,
            spin='both')

        incar = Incar.from_file(f'{self.path}dftu/band/INCAR')
        u = incar['LDAUU']     

        u.append(bg)
        u.append(delta_band)
        output = ' '.join(str(x) for x in u)

        with open(self.path+'u_tmp.txt', 'a') as f:
            f.write(output + '\n')
            f.close()

        global gap_aims
        gap_aims = aims_cbm - aims_vbm

        output_dict = {'U_value_0': u[0], 'U_value_1': u[1], 'bg_dftu': round(bg, 6),
                        'bg_hse': round(gap_aims, 6), 'delta_band': round(delta_band, 6)}

        with open(self.path + 'u_temp_dict.txt', 'a') as f_dict:
            f_dict.write(json.dumps(output_dict) + '\n')
            f_dict.close()

        return delta_band

class get_optimizer:
    def __init__(self, utxt_path, opt_u_index, u_range, gap_hse, a1, a2, kappa):
        data = pd.read_csv(utxt_path, header=0, delimiter="\s", engine='python')
        self.opt_u_index = opt_u_index
        self.u_range = u_range
        self.gap_hse = gap_hse
        self.a1 = a1
        self.a2 = a2
        self.kappa = kappa
        self.n_obs, _ = data.shape
        self.data = data
        self.utility_function = UtilityFunction(kind="ucb", kappa=kappa, xi=0)

    def loss(self, y, y_hat, delta_band, alpha_1, alpha_2):
        return -alpha_1 * (y - y_hat) ** 2 - alpha_2 * delta_band ** 2
    
    def set_bounds(self):
        # Set up the number of variables are going to be optimized.
        num_variables = int(sum(self.opt_u_index))
        variables_string = ['u_'+ str(i) for i, o in enumerate(self.opt_u_index) if o]

        # Set up the U ranges for each variable.
        pbounds = {}
        for variable in variables_string:
            pbounds[variable] = self.u_range
        return pbounds
    
    def optimizer(self):   
        pbounds = self.set_bounds()  
        optimizer = BayesianOptimization(
            f=None,
            pbounds=pbounds,
            verbose=2,
            random_state=1,
        )

        v_strings = list(pbounds.keys())
        opt_index = [int(v.split('_')[1]) for v in v_strings]

        U_values_0 =[]
        U_values_1 = []

        for i in range(self.n_obs):
            values = list()
            for j in range(len(opt_index)):
                #values.append(self.data.iloc[i][j])
                if len(opt_index) == 1:
                    if self.data.iloc[i][0] in U_values_0:		# 1D 情况
                        # print("hello world")
                        values.append(self.data.iloc[i][j]+0.001)
                    else:
                        values.append(self.data.iloc[i][j])
                elif len(opt_index) == 2:						 # TODO: 2D重复值情况需要继续开发
                    if self.data.iloc[i][0] in U_values_0:		
                        # print("hello world")
                        values.append(self.data.iloc[i][j]+0.001)
                    else:
                        values.append(self.data.iloc[i][j])

            params = {}
            for (value, variable) in zip(values, v_strings):
                params[variable] = value
            target = self.loss(self.gap_hse, 
                               self.data.iloc[i].band_gap, 
                               self.data.iloc[i].delta_band, 
                               self.a1, 
                               self.a2)

            optimizer.register(
                params=params,
                target=target,
            )
            U_values_0.append(self.data.iloc[i][0])
            U_values_1.append(self.data.iloc[i][1])
        return optimizer, target
        

class plot_bo(get_optimizer):
    def __init__(self, utxt_path, opt_u_index, u_range, gap_hse, a1, a2, kappa, elements):
        super().__init__(utxt_path, opt_u_index, u_range, gap_hse, a1, a2, kappa)
        optimizer, target = self.optimizer()
        self.optimizer = optimizer
        self.target = target
        self.elements = elements
        self.optimal = 0
    
    def get_optimal(self, x, mu):
        best_obj = mu.max()
        best_index = np.where(mu == mu.max())[0][0]
        best_u = x[best_index]
        optimal = (best_u, best_obj)
        return optimal
        
    def predict(self, ratio=1):
        u = list(self.optimizer.res[0]["params"].keys())
        dim = len(u)
        plot_size = len(self.optimizer.res)*ratio
        if dim == 1:
            x = np.linspace(self.u_range[0], self.u_range[1], 10000).reshape(-1, 1)
            x_obs = np.array([res["params"][u[0]] for res in self.optimizer.res]).reshape(-1,1)[:plot_size]
            y_obs = np.array([res["target"] for res in self.optimizer.res])[:plot_size]
            
            self.optimizer._gp.fit(x_obs, y_obs)
            mu, sigma = self.optimizer._gp.predict(x, return_std=True)
            self.optimal = self.get_optimal(x, mu)

            data4plot = {'mu': mu,
                         'sigma': sigma,
                         'x': x,
                         'x_obs': x_obs,
                         'y_obs': y_obs}

            return data4plot
        
        if dim == 2:
            x = y = np.linspace(self.u_range[0], self.u_range[1], 300)
            X, Y = np.meshgrid(x, y)
            x = X.ravel()
            y = Y.ravel()
            X = np.vstack([x, y]).T

            x1_obs = np.array([[res["params"][u[0]]] for res in self.optimizer.res])[:plot_size]
            x2_obs = np.array([[res["params"][u[1]]] for res in self.optimizer.res])[:plot_size]
            y_obs = np.array([res["target"] for res in self.optimizer.res])[:plot_size]
            obs = np.column_stack((x1_obs, x2_obs))

            self.optimizer._gp.fit(obs, y_obs)
            mu, sigma = self.optimizer._gp.predict(X, eval)
            self.optimal = self.get_optimal(X, mu)

            data4plot = {'mu': mu,
                         'sigma': sigma,
                         'obs': obs,
                         'x1_obs': x1_obs,
                         'x2_obs': x2_obs,
                         'x': x,
                         'y': y,
                         'X': X}

            return data4plot
        
        if dim == 3:
            x = y = z = np.linspace(self.u_range[0], self.u_range[1], 100)
            X, Y, Z= np.meshgrid(x, y, z)
            x = X.ravel()
            y = Y.ravel()
            z = Z.ravel()
            X = np.vstack([x, y, z]).T

            x1_obs = np.array([[res["params"][u[0]]] for res in self.optimizer.res])[:plot_size]
            x2_obs = np.array([[res["params"][u[1]]] for res in self.optimizer.res])[:plot_size]
            x3_obs = np.array([[res["params"][u[2]]] for res in self.optimizer.res])[:plot_size]
            y_obs = np.array([res["target"] for res in self.optimizer.res])[:plot_size]
            obs = np.column_stack((x1_obs, x2_obs, x3_obs))

            self.optimizer._gp.fit(obs, y_obs)
            mu, sigma = self.optimizer._gp.predict(X, eval)
            self.optimal = self.get_optimal(X, mu)

            return mu, sigma

    def plot(self, ratio=1):
        u = list(self.optimizer.res[0]["params"].keys())
        dim = len(u)
        plot_size = len(self.optimizer.res)*ratio
        opt_eles = [ele for i, ele in enumerate(self.elements) if self.opt_u_index[i]]

        if dim == 1:
            d = self.predict()
            fig = plt.figure()
            gs = gridspec.GridSpec(2, 1) 
            axis = plt.subplot(gs[0])
            acq = plt.subplot(gs[1])
            axis.plot(d['x_obs'].flatten(), d['y_obs'], 'D', markersize=8, label=u'Observations', color='r')
            axis.plot(d['x'], d['mu'], '--', color='k', label='Prediction')
            axis.fill(np.concatenate([d['x'], d['x'][::-1]]), 
                    np.concatenate([d['mu'] - 1.9600 * d['sigma'], (d['mu'] + 1.9600 * d['sigma'])[::-1]]),
                    alpha=.6, fc='c', ec='None', label='95% confidence interval')
                
            axis.set_xlim(self.u_range)
            axis.set_ylim((None, None))
            axis.set_ylabel('f(x)')

            utility = self.utility_function.utility(d['x'], self.optimizer._gp, 0)
            acq.plot(d['x'], utility, label='Acquisition Function', color='purple')
            acq.plot(d['x'][np.argmax(utility)], np.max(utility), '*', markersize=15, 
                    label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
            acq.set_xlim(self.u_range)
            acq.set_ylim((np.min(utility)-0.5,np.max(utility)+0.5))
            acq.set_ylabel('Acquisition')
            acq.set_xlabel('U (eV)')
            axis.legend(loc=4, borderaxespad=0.)
            acq.legend(loc=4, borderaxespad=0.)

            plt.savefig('1D_kappa_%s_a1_%s_a2_%s.png' %(self.kappa, self.a1, self.a2), dpi = 400)

        if dim == 2:
            
            d = self.predict()
            fig, axis = plt.subplots(1, 2, figsize=(15,5))
            plt.subplots_adjust(wspace = 0.2)
            
            axis[0].plot(d['x1_obs'], d['x2_obs'], 'D', markersize=4, color='k', label='Observations')
            axis[0].set_title('Gaussian Process Predicted Mean',pad=10)
            im1 = axis[0].hexbin(d['x'], d['y'], C=d['mu'], cmap=cm.jet, bins=None)
            axis[0].axis([d['x'].min(), d['x'].max(), d['y'].min(), d['y'].max()])
            axis[0].set_xlabel(r'U_%s (eV)' %opt_eles[0],labelpad=5)
            axis[0].set_ylabel(r'U_%s (eV)' %opt_eles[1],labelpad=10,va='center')
            cbar1 = plt.colorbar(im1, ax = axis[0])

            utility = self.utility_function.utility(d['X'], self.optimizer._gp, self.optimizer.max)
            axis[1].plot(d['x1_obs'], d['x2_obs'], 'D', markersize=4, color='k', label='Observations')
            axis[1].set_title('Acquisition Function',pad=10)
            axis[1].set_xlabel(r'U_%s (eV)' %opt_eles[0],labelpad=5)
            axis[1].set_ylabel(r'U_%s (eV)' %opt_eles[1],labelpad=10,va='center')
            im2 = axis[1].hexbin(d['x'], d['y'], C=utility, cmap=cm.jet, bins=None)
            axis[1].axis([d['x'].min(), d['x'].max(), d['y'].min(), d['y'].max()])
            cbar2 = plt.colorbar(im2, ax = axis[1])

            plt.savefig('2D_kappa_%s_a1_%s_a2_%s.png' %(self.kappa, self.a1, self.a2), dpi = 400)

class bayesOpt_DFTU(plot_bo):
    def __init__(self, 
                 path, 
                 opt_u_index=(1, 1, 0), 
                 u_range=(0, 10), 
                 a1=0.25, 
                 a2=0.75, 
                 kappa=2.5,
                 elements=['ele1','ele2','ele3'],
                 plot=False):
        #gap_hse = BandGap(folder=os.path.join(path, 'hse/band'), method=1, spin='both',).bg   # TODO: 这里要改成aims的gap
        gap_hse = gap_aims 
        if plot:
            upath = "./u_kappa_%s_a1_%s_a2_%s.txt" %(kappa, a1, a2)
        if not plot:
            upath = './u_tmp.txt'
        plot_bo.__init__(self, upath, opt_u_index, u_range, gap_hse, a1, a2, kappa, elements)


    def bo(self):
        next_point_to_probe = self.optimizer.suggest(self.utility_function)

        points = list(next_point_to_probe.values())
        points = [round(elem, 6) for elem in points]

        U = [str(x) for x in points]
        with open('input.json', 'r') as f:
            data = json.load(f)
            elements = list(data["pbe"]["ldau_luj"].keys())
            for i in range(len(self.opt_u_index)):
                if self.opt_u_index[i]:
                    try:
                        data["pbe"]["ldau_luj"][self.elements[i]
                                                ]["U"] = round(float(U[i]), 6)
                    except:
                        data["pbe"]["ldau_luj"][self.elements[i]
                                                ]["U"] = round(float(U[i-1]), 6)
            f.close()

        with open('input.json', 'w') as f:
            json.dump(data, f, indent=4)
            f.close()
            
        return self.target
    


def calculate(command, outfilename, method, auto_kpoint=False, import_kpath=False):
    olddir = os.getcwd()
    calc = vasp_init(olddir+'/input.json')
    calc.init_atoms()

    if method == 'dftu':
        calc.generate_input(olddir+'/%s/scf' %
                            method, 'scf', 'pbe', auto_kpoint=auto_kpoint, import_kpath=import_kpath)
        calc.generate_input(olddir+'/%s/band' %
                            method, 'band', 'pbe', auto_kpoint=auto_kpoint, import_kpath=import_kpath)

    if os.path.isfile(f'{olddir}/{method}/band/eigenvalues.npy'):
        os.remove(f'{olddir}/{method}/band/eigenvalues.npy')

    elif method == 'hse':
        calc.generate_input(olddir+'/%s/scf' %
                            method, 'scf', 'hse', auto_kpoint=auto_kpoint, import_kpath=import_kpath)
        if not os.path.exists(olddir+'/%s/band' % method):
            os.mkdir(olddir+'/%s/band' % method)

    try:
        os.chdir(olddir+'/%s/scf' % method)
        errorcode_scf = subprocess.call(
            '%s > %s' % (command, outfilename), shell=True)
        os.system('cp CHG* WAVECAR IBZKPT %s/%s/band' % (olddir, method))
        if method == 'hse':
            calc.generate_input(olddir+'/%s/band' %
                                method, 'band', 'hse', auto_kpoint=auto_kpoint, import_kpath=import_kpath)
    finally:
        os.chdir(olddir+'/%s/band' % method)
        errorcode_band = subprocess.call(
            '%s > %s' % (command, outfilename), shell=True)
        os.chdir(olddir)


class aims_init(object):
    def __init__(self, input_path):
        with open(input_path, 'r') as f:
            self.input_dict = json.load(f)
        self.struct_info = self.input_dict['structure_info']
        self.general_flags = self.input_dict['general_flags']
        self.plus_u = None
        self.atoms = None

    def init_atoms(self):
        lattice_param = self.struct_info['lattice_param']
        cell = np.array(self.struct_info['cell'])
        self.atoms = Atoms(cell=cell * lattice_param)
        for atom in self.struct_info['atoms']:
            if len(atom) == 3:  # 有初始磁矩
                if type(atom[2]) != list:
                    self.atoms.append(Atom(atom[0], atom[1], magmom=atom[2]))
                elif len(atom[2]) == 3:
                    self.atoms.append(Atom(atom[0], atom[1], magmom=atom[2][2]))
            else:  # 没有设置磁矩
                self.atoms.append(Atom(atom[0], atom[1]))
        self.atoms.set_pbc(True)

        return self.atoms

    def init_plus_u(self):
        plus_u_origin = self.input_dict['plus_u']
        plus_u = {}
        for element in plus_u_origin.keys():
            u_value = ''
            for para in plus_u_origin[element].keys():
                u_value += str(plus_u_origin[element][para]) + ' '
            plus_u.update({element: u_value})
        self.plus_u = plus_u

    def modify_geomerty(self, path='./'):
        with open(path + '/geometry.in', 'r') as f:
            lines = f.readlines()
            f.close()
        for i in range(len(lines)):
            if "atom" in lines[i]:
                lines[i] = lines[i].replace("atom", "atom_frac")

        with open(path + '/geometry.in', 'w') as d:
            d.writelines(lines)
            d.close()

    def generate_input(self, directory='.', xc='pbe', auto_kpoint=False, import_kpath=False):
        geometry_in = os.path.join(directory, 'geometry.in')
        write_aims(geometry_in, atoms=self.atoms)  # 用这个来输出aims几何文件
        self.modify_geomerty(path=directory)  # 修改一下几何文件

        species_dir = os.getenv("AIMS_SPECIES_DIR")
        flags = {}
        #=========================================== 针对aims的input.json文件
        flags_vasp = {}
        flags_vasp .update(self.general_flags)
        flags_vasp.update(self.input_dict[xc])
        #===========================================
        flags.update({"exx_band_structure_version": 1})
        flags.update({"relativistic": "atomic_zora scalar"})
        flags.update({"k_grid_density": 5})
        
        if "ispin" in flags_vasp.keys() and flags_vasp["ispin"] == 2:
            flags.update({"spin": "collinear"})
        else:
            flags.update({"spin": "none"})
        #flags.update({"spin": "collinear"})        #for test

        if "lsorbit" in flags_vasp.keys() and flags_vasp["lsorbit"]==True:
            flags.update({"include_spin_orbit": " "})
        # ====================================================================#TODO: 测试针对不同原子序数的原子采用不同的tier
        # atom_num_ = self.atoms.numbers.tolist()
        # atoms_num = list(set(atom_num_))
        # atoms_num.sort(key=atom_num_.index)

        # tier = []
        # for num in atoms_num:
        #	  if num <= 10:
        #		  tier.append(2)
        #	  else:
        #		  tier.append(1)
        # =====================================================================
        # flags.update({"tier": tier})  # 控制原子轨道基组的大小

        flags.update({"tier": 1})  # 控制原子轨道基组的大小
        if xc == 'pbe':
            flags.update({"xc": "pbe"})

            species_dir = os.path.join(species_dir, "tight")
            flags.update({'species_dir': species_dir})
            calc_aims = Aims(plus_u=self.plus_u, **flags)
        elif xc == 'hse':
            flags.update({"xc": "hse06 0.11"})
            flags.update({"hse_unit": "bohr"})
            flags.update({"hybrid_xc_coeff": 0.25})

            species_dir = os.path.join(species_dir, "intermediate")
            flags.update({'species_dir': species_dir})
            calc_aims = Aims(**flags)
        else:
            print("it is not support now! please check the xc functional")

        control_in = os.path.join(directory, 'control.in')
        calc_aims.write_control(filename=control_in, atoms=self.atoms, debug=False)
        calc_aims.write_species(filename=control_in, atoms=self.atoms)

        npoints = self.struct_info['num_kpts']
        self.kpt4band(directory=directory, npoints=npoints, auto_kpoint=auto_kpoint, import_kpath=import_kpath)

    def kpt4band(self, directory='.', npoints=21, auto_kpoint=False, import_kpath=False):  # 产生能带路径
        if auto_kpoint:
            bandpath = self.get_bandpath(npoints)  # 自动产生能带路径
        else:
            bandpath = self.kpt4pbeband(import_kpath=import_kpath)  # 手动产生能带路径(input.json)
        control_in = os.path.join(directory, 'control.in')
        f = open(control_in, "r")
        lines = f.readlines()
        f.close()
        for i, line in enumerate(lines):
            if 'exx_band_structure_version' in line:
                ii = i
                break
        for i, path in enumerate(bandpath):
            if path[0] != '#':
                path = "output band " + path + '\n'
            lines.insert(ii + i, path)
        f = open(control_in, "w+")
        for line in lines:
            f.write(line)
        f.close()

    def modify_poscar(self, path='./'):
        with open(path + '/POSCAR', 'r') as f:
            poscar = f.readlines()
            poscar[7] = 'Direct\n'
            f.close()

        with open(path + '/POSCAR', 'w') as d:
            d.writelines(poscar)
            d.close()

    def get_bandpath(self, npoints=21):  # npoints 为插点数量

        self.atoms.write("./hse/POSCAR")  # TODO: 以后改成相对路径（如果有时间的话）
        self.modify_poscar(path="./hse")
        poscar = Poscar.from_file("./hse/POSCAR")
        struct = poscar.structure

        spg_analyzer = SpacegroupAnalyzer(struct)
        primitive_standard_structure = spg_analyzer.get_primitive_standard_structure(international_monoclinic=False)

        kpath = HighSymmKpath(primitive_standard_structure)
        kpts = Kpoints.automatic_linemode(divisions=npoints, ibz=kpath)

        kpts_dict = kpts.as_dict()
        labels = kpts_dict['labels']
        kptset = kpts_dict['kpoints']
        #=====================================================
        output_bands = []
        i = 0
        while i < len(labels):
            vec1 = ' '.join(str(ii) for ii in kptset[i])
            vec2 = ' '.join(str(ii) for ii in kptset[i+1])
            output_bands.append(
                "{vec1} \t {vec2} \t {npoints} \t {label1} {label2}".format(
                    label1=labels[i],
                    label2=labels[i+1],
                    npoints=npoints,
                    vec1=vec1,
                    vec2=vec2,
                )
            )
            i = i + 2
        return output_bands

    def kpt4pbeband(self, import_kpath=False):  # 读取input.json中的能带路径 (手动产生能带路径)
        if import_kpath:
            special_kpoints = kpath_dict
        else:
            special_kpoints = get_special_points(self.atoms.cell)
        num_kpts = self.struct_info['num_kpts']
        labels = self.struct_info['kpath']
        kptset = list()
        lbs = list()
        if labels[0] in special_kpoints.keys():
            kptset.append(special_kpoints[labels[0]])
            lbs.append(labels[0])

        for i in range(1, len(labels) - 1):
            if labels[i] in special_kpoints.keys():
                kptset.append(special_kpoints[labels[i]])
                lbs.append(labels[i])
                kptset.append(special_kpoints[labels[i]])
                lbs.append(labels[i])
        if labels[-1] in special_kpoints.keys():
            kptset.append(special_kpoints[labels[-1]])
            lbs.append(labels[-1])

        # Hardcoded for EuS and EuTe since one of the k-point is not in the special kpoints list.
        if 'EuS' in self.atoms.symbols or 'EuTe' in self.atoms.symbols:
            kptset[0] = np.array([0.5, 0.5, 1])

        output_bands = []
        for i in range(len(lbs)):
            output_band = " "
            if i % 2 == 0:
                for j in range(len(kptset[i])):
                    output_band += str(kptset[i][j]) + " "
                for j in range(len(kptset[i + 1])):
                    output_band += str(kptset[i + 1][j]) + " "
                output_band += str(num_kpts) + " "
                output_band += lbs[i] + " " + lbs[i + 1]
                output_bands.append(output_band)

        return output_bands

def calculate_aims(command, outfilename, method='pbe', auto_kpoint=False, import_kpath=False):
    olddir = os.getcwd()
    calc = aims_init(olddir + '/input.json')
    calc.init_atoms()
    if method == "pbe":  #TODO: 这里需要完善一下，PBE的aims计算还没有开发
        calc.init_plus_u()

    if not os.path.exists(olddir + '/%s' % method):
        os.mkdir(olddir + '/%s' % method)

    if method == 'dftu':
        calc.generate_input(olddir + '/%s/' %
                            method, xc='pbe', auto_kpoint=auto_kpoint, import_kpath=import_kpath)
    elif method == 'hse':
        calc.generate_input(olddir + '/%s/' %
                            method, xc='hse', auto_kpoint=auto_kpoint, import_kpath=import_kpath)

    os.chdir(olddir + '/%s' % method)
    errorcode = subprocess.call(
        '%s > %s' % (command, outfilename), shell=True)

    # aims_output = FHIAimsOutputReader("./"+outfilename)
    #----------------------------------------------------#TODO: 这里要重构
    converged = False
    with open('./'+outfilename, 'r') as file:
        for l in file.readlines()[::-1]:
            if "Have a nice day." in l:
                converged = True
                break
    #----------------------------------------------------
    os.chdir(olddir) 

    return converged
