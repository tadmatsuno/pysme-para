import numpy as np
import pandas as pd
import os, gc

from pysme.sme import SME_Structure
from pysme.abund import Abund
from pysme.synthesize import synthesize_spectrum
from pysme.smelib import _smelib

from pqdm.processes import pqdm
from multiprocessing import cpu_count
import pickle, time
from copy import copy, deepcopy

import matplotlib.pyplot as plt

import warnings

warnings.simplefilter("ignore")

# def generate_chunk_spectra(i, paras, nlte=False, strong=True, normalize_by_continuum=True, specific_intensities_only=False):

#     start_time = time.time()

#     sme = SME_Structure()
#     # sme.abund = Abund(pattern='asplund2009')
#     # Renew the solar abundance to Magg+2022
#     # for ele in solar_abundances.keys():
#     #     if solar_abundances[ele] != -99:
#     #         sme.abund[ele] = solar_abundances[ele]

#     sme.abund = copy(paras['abund'])
#     if paras['FeH'] != sme.abund.monh:
#         raise ValueError('FeH != abund.monh')

#     sme.teff, sme.logg, sme.monh, sme.vmic = paras['Teff'], paras['logg'], paras['FeH'], paras['Vturb']

#     sme.atmo = atmo
#     sme.atmo.method = 'embedded'

#     if sme.teff < 5000:
#         use_list = full_list_wo_strong[line_index[i][0]:line_index[i][1]]
#     else:
#         use_list = full_list_wo_strong_wo_TiO[line_index[i][0]:line_index[i][1]]

#     if strong:
#         # Include the strong lines, but only thosw with 1000AA of the target wavelength. 
#         indices = (strong_list['wlcent'] > wav_array[i]-200) & (strong_list['wlcent'] < wav_array[i+1]+200)
#         linelist_df = pd.concat([use_list._lines, strong_list[indices]._lines])
#         linelist_df = linelist_df.sort_values('wlcent')
#         use_list._lines = linelist_df
    
#     sme.linelist = use_list
    
#     ele_list = [ele.split(' ')[0] for ele in use_list._lines.groupby('species').size().index]
#     # print(ele_list)

#     nlte_grids = {'H': '/home/mingjie/researches/pySME/nlte_grids/nlte_H_ama51_pysme/nlte_H_ama51_pysme.grd',
#                   "Li": "/home/mingjie/researches/pySME/nlte_grids/nlte_Li_ama51_pysme/nlte_Li_ama51_pysme.grd",
#                   'C': '/home/mingjie/researches/pySME/nlte_grids/nlte_C_ama51_pysme/nlte_C_ama51_pysme.grd',
#                   'N': '/home/mingjie/researches/pySME/nlte_grids/nlte_N_ama51_pysme/nlte_N_ama51_pysme.grd',
#                   'O': '/home/mingjie/researches/pySME/nlte_grids/nlte_O_ama51_pysme/nlte_O_ama51_pysme.grd',
#                   'Na': '/home/mingjie/researches/pySME/nlte_grids/nlte_Na_ama51_pysme/nlte_Na_ama51_pysme.grd',
#                   'Mg': '/home/mingjie/researches/pySME/nlte_grids/nlte_Mg_ama51_pysme/nlte_Mg_ama51_pysme.grd',
#                   'Al': '/home/mingjie/researches/pySME/nlte_grids/nlte_Al_ama51_pysme/nlte_Al_ama51_pysme.grd',
#                   'Si': '/home/mingjie/researches/pySME/nlte_grids/nlte_Si_ama51_pysme/nlte_Si_ama51_pysme.grd',
#                   'K': '/home/mingjie/researches/pySME/nlte_grids/nlte_K_ama51_pysme/nlte_K_ama51_pysme.grd',
#                   'Ca': '/home/mingjie/researches/pySME/nlte_grids/nlte_Ca_ama51_pysme/nlte_Ca_ama51_pysme.grd',
#                   'Ti': '/home/mingjie/researches/pySME/nlte_grids/nlte_Ti_pysme.grd',
#                   'Mn': '/home/mingjie/researches/pySME/nlte_grids/nlte_Mn_ama51_pysme/nlte_Mn_ama51_pysme.grd',
#                   'Fe': '/home/mingjie/researches/pySME/nlte_grids/nlte_Fe_pysme/nlte_Fe_ama51_Feb2022_pysme.grd',
#                   'Ba': '/home/mingjie/researches/pySME/nlte_grids/nlte_Ba_ama51_pysme/nlte_Ba_ama51_pysme.grd'}

#     if nlte:
#         for ele in ['H', 'Li', 'C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ti', 'Mn', 'Fe', 'Ba']:
#             if ele in ele_list: 
#                 if ele in nlte_grids.keys():
#                     sme.nlte.set_nlte(ele, nlte_grids[ele])
#                 else:
#                     sme.nlte.set_nlte(ele)
    
#     sme.wave = wave[(wave >= wav_array[i]) & (wave < wav_array[i+1])]
    
#     sme.normalize_by_continuum = normalize_by_continuum
#     sme.specific_intensities_only = specific_intensities_only

#     try:
#         sme_res = synthesize_spectrum(sme)
#         end_time = time.time()
#         elapsed_time = end_time - start_time
#         # The synthesis is done without error
#         if specific_intensities_only:
#             wav = sme_res[0][0]
#             indices = (wav >= wav_array[i]) & (wav < wav_array[i+1])
#             wav = wav[indices]
#             sint = np.array([ele[indices] for ele in sme_res[1][0]])
#             cint = np.array([ele[indices] for ele in sme_res[2][0]])
#             return i, wav_array[i], wav_array[i+1], elapsed_time, wav, sint, cint
#         else:
#             return i, wav_array[i], wav_array[i+1], elapsed_time, sme_res.wave[0], sme_res.synth[0], sme_res.cont[0]
#     except:
#         # There is something wrong
#         if specific_intensities_only:
#             wav = sme.wave[0]
#             sint = np.full([sme.nmu, len(wav)], np.nan)
#             cint = np.full([sme.nmu, len(wav)], np.nan)
#             return i, wav_array[i], wav_array[i+1], 0, wav, sint, cint
#         else:
#             wav = sme.wave[0][indices]
#             sflu = np.full(len(wav), np.nan)
#             cflu = np.full(len(wav), np.nan)
#             return i, wav_array[i], wav_array[i+1], 0, wav, sflu, cflu

# def generate_whole_spectra(paras, save=None, nlte=False, strong=True, chunk=3500, margin=2, normalize_by_continuum=True, specific_intensities_only=False, overwrite=False):

#     # Skip if overwrite is False and the spectra already exists. 
#     if save != None:
#         if os.path.isfile(save) and not(overwrite):
#             print('Skip synthesizing spectra {}'.format(save))
#             return -99

#     global wave, wav_array, line_index, atmo

#     sme = SME_Structure()
#     sme.abund = Abund(pattern='asplund2009')
#     sme.teff, sme.logg, sme.monh, sme.vmic = paras['Teff'], paras['logg'], paras['FeH'], paras['Vturb']


#     if sme.logg < 3.25:
#         # Use spherical geometry
#         sme.atmo.geom = 'SPH'
#         sme.atmo.source = 'marcs2012s_t2.0.sav'
#     else:
#         sme.atmo.geom = 'PP'
#         sme.atmo.source = 'marcs2012p_t2.0.sav'

    
#     if sme.teff < 5000:
#         use_list = full_list_wo_strong[:20]
#         N_chunk = int(np.ceil(len(full_list_wo_strong) / chunk))
#     else:
#         use_list = full_list_wo_strong_wo_TiO[:20]
#         N_chunk = int(np.ceil(len(full_list_wo_strong_wo_TiO) / chunk))
    
#     sme.linelist = use_list
    
#     wave = np.arange(3700, 9500, 0.02)
#     sme.wave = wave

#     _ = synthesize_spectrum(sme)
#     atmo = sme.atmo

#     if sme.teff < 5000:
#         wav_array = np.load('../test/linelist/pkl/wav_array_full_list_wo_strong_chunk{}_margin{}.npy'.format(chunk, margin))
#         with open('../test/linelist/pkl/line_index_full_list_wo_strong_chunk{}_margin{}.pkl'.format(chunk, margin), 'rb') as file: 
#             line_index = pickle.load(file)
#     else:
#         wav_array = np.load('../test/linelist/pkl/wav_array_full_list_wo_strong_wo_TiO_chunk{}_margin{}.npy'.format(chunk, margin))
#         with open('../test/linelist/pkl/line_index_full_list_wo_strong_wo_TiO_chunk{}_margin{}.pkl'.format(chunk, margin), 'rb') as file: 
#             line_index = pickle.load(file)

#     args = [[i, paras, nlte, strong, normalize_by_continuum, specific_intensities_only] for i in range(0, N_chunk, 1)]

#     res = pqdm(args, generate_chunk_spectra, n_jobs=cpu_count()-5, argument_type='args')

#     res_out = {}
#     # Merge the chunked spectra
#     if specific_intensities_only:
#         wav = np.concatenate([ele[4] for ele in res])
#         sint = np.concatenate([ele[5] for ele in res], axis=1)
#         cint = np.concatenate([ele[6] for ele in res], axis=1)
#         # Resmaple to the specified wavelength scale to reduce the file size
#         if len(wav) > len(wave):
#             sint_resamp = np.array([np.interp(wave, wav, sint_single) for sint_single in sint])
#             cint_resamp = np.array([np.interp(wave, wav, cint_single) for cint_single in cint])    
#             wav = wave
#             sint = sint_resamp
#             cint = cint_resamp
#         res_out['merged_spectra'] = [wav, sint, cint]
#     else:
#         wav = np.concatenate([ele[4] for ele in res])
#         sflu = np.concatenate([ele[5] for ele in res])
#         cflu = np.concatenate([ele[6] for ele in res])
#         res_out['merged_spectra'] = [wav, sflu, cflu]

#     # res_out['chunk_spectra'] = res

#     if save != None:
#         with open(save, 'wb') as file:
#             pickle.dump(res_out, file)

# with open('../test/linelist/full_list_wo_strong_reset.pkl', 'rb') as file:
#     full_list_wo_strong = pickle.load(file)
# full_list_wo_strong_wo_TiO = full_list_wo_strong[full_list_wo_strong['species'] != 'TiO 1']
# full_list_wo_strong_wo_TiO._lines = full_list_wo_strong_wo_TiO._lines.reset_index(drop=True)
# with open('../test/linelist/strong_list_reset.pkl', 'rb') as file:
#     strong_list = pickle.load(file)

# element_number = pd.DataFrame([{
#     'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
#     'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
#     'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
#     'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
#     'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
#     'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
#     'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
#     'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
#     'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
#     'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
#     'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
#     'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
# }]).T.reset_index()
# element_number.columns = ['symbol', 'atomic_number']
# standard_para = pd.read_csv('benchmark_parameters_ver2.csv')

# nlte = True
# spec_type = 'int'
# # star_name = 'sun'
# for star_name in ['sun', 'arcturus', 'hd84937', 'hd122563', 'NGC6752-mg9', 'NGC1851-003']:
#     for nlte in [True, False]:
#         for spec_type in ['flu', 'int']:

#             teff, logg, Vturb, monh = standard_para.loc[:3, star_name]
#             print(star_name, teff, logg, Vturb, monh)
#             abund = Abund()
#             abund.monh = monh
#             plt.figure(figsize=(19, 3))
#             default_abuns = copy(abund.to_dict()['data'])
#             default_abuns[2:] += monh
#             plt.plot(np.arange(len(abund.pattern)), default_abuns, label='default abundances')
#             plt.xticks(np.arange(len(abund.pattern)), abund.pattern.keys(), rotation=0);

#             for i in standard_para.loc[5:, ['Unnamed: 0', star_name]].index:
#                 symbol = element_number.loc[element_number['atomic_number'] == int(standard_para.loc[i, 'Unnamed: 0']), 'symbol'].values[0]
#                 if not np.isnan(standard_para.loc[i, star_name]):
#                     abund[symbol] = standard_para.loc[i, star_name] - monh
                
#             target_abuns = copy(abund.to_dict()['data'])
#             target_abuns[2:] += monh
#             plt.plot(np.arange(len(abund.pattern)), target_abuns, label='target abundances')
#             plt.grid()
#             plt.legend()
#             plt.title(f'[M/H]={monh}')
#             plt.tight_layout()
#             plt.savefig(f'synth_spectra/{star_name}_abund.png', dpi=150)

#             if nlte:
#                 nlte_str = 'nlte'
#             else:
#                 nlte_str = 'lte'

#             if spec_type == 'flu':
#                 normalize_by_continuum=True
#                 specific_intensities_only=False
#             elif spec_type == 'int':
#                 normalize_by_continuum=False
#                 specific_intensities_only=True

#             save_name = f'synth_spectra/{star_name}_{nlte_str}_{spec_type}_pysme.pkl'
#             # try:
#             _ = generate_whole_spectra({'Teff':teff, 'logg':logg, 'FeH':monh, 'Vturb':Vturb, 'abund':abund}, save=save_name, nlte=nlte, normalize_by_continuum=normalize_by_continuum, specific_intensities_only=specific_intensities_only, overwrite=True)
#             # except:
#             #     print('Failed to generate {}'.format(save_name))



def find_strong_lines(line_list, strong_line_element=['H', 'Mg', 'Ca', 'Na']):
    '''
    Find strong lines from the elements.
    '''
    
    strong_indices = line_list['wlcent'] < 0
    species = line_list._lines['species'].apply(lambda x: x.split(' ')[0])
    for ele in strong_line_element:
        strong_indices = strong_indices | (species.values == ele)
            
    return line_list[~strong_indices], line_list[strong_indices]

def batch_synth_simple(sme, line_list, strong_list=None, strong_line_element=['H', 'Mg', 'Ca', 'Na'], N_line_chunk=2000, line_margin=2, strong_line_margin=100, parallel=False, n_jobs=5):
    '''
    The function to synthize the spectra using pysme in batch. This would work faster than doing the whole spectra at once.
    For the synthesize accuracy, the code will separate the strong lines from the line list if strong_line is None, and
    always include t
   
    Example for the pre-process:
    
    sme = SME_Structure()
    sme.teff, sme.logg, sme.monh, sme.vmic, sme.vmac, sme.vsini = 5026, 3.62, 0.2, 1.17, 0, 7.55
    sme.iptype = 'gauss'
    sme.ipres = 50000
    sme.wave = np.arange(wav_s, wav_end, 0.05)
    '''
    
    wav_s, wav_end = np.min(sme.wave), np.max(sme.wave)
    # Find the strong lines if not provided
    if strong_list is None:
        line_list, strong_list = find_strong_lines(line_list, strong_line_element=strong_line_element)
    
    # Crop the line_list to target wavelength
    indices = (line_list['wlcent'] >= wav_s - line_margin) & (line_list['wlcent'] <= wav_end + line_margin)
    line_list = line_list[indices]

    # Decide how many chunks to be divided
    N_chunk = int(np.ceil(len(line_list) / N_line_chunk))

    # Divide the line list to sub line lists
    sub_line_list = [line_list[N_line_chunk*i:N_line_chunk*(i+1)] for i in range(N_chunk)]
    sub_wave_range = [[line_list_single.wlcent[0], line_list_single.wlcent[-1]] for line_list_single in sub_line_list]
    sub_wave_range = [ele for ele in sub_wave_range if not((ele[1] <= wav_s) or (ele[0] >= wav_end))]
    for i in range(len(sub_wave_range)-1):
        sub_wave_range[i+1][0] = sub_wave_range[i][1]
    sub_wave_range[0][0] = sme.wave[0][0] - 1e-3
    sub_wave_range[-1][-1] = sme.wave[0][-1] + 1e-3
    
    sub_line_list = []
    for (wav_start, wav_end) in sub_wave_range:
        line_list_sub = line_list[(line_list['wlcent'] >= wav_start-line_margin) & (line_list['wlcent'] <= wav_end+line_margin)]
        strong_lis_sub = strong_list[(strong_list['wlcent'] >= wav_start-strong_line_margin) & (strong_list['wlcent'] <= wav_end+strong_line_margin)]
        full_list_sub = deepcopy(strong_lis_sub)
        full_list_sub._lines = pd.concat([strong_lis_sub._lines, line_list_sub._lines]).sort_values('wlcent').reset_index(drop=True)
        sub_line_list += [full_list_sub]
    
    N_chunk = len(sub_wave_range)

    sub_sme = []
    for i in range(N_chunk):
        sub_sme.append(deepcopy(sme))
        sub_sme[i].linelist = sub_line_list[i]
        sub_sme[i].wave = sme.wave[(sme.wave >= sub_wave_range[i][0]) & (sme.wave < sub_wave_range[i][1])]
        if not parallel:
            sub_sme[i] = synthesize_spectrum(sub_sme[i])

    if parallel:
        sub_sme = pqdm(sub_sme, synthesize_spectrum, n_jobs=n_jobs)

    # Merge the spectra
    wav, flux = np.concatenate([sub_sme[i].wave[0] for i in range(N_chunk)]), np.concatenate([sub_sme[i].synth[0] for i in range(N_chunk)])
    
    if np.all(wav != sme.wave[0]):
        raise ValueError
    
    return wav, flux, sub_line_list, sub_wave_range
