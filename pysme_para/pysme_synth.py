import numpy as np
import pandas as pd

from pysme.sme import SME_Structure
from pysme.abund import Abund
from pysme.synthesize import synthesize_spectrum

from pqdm.processes import pqdm
from multiprocessing import cpu_count
from copy import deepcopy

import matplotlib.pyplot as plt

import warnings

from tqdm import tqdm

warnings.simplefilter("ignore")

def find_strong_lines(line_list, strong_line_element=['H', 'Mg', 'Ca', 'Na']):
    '''
    Find strong lines from the elements.
    '''
    
    strong_indices = line_list['wlcent'] < 0
    species = line_list._lines['species'].apply(lambda x: x.split(' ')[0])
    for ele in strong_line_element:
        strong_indices = strong_indices | (species.values == ele)
            
    return line_list[~strong_indices], line_list[strong_indices]

def batch_synth_simple(sme, line_list, strong_list=None, strong_line_element=['H', 'Mg', 'Ca', 'Na'], batch_mode='line', wav_range=2, N_line_chunk=2000, line_margin=2, strong_line_margin=100, parallel=False, n_jobs=5):
    '''
    The function to synthize the spectra using pysme in batch. This would work faster than doing the whole spectra at once.
    For the synthesize accuracy, the code will separate the strong lines from the line list if strong_line is None, and
    always include t

    batcvh_mode : str, defaule 'line'
        The mode for performing batch synthesis. If 'line', then the spectra will be divided according to the number of lines. If 'wave' then will be divided according to wavelength range. 
   
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

    if batch_mode == 'line':
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

        # Remove the wave ranges with no sme.wave inside 
        sub_wave_range = [ele for ele in sub_wave_range if len(sme.wave[0][(sme.wave[0] >= ele[0]) & (sme.wave[0] < ele[1])]) > 0]
        
        sub_line_list = []
        for (wav_start, wav_end) in sub_wave_range:
            line_list_sub = line_list[(line_list['wlcent'] >= wav_start-line_margin) & (line_list['wlcent'] <= wav_end+line_margin)]
            strong_lis_sub = strong_list[(strong_list['wlcent'] >= wav_start-strong_line_margin) & (strong_list['wlcent'] <= wav_end+strong_line_margin)]
            full_list_sub = deepcopy(strong_lis_sub)
            full_list_sub._lines = pd.concat([strong_lis_sub._lines, line_list_sub._lines]).sort_values('wlcent').reset_index(drop=True)
            sub_line_list += [full_list_sub]
        
        N_chunk = len(sub_wave_range)

        sub_sme = []
        for i in tqdm(range(N_chunk)):
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
