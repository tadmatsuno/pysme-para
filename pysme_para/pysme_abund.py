import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pqdm

from tqdm.notebook import tqdm

from pysme.sme import SME_Structure
# from pysme.abund import Abund
# from pysme.linelist.vald import ValdFile
from pysme.synthesize import synthesize_spectrum
from pysme.solve import solve
from pysme.smelib import _smelib

from pqdm.processes import pqdm

from copy import copy

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


def sensitive_spectra_single(wav_start, wav_end, teff, logg, m_h, vmic, vmac, vsini, abund, R, ele_ion, use_list, 
                             wave_synth_array=None, error_array=None, abun_shift=0.1, margin=1):
    '''
    Calculate the sensitivite spectra as well as the exptected error of abudnance for a given flux 
    error (if None then will be assume as 0.01).
    '''
    # Sanity check
    if abund.monh != m_h:
        raise ValueError('abund.monh != m_h')

    sme = SME_Structure()

    sme.teff, sme.logg, sme.monh, sme.vmic, sme.vmac, sme.vsini = teff, logg, m_h, vmic, vmac, vsini
    sme.iptype = 'gauss'
    sme.ipres = R
    sme.abund = abund
    if wave_synth_array is None:
        wave_synth_array = np.arange(wav_start-1, wav_end+1, 0.02)
    sme.wave = wave_synth_array
    sme.linelist = use_list
    
    sme.abund['{}'.format(ele_ion.split(' ')[0])] += abun_shift
    sme = synthesize_spectrum(sme)
    synth_plus = sme.synth[0].copy()
    
    sme.abund['{}'.format(ele_ion.split(' ')[0])] -= abun_shift
    sme = synthesize_spectrum(sme)
    synth_minus = sme.synth[0].copy()
    
    del sme
    
    if error_array is None:
        error_array = np.zeros_like(wave_synth_array)
        error_array[:] = 0.01
    
    abun_pixel_precision = 1 / (np.abs(synth_plus - synth_minus) / (2*abun_shift)) * error_array
    overall_precision = np.sqrt(1 / np.sum(1 / abun_pixel_precision**2))
    
    return overall_precision, synth_minus, synth_plus, abun_pixel_precision

def abund_fit(wav, flux, flux_uncs, teff, logg, m_h, vmic, R, ele, abund, use_list, save_path):

    sme_fit = SME_Structure()

    sme_fit.teff, sme_fit.logg, sme_fit.monh, sme_fit.vmic = teff, logg, m_h, vmic
    sme_fit.iptype = 'gauss'
    sme_fit.ipres = R
    sme_fit.abund = abund
    
    sme_fit.linelist = use_list
    sme_fit.wave = wav
    sme_fit.spec = flux
    sme_fit.uncs = flux_uncs
    
    sme_fit = solve(sme_fit, ['abund {}'.format(ele)])

    fig_sub = plt.figure()
    ax = fig_sub.subplots()
    ax.plot(wav, flux, label='Oberved spectra')
    ax.plot(wav, sme_fit.synth, label='Synthesized spectra')
    ax.title("Fitted A({})={}".format(ele, sme_fit.fitresults['values'][0]))
    fig_sub.savefig(save_path, dpi=150)
    plt.close(fig_sub)
    print('abund_fit:', res)
    
    res = copy(sme_fit.fitresults)
    del sme_fit
    return res

def sigma_clipping(x, x_uncertainties, sigma=3, clip_indices=[None]):

    iterate_switch = True
    if clip_indices[0] == None:
        clip_indices = [True] * len(x)
        
    while iterate_switch:
        average_values = np.average(x[clip_indices], weights=1 / x_uncertainties[clip_indices]**2)
        average_error = np.average((x[clip_indices]-average_values)**2, weights=1 / x_uncertainties[clip_indices]**2)
        average_error = np.sqrt(average_error + 1 / np.sum(1 / x_uncertainties[clip_indices]**2))
        clip_indices_out = np.abs(x - average_values) < sigma * average_error

        if len(x[clip_indices]) == len(x[clip_indices_out]):
            iterate_switch = False
        else:
            clip_indices = clip_indices_out
            
    return average_values, average_error, clip_indices_out

def in_mask(wav, mask):
    in_mask_result = False
    if type(wav) != list:
        for ele in mask:
            in_mask_result = in_mask_result or (wav > ele[0] and wav < ele[1])
        return in_mask_result
    else:
        return np.all([in_mask(wav_single, mask) for wav_single in wav])

def get_line_group(line_list, ele, overwrite=False, line_group_thres=1, line_mask=None):
    '''
    Group the lines with one element to groups, and calculate their senstivity spectra.
    '''

    print('Start line grouping of {}'.format(ele))
    # Extract all the lines of target element
    tar_e_list = line_list[line_list._lines['species'].apply(lambda x: x.split()[0] == ele).values]

    # Find all the species in the line list
    tar_e_index = tar_e_list._lines.groupby('species').size().index

    # Group the lines if they are closer than line_group_thres
    line_group = {}
    for atom in tqdm(tar_e_index, desc=f'Grouping element {ele}'):
        line_group_single = []
        tar_e_list_single = tar_e_list[tar_e_list['species'] == atom]
        for i in tqdm(range(len(tar_e_list_single)-1), desc=f'Grouping {atom} lines'):
            if i == 0:
                line_group_single.append([tar_e_list_single.index[i]])
            if np.abs(tar_e_list_single._lines.iloc[i+1]['wlcent'] - tar_e_list_single._lines.iloc[i]['wlcent']) < line_group_thres:
                line_group_single[-1].append(tar_e_list_single.index[i+1])
            else:
                line_group_single.append([tar_e_list_single.index[i+1]])

            line_group[atom] = [[i] for i in line_group_single]

    for atom in tar_e_index:
        line_group[atom] = [[np.min(tar_e_list._lines.loc[atom_single[0], 'wlcent']), np.max(tar_e_list._lines.loc[atom_single[0], 'wlcent'])] for atom_single in line_group[atom]]

    # Remove those outside the line_mask, if not None
    if line_mask is not None:
        for atom in tar_e_index:
            line_group[atom] = [i for i in line_group[atom] if in_mask(i[0], line_mask)]

    return line_group

def sensitive_spectra(line_group, line_list, stellar_paras, R, ref_spec=None, njobs=50):
    '''
    Get the sensitive spectra for each line group
    ''' 
    teff, logg, m_h, vmic, vmac, vsini, abund = stellar_paras
    
    print('Synthesize the sensitive spectra for all the lines.')
    for atom in line_group.keys():
        args = []
        for i in range(len(line_group[atom])):
            indices = (line_list['wlcent'] > line_group[atom][i][0]-2) & (line_list['wlcent'] < line_group[atom][i][1]+2)
            if ref_spec is None:
                args.append([*line_group[atom][i], teff, logg, m_h, vmic, 0, 0, copy(abund), R, atom, 
                             line_list[indices]])
            else:
                args.append([*line_group[atom][i], teff, logg, m_h, vmic, 0, 0, copy(abund), R, atom, 
                             line_list[indices], ref_spec[0][(ref_spec[0] > line_group[atom][i][0]-2) & (ref_spec[0] < line_group[atom][i][1]+2)],
                             ref_spec[2][(ref_spec[0] > line_group[atom][i][0]-2) & (ref_spec[0] < line_group[atom][i][1]+2)]])
        res = pqdm(args, sensitive_spectra_single, n_jobs=njobs, argument_type='args')

        for i in range(len(line_group[atom])):
            try:
                line_group[atom][i] = [line_group[atom][i]] + list(res[i])
            except:
                print("{} failed.".format(line_group[atom][i]))
                print(res[i])
                line_group[atom][i] = [line_group[atom][i]] + [np.nan, np.array([]), np.array([])]
    return line_group

def abund_fit(wav, flux, flux_uncs, teff, logg, m_h, vmic, vmac, vsini, R, ele, abund, use_list, line_use, save_path):

    sme_fit = SME_Structure()

    sme_fit.teff, sme_fit.logg, sme_fit.monh, sme_fit.vmic, sme_fit.vmac, sme_fit.vsini = teff, logg, m_h, vmic, vmac, vsini
    sme_fit.iptype = 'gauss'
    sme_fit.ipres = R
    sme_fit.abund = abund
    sme_fit.linelist = use_list
    sme_fit.wave = wav
    sme_fit.spec = flux
    sme_fit.uncs = flux_uncs
    
    sme_fit = solve(sme_fit, ['abund {}'.format(ele)])

    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.fill_between(wav, line_use[2], line_use[3], label=f"Spectra with [{ele}/Fe]$\pm$0.1")
    # plt.ylim(0, 1.05)
    # plt.twinx()
    plt.plot(wav, line_use[4], '--', c='C1', label=f"Pixel percision")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.title(f"Final precision: {line_use[1]:.3f}")
    
    plt.subplot(212)
    plt.fill_between(wav, flux-flux_uncs, flux+flux_uncs, label='', alpha=0.3)
    plt.plot(wav, flux, label='Oberved spectra')
    plt.plot(wav, sme_fit.synth[0], label='Synthesized spectra')
    plt.title(f"Fitted A({ele})={sme_fit.fitresults['values'][0]:.3f}$\pm${sme_fit.fitresults['uncertainties'][0]:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path}_{line_use[0][0]:.3f}-{line_use[0][1]:.3f}_line_fit.pdf")
    plt.close()
    
    res = copy(sme_fit.fitresults)
    del sme_fit
    return res

def plot_():

    plt.figure(figsize=(15, 4*2), dpi=150)

    plt.subplot(211)
    plt.plot(wav_obs, flux_obs, lw=0.5, label='Observed spectra')
    plt.ylim(plt.ylim())

    i = 1
    for ele_ion in line_group.keys():
        for ele in line_group[ele_ion]:
            if ele[0][0] == ele[0][1]:
                plt.axvline(ele[0][0], alpha=0.5, color='C{}'.format(i))
            else:
                plt.axvspan(*ele[0], alpha=0.5, color='C{}'.format(i))
        # plot a fake line for labeling
        plt.axhline(-0.1, color='C{}'.format(i), label='{}, {} features'.format(ele_ion, len(line_group[ele_ion])))
        i += 1

    plt.legend(loc=4)
    plt.title('Element {}'.format(ele_fit))

    plt.subplot(212)
    plt.plot(wav_obs, flux_obs, lw=0.5, label='Observed spectra')
    plt.ylim(plt.ylim())

    i = 1
    for ele_ion in line_group.keys():
        # label_dict = {'C{}'.format(i)}
        for ele in line_group[ele_ion]:
            if ele[1] < sensitivity_thres:
                continue
            if ele[0][0] == ele[0][1]:
                plt.axvline(ele[0][0], alpha=0.5, color='C{}'.format(i))
            else:
                plt.axvspan(*ele[0], alpha=0.5, color='C{}'.format(i))
        # plot a fake line for labeling
        plt.axhline(-0.1, color='C{}'.format(i), label='{}, {} features'.format(ele_ion, len([ele for ele in line_group[ele_ion] if ele[1] >= sensitivity_thres])))
        i += 1

    plt.legend(loc=4)
    plt.title('Element {}, sensitivity > {}'.format(ele_fit, sensitivity_thres))
    plt.tight_layout()
    plt.savefig(f'temp-lines.pdf')
    plt.close()
    