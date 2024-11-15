import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pysme.sme import SME_Structure
# from pysme.abund import Abund
# from pysme.linelist.vald import ValdFile
from pysme.synthesize import synthesize_spectrum
from pysme.solve import solve
from pysme.smelib import _smelib
from pysme.abund import Abund

from pqdm.processes import pqdm

from copy import copy
import os, pickle, logging

from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

from contextlib import redirect_stdout

from astropy.constants import c

from . import pysme_synth

import time

try:
    from IPython import get_ipython
    if 'ipykernel' in str(get_ipython()):
        from tqdm.notebook import tqdm
    else:
        from tqdm.notebook import tqdm
except:
    from tqdm import tqdm

# def setup_logger(log_dir, log_filename):
#     # 创建日志记录器
#     logger = logging.getLogger(log_filename)
#     logger.setLevel(logging.DEBUG)

#     # 创建文件处理器
#     file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
#     file_handler.setLevel(logging.DEBUG)

#     # 创建日志格式器并将其添加到处理器
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     file_handler.setFormatter(formatter)

#     # 将处理器添加到记录器
#     logger.addHandler(file_handler)
    
#     return logger

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
    # if len(sme.wave) == 0:
    #     return np.nan, np.array([]), np.array([]), np.array([]) 
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
    min_precision = np.min(abun_pixel_precision)
    
    return min_precision, synth_minus, synth_plus, abun_pixel_precision

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

def get_line_group(line_list, ele, R, overwrite=False, line_group_thres=2, line_mask=None, group_max_span=1):
    '''
    Group the lines with one element to groups, and calculate their senstivity spectra.
    '''
    print(f'Start line grouping of {ele}')
    # Extract all the lines of target element
    tar_e_list = line_list[line_list._lines['species'].apply(lambda x: x.split()[0] == ele).values]

    if len(tar_e_list) == 0:
        return {f'{ele} 1':[]}

    # Find all the species in the line list
    tar_e_index = tar_e_list._lines.groupby('species').size().index

    # Group the lines if they are closer than line_group_thres
    line_group = {}

    for atom in tqdm(tar_e_index, desc=f'Grouping element {ele}'):
        line_group_single = []
        tar_e_list_single = tar_e_list[tar_e_list['species'] == atom]

        for i in tqdm(range(np.max([1, len(tar_e_list_single)-1])), desc=f'Grouping {atom} lines'):
            if i == 0:
                line_group_single.append([tar_e_list_single.index[i]])
            if len(tar_e_list_single) > 1:
                if (np.abs(tar_e_list_single._lines.iloc[i+1]['wlcent'] - tar_e_list_single._lines.iloc[i]['wlcent']) < line_group_thres*tar_e_list_single._lines.iloc[i]['wlcent']/R) and (np.abs(tar_e_list_single._lines.loc[line_group_single[-1][0], 'wlcent'] - tar_e_list_single._lines.iloc[i]['wlcent']) <= group_max_span):
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

def sensitive_spectra(line_group, line_list, stellar_paras, R, ref_spec=None, njobs=os.cpu_count()-3, parallel=True, line_range_ratio=3):
    '''
    Get the sensitive spectra for each line group
    ''' 
    teff, logg, m_h, vmic, vmac, vsini, abund = stellar_paras
    # broden_v = np.sqrt(vmic**2 + vmac**2 + vsini**2 + (c.value/1000/R)**2)
    
    print('Synthesize the sensitive spectra for all the lines.')
    for atom in line_group.keys():
        args = []
        if not parallel:
            res = []
        for i in range(len(line_group[atom])):
            line_range_use = [line_group[atom][i][0], line_group[atom][i][1]]
            indices = (line_list['wlcent'] > line_range_use[0]-2) & (line_list['wlcent'] < line_range_use[1]+2)
            if ref_spec is None:
                args_single = [*line_range_use, teff, logg, m_h, vmic, vmac, vsini, copy(abund), R, atom, line_list[indices]]
                args.append(args_single)
            else:
                args_single = [*line_range_use, teff, logg, m_h, vmic, vmac, vsini, copy(abund), R, atom, line_list[indices], ref_spec[0][(ref_spec[0] > line_range_use[0]-2) & (ref_spec[0] < line_range_use[1]+2)], ref_spec[2][(ref_spec[0] > line_range_use[0]-2) & (ref_spec[0] < line_range_use[1]+2)]]
                args.append(args_single)
            if not parallel:
                print(line_range_use)
                print(teff, logg, m_h, vmic, vmac, vsini)
                print(R, atom, line_list[indices])
                res.append(sensitive_spectra_single(*args_single))
        if parallel:
            res = pqdm(args, sensitive_spectra_single, n_jobs=njobs, argument_type='args')
            

        for i in range(len(line_group[atom])):
            try:
                line_group[atom][i] = [line_group[atom][i]] + list(res[i])
            except:
                print("{} failed.".format(line_group[atom][i]))
                print(res[i])
                line_group[atom][i] = [line_group[atom][i]] + [np.nan, np.array([]), np.array([])]
            
    return line_group

def abund_fit(ele, wav, flux, flux_uncs, R, teff, logg, m_h, vmic, vmac, vsini, abund, use_list, line_use, save_path, plot=False, atmo=None):

    sme_fit = SME_Structure()

    sme_fit.teff, sme_fit.logg, sme_fit.monh, sme_fit.vmic, sme_fit.vmac, sme_fit.vsini = teff, logg, m_h, vmic, vmac, vsini
    sme_fit.iptype = 'gauss'
    sme_fit.ipres = R
    sme_fit.abund = abund
    sme_fit.linelist = use_list
    sme_fit.wave = wav
    sme_fit.spec = flux
    sme_fit.uncs = flux_uncs

    if atmo is not None:
        sme_fit.atmo = atmo
        sme_fit.atmo.method = 'embedded'

    sme_fit = synthesize_spectrum(sme_fit)
    con_level = np.median(sme_fit.spec[0] / sme_fit.synth[0])
    sme_fit.spec = sme_fit.spec[0] / con_level
    sme_fit.uncs = sme_fit.uncs[0] / con_level

    # sme_fit = solve(sme_fit, [f'abund {ele}', 'vrad'], bounds=[[-10, -5], [11, 5]])
    # return sme_fit
    sme_fit = solve(sme_fit, [f'abund {ele}'])

    if plot:
        plt.figure(figsize=(10, 6))
        plt.subplot(211)
        plt.fill_between(sme_fit.wave[0], line_use[2], line_use[3], label=f"Spectra with [{ele}/Fe]$\pm$0.1")
        # plt.ylim(0, 1.05)
        # plt.twinx()
        plt.plot(sme_fit.wave[0], line_use[4], '--', c='C1', label=f"Pixel percision")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.title(f"Final precision: {line_use[1]:.3f}")

        plt.subplot(212)
        plt.fill_between(sme_fit.wave[0], sme_fit.spec[0]-sme_fit.uncs[0], sme_fit.spec[0]+sme_fit.uncs[0], label='', alpha=0.3)
        plt.plot(sme_fit.wave[0], sme_fit.spec[0], label='Oberved spectra')
        plt.plot(sme_fit.wave[0], sme_fit.synth[0], label='Synthesized spectra')
        if sme_fit.fitresults['fit_uncertainties'][0] < 8:
            sme_fit.abund[ele] += sme_fit.fitresults['fit_uncertainties'][0] - sme_fit.monh
            sme_fit = synthesize_spectrum(sme_fit)
            plt.plot(sme_fit.wave[0], sme_fit.synth[0], label='', c='C1', ls='--')
            sme_fit.abund[ele] -= 2*sme_fit.fitresults['fit_uncertainties'][0] + sme_fit.monh
            sme_fit = synthesize_spectrum(sme_fit)
            plt.plot(sme_fit.wave[0], sme_fit.synth[0], label='', c='C1', ls='--')
        plt.title(f"Fitted A({ele})={sme_fit.fitresults['values'][0]:.3f}$\pm${sme_fit.fitresults['fit_uncertainties'][0]:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_path}_{line_use[0][0]:.3f}-{line_use[0][1]:.3f}_line_fit.pdf")
        plt.close()
    
    res = copy(sme_fit.fitresults)
    del sme_fit
    return res

# def abund_fit(wav, flux, flux_uncs, teff, logg, m_h, vmic, vmac, vsini, R, ele, abund, use_list, line_use, save_path, plot=True):

#     sme_fit = SME_Structure()

#     sme_fit.teff, sme_fit.logg, sme_fit.monh, sme_fit.vmic, sme_fit.vmac, sme_fit.vsini = teff, logg, m_h, vmic, vmac, vsini
#     sme_fit.iptype = 'gauss'
#     sme_fit.ipres = R
#     sme_fit.abund = abund
#     sme_fit.linelist = use_list
#     sme_fit.wave = wav
#     sme_fit.spec = flux
#     sme_fit.uncs = flux_uncs

#     sme_fit = synthesize_spectrum(sme_fit)
#     con_level = np.median(sme_fit.spec[0] / sme_fit.synth[0])
#     sme_fit.spec = flux / con_level
#     sme_fit.uncs = flux_uncs / con_level
    
#     # sme_fit = solve(sme_fit, [f'abund {ele}', 'vrad'], bounds=[[-10, -5], [11, 5]])
#     # return sme_fit
#     sme_fit = solve(sme_fit, [f'abund {ele}'])

#     if plot:
#         plt.figure(figsize=(10, 6))
#         plt.subplot(211)
#         plt.fill_between(wav, line_use[2], line_use[3], label=f"Spectra with [{ele}/Fe]$\pm$0.1")
#         # plt.ylim(0, 1.05)
#         # plt.twinx()
#         plt.plot(wav, line_use[4], '--', c='C1', label=f"Pixel percision")
#         plt.ylim(0, 1.05)
#         plt.legend()
#         plt.title(f"Final precision: {line_use[1]:.3f}")
        
#         plt.subplot(212)
#         plt.fill_between(wav, flux-flux_uncs, flux+flux_uncs, label='', alpha=0.3)
#         plt.plot(wav, flux, label='Oberved spectra')
#         plt.plot(wav, sme_fit.synth[0], label='Synthesized spectra')
#         sme_fit.abund[ele] += sme_fit.fitresults['fit_uncertainties'][0] - sme_fit.monh
#         sme_fit = synthesize_spectrum(sme_fit)
#         plt.plot(wav, sme_fit.synth[0], label='', c='C1', ls='--')
#         sme_fit.abund[ele] -= 2*sme_fit.fitresults['fit_uncertainties'][0] + sme_fit.monh
#         sme_fit = synthesize_spectrum(sme_fit)
#         plt.plot(wav, sme_fit.synth[0], label='', c='C1', ls='--')
#         plt.title(f"Fitted A({ele})={sme_fit.fitresults['values'][0]:.3f}$\pm${sme_fit.fitresults['fit_uncertainties'][0]:.3f}")
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(f"{save_path}_{line_use[0][0]:.3f}-{line_use[0][1]:.3f}_line_fit.pdf")
#         plt.close()
    
#     res = copy(sme_fit.fitresults)
#     del sme_fit
#     return res

def plot_lines(wave, flux, line_group, line_group_use, teff, logg, m_h, vmic, vmac, vsini, precision_thres, result_folder, line_mask=None):
    '''
    wave: the observed wavelength
    flux: the observed normazlied flux
    '''
    
    ele_fit = list(line_group.keys())[0].split()[0]

    plt.figure(figsize=(15, 4*2), dpi=150)

    plt.subplot(211)
    plt.plot(wave, flux, lw=0.5, label='Observed spectra')
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

    if line_mask is not None:
        for ele in line_mask:
            plt.axvline(ele[0], color='brown', ls='--')
            plt.axvline(ele[1], color='brown', ls='--')

    plt.ylim(0, 1.1)

    plt.legend(loc=4)
    plt.title(f'Element {ele_fit}, Teff={teff:.0f}, logg={logg:.2f}, [Fe/H]={m_h:.2f}, Vmic={vmic:.2f}, Vmia={vmac:.2f}, vsini={vsini:.2f}')

    plt.subplot(212)
    plt.plot(wave, flux, lw=0.5, label='Observed spectra')
    plt.ylim(plt.ylim())

    i = 1
    for ele_ion in line_group_use.keys():
        # label_dict = {'C{}'.format(i)}
        for ele in line_group_use[ele_ion]:
            if ele[1] > precision_thres:
                continue
            if ele[0][0] == ele[0][1]:
                plt.axvline(ele[0][0], alpha=0.5, color='C{}'.format(i))
            else:
                plt.axvspan(*ele[0], alpha=0.5, color='C{}'.format(i))
        # plot a fake line for labeling
        plt.axhline(-0.1, color='C{}'.format(i), label='{}, {} features'.format(ele_ion, len([ele for ele in line_group_use[ele_ion] if ele[1] < precision_thres])))
        i += 1

    if line_mask is not None:
        for ele in line_mask:
            plt.axvline(ele[0], color='brown', ls='--')
            plt.axvline(ele[1], color='brown', ls='--')

    plt.ylim(0, 1.1)
    plt.legend(loc=4)
    plt.title('Element {}, minimum precision < {}'.format(ele_fit, precision_thres))
    plt.tight_layout()
    plt.savefig(f'{result_folder}/{ele_fit}/{ele_fit}-lines.pdf')
    plt.close()

def plot_average_abun(abun_all, line_group, average_ions, average_values, average_error, result_folder, standard_value=None):

    plt.figure(figsize=(13, 4))

    plt.scatter(np.arange(len(abun_all[0])), abun_all[0], zorder=2, color=[f'C{ele-1}' for ele in abun_all[4]])
    plt.ylim(plt.ylim())
    plt.errorbar(np.arange(len(abun_all[0])), abun_all[0], yerr=abun_all[1], fmt='.', zorder=1, color='gray', alpha=0.5)
    
    # Make fake points for legend
    for ele_ion in line_group.keys():
        if int(ele_ion.split()[1]) in average_ions and len(line_group[ele_ion]) > 0:
            plt.scatter([0], abun_all[0][0], zorder=0, label=f'{ele_ion} line', c=f'C{int(ele_ion.split()[1])-1}')

    if standard_value is not None:
        plt.axhline(standard_value, c='C3', label=f'Standard value: {standard_value:.2f}', ls='--')
    plt.axhline(average_values, label='Fitted value', ls='--')
    plt.axhspan(average_values-average_error, average_values+average_error, alpha=0.2, label='Fitted std')

    plt.xticks(np.arange(len(abun_all[0])), abun_all[3], rotation=-45);
    plt.legend(fontsize=7)
    plt.ylabel(f'A({list(line_group.keys())[0].split()[0]})')
    plt.title(f'A({list(line_group.keys())[0].split()[0]})={average_values:.2f}$\pm${average_error:.2f}')
    plt.tight_layout()
    

    plt.savefig(f'{result_folder}/{list(line_group.keys())[0].split()[0]}/{list(line_group.keys())[0].split()[0]}-fit.pdf')
    plt.close()
    
def pysme_abund(wave, flux, flux_err, R, teff, logg, m_h, vmic, vmac, vsini, line_list, fit_ele, result_folder=None, line_list_strong=None, strong_line_element=['H', 'Mg', 'Ca', 'Na'], line_mask=None, abund=None, atmo=None, plot=False, precision_thres=0.2, average_ions=[1, 2], standard_values=None, max_N=None, abund_record=None, save=False, overwrite=False):
    '''
    The main function for determining abundances using pysme.
    Input: observed wavelength, normalized flux, teff, logg, [M/H], vmic, vmac, vsini, line_list, pysme initial abundance list, line mask of wavelength to be removed.
    fit_ele have to be either list or string.
    Output: the fitted abundances and reports on the abundances. Can be more than one element, but we do not do parallal computing, and the elements should be fit in sequence.
    '''

    if abund is None:
        abund = Abund.solar() 
        abund.monh = m_h

    if abund_record is None:
        abund_record = {}

    if type(fit_ele) == str:
        fit_ele = [fit_ele]
    elif type(fit_ele) != list:
        raise TypeError('fit_ele have to be either list or string.')

    if line_list_strong is None:
        line_list_no_strong, line_list_strong = pysme_synth.find_strong_lines(line_list, strong_line_element=strong_line_element)

    if plot:
        # Create sub-folders for the star.
        os.makedirs(f"{result_folder}/", exist_ok=True)
    if result_folder is None:
        log_file = '/dev/null'
    else:
        log_file = f"{result_folder}/pysme-abun.log"
    with redirect_stdout(open(log_file, 'w')):
        # Iterate for all the elements
        ele_i = 0
        for ele in fit_ele:
            if plot:
                # Create sub-folders for each element.
                os.makedirs(f"{result_folder}/{ele}/", exist_ok=True)
                if overwrite:
                    # Remove all the files in each element folder.
                    files = os.listdir(f"{result_folder}/{ele}/")
                    for file in files:
                        file_path = os.path.join(f"{result_folder}/{ele}/", file)
                        
                        # 检查是否是文件（排除子文件夹）
                        if os.path.isfile(file_path):
                            try:
                                os.remove(file_path)
                            except Exception as e:
                                pass
        
            time_select_line_s = time.time()

            line_group = get_line_group(line_list, ele, R, line_mask=line_mask)
            line_group = sensitive_spectra(line_group, line_list, [teff, logg, m_h, vmic, vmac, vsini, abund], R, njobs=5, ref_spec=[wave, flux, flux_err])

            # Select the lines with precision smaller than the threshold
            line_group_use = {}
            line_group_len = 0
            for key in line_group.keys():
                line_group_use[key] = [ele for ele in line_group[key] if ele[1] < precision_thres]
                line_group_len += len(line_group_use[key])

            # Sort line_group_use by precision
            for atom in line_group_use.keys():
                line_group_use[atom] = sorted(line_group_use[atom], key=lambda x: x[1], reverse=False)
                if max_N is not None:
                    line_group_use[atom] = line_group_use[atom][:max_N]

            if plot:
                plot_lines(wave, flux, line_group, line_group_use, teff, logg, m_h, vmic, vmac, vsini, precision_thres, result_folder, line_mask=line_mask)

                if line_group_len > 0:
                    plt.figure()
                    for key in line_group.keys():
                        precision = np.array([ele[1] for ele in line_group[key]])
                        precision = precision[precision <= 0.5]
                        if len(precision) > 1:
                            print(precision)
                            plt.hist(precision, bins=int(np.ceil(np.ptp(precision) / 0.05)), label=f'{key} ({len(precision[precision < precision_thres])}/{len(precision)})', alpha=0.7)
                        elif len(precision) == 1:
                            plt.hist(precision, label=f'{key} ({len(precision[precision < precision_thres])}/{len(precision)})', alpha=0.7)

                    plt.axvline(precision_thres, color='red')
                    plt.xlim(0, 0.5)
                    plt.legend()
                    plt.xlabel('Minimum precision')
                    plt.savefig(f'{result_folder}/{ele}/{ele}-precisions.pdf')
                    plt.close()

            time_select_line_e = time.time()

            if line_group_len == 0:
                # No good lines, return current abundnce as NaN.
                print('No good lines.')
                abund[ele] = np.nan
                abund_record[ele] = [np.nan, np.nan, time_select_line_e-time_select_line_s, np.nan]
                ele_i += 1
                continue
            
            # Perform the chi^2 minimization for each useful lines
            res_all = {}
            
            for ele_ion in line_group_use.keys():
                res_all[ele_ion] = []
                for i in tqdm(range(len(line_group_use[ele_ion]))):
                # for i in tqdm([48, 49, 50]):
                    indices = (wave > line_group_use[ele_ion][i][0][0] - 2) & (wave < line_group_use[ele_ion][i][0][1] + 2)
                    
                    use_list_indices = (line_list_no_strong['wlcent'] > line_group_use[ele_ion][i][0][0]-2) & (line_list_no_strong['wlcent'] < line_group_use[ele_ion][i][0][1]+2)
                    use_list = line_list_no_strong[use_list_indices]
                    # Add strong lines
                    linelist_df = pd.concat([use_list._lines, line_list_strong._lines]).sort_values('wlcent')
                    use_list._lines = linelist_df
                    use_list_indices = ((use_list['wlcent'] > line_group_use[ele_ion][i][0][0]-100) & (use_list['wlcent'] < line_group_use[ele_ion][i][0][1]+100))
                    use_list = use_list[use_list_indices]
                    res = abund_fit(ele_ion.split()[0], wave[indices], flux[indices], flux_err[indices], R, teff, logg, m_h, vmic, vmac, vsini, copy(abund), use_list, line_group_use[ele_ion][i], f"{result_folder}/{ele}/{ele_ion.replace(' ', '_')}", atmo=atmo, plot=plot)
                    line_group_use[ele_ion][i].append(res)
                    res_all[ele_ion].append(res)
            
            # Get the final average abundance for current element
            abun_result = {}
            abun_all = [[], [], [], [], []]
            for ele_ion in line_group.keys():
                abun_result[ele_ion] = {}
                abun_result[ele_ion]['values'] = np.array([ele.values[0] for ele in res_all[ele_ion]])
                abun_result[ele_ion]['fit_uncertainties'] = np.array([ele.uncertainties[0] for ele in res_all[ele_ion]])
                if int(ele_ion.split()[1]) in average_ions:
                    abun_all[0] += [ele.values[0] for ele in res_all[ele_ion]]
                    abun_all[1] += [ele.fit_uncertainties[0] for ele in res_all[ele_ion]]
                    abun_all[2] += [ele[0][0] for ele in line_group_use[ele_ion]] 
                    abun_all[3] += ['{:.3f}-{:.3f}$\mathrm{{\AA}}$'.format(*ele[0]) for ele in line_group_use[ele_ion]]
                    abun_all[4] += [int(ele_ion.split()[1])] * len(res_all[ele_ion])

            time_chi2_e = time.time()

            # Sort the abun_all (for plotting)
            abun_all = [np.array(i) for i in abun_all]
            sort_index = abun_all[2].argsort()
            abun_all = [ele[sort_index] for ele in abun_all]

            abun_all = [np.array(i) for i in abun_all]
            if len(abun_all) > 0:

                average_values = np.average(abun_all[0], weights=1 / abun_all[1]**2)
                average_error = np.average((abun_all[0]-average_values)**2, weights=1 / abun_all[1]**2)
                average_error = np.sqrt(average_error + 1 / np.sum(1 / abun_all[1]**2))

                if plot:
                    if standard_values is not None:
                        plot_average_abun(abun_all, line_group, average_ions, average_values, average_error, result_folder, standard_value=standard_values[0][ele_i])
                    else:
                        plot_average_abun(abun_all, line_group, average_ions, average_values, average_error, result_folder)
                # Update the abund
                abund[ele] = average_values - m_h
                abund_record[ele] = [average_values, average_error, time_select_line_e-time_select_line_s, time_chi2_e-time_select_line_e]

            ele_i += 1

        if plot:
            # Plot the final abundance and comparison
            plt.figure(figsize=(14, 3))
            plot_x = []
            label_func1 = lambda x: 'standard abunds' if x == 0 else ''
            label_func2 = lambda x: 'pysme abunds' if x == 0 else ''
            
            if standard_values is not None:
                plt.scatter(range(len(fit_ele)), standard_values[0])
            plt.scatter(range(len(fit_ele)), [abund_record[ele][0] for ele in fit_ele])
            plt.ylim(plt.ylim())

            j = 0
            for ele in fit_ele:
                plot_x.append(j)
                if standard_values is not None:
                    plt.errorbar(j, standard_values[0][j],
                                yerr=standard_values[1][j], fmt='.', alpha=0.7, label=label_func1(j), color='C0')
                plt.errorbar(j, abund_record[ele][0], yerr=abund_record[ele][1], fmt='.', alpha=0.7, label=label_func2(j), color='C1')
                j += 1
            plt.xticks(plot_x, fit_ele)
            plt.legend()
            plt.ylabel('A(X)')
            plt.tight_layout()
            plt.savefig(f'{result_folder}/abund-result.pdf')
            plt.close()

            plt.figure(figsize=(14, 3))
            plot_x = []
            label_func1 = lambda x: 'standard abunds error' if x == 0 else ''
            label_func2 = lambda x: 'pysme abunds error' if x == 0 else ''

            if standard_values is not None:
                plt.scatter(range(len(fit_ele)), np.array([abund_record[ele][0] for ele in fit_ele]) - np.array(standard_values[0]), zorder=3)
                plt.ylim(plt.ylim())
                
                j = 0
                for ele in fit_ele:
                    plot_x.append(j)
                    if standard_values is not None:
                        plt.errorbar(range(len(fit_ele)), np.array([abund_record[ele][0] for ele in fit_ele]) - np.array(standard_values[0]),
                                    yerr=standard_values[1], fmt='.', alpha=0.7, label=label_func1(j), color='C0', zorder=1)
                    plt.errorbar(range(len(fit_ele)), np.array([abund_record[ele][0] for ele in fit_ele]) - np.array(standard_values[0]), 
                                yerr=[abund_record[ele][1] for ele in fit_ele], fmt='.', alpha=0.7, label=label_func2(j), color='C1', zorder=2)
                    j += 1
                plt.axhline(0, ls='--', color='brown')
                plt.xticks(plot_x, fit_ele)
                plt.legend()
                plt.ylabel('A(X)$_\mathrm{measure}$ - A(X)$_\mathrm{standard}$')
                plt.tight_layout()
                plt.grid(zorder=0)
            else:
                plt.title('No standard value.')
            plt.savefig(f'{result_folder}/diff-result.pdf')
            plt.close()

        if save:
            pickle.dump([abund_record, abund], open(f'{result_folder}/abun_res.pkl', 'wb'))
            abun_res_df = pd.DataFrame(abund_record).T
            abun_res_df.columns = ['A(X)', 'err_A(X)', 'time_line_selection' ,'time_chi2']
            abun_res_df.to_csv(f'{result_folder}/abun_fit.csv')

    return abund_record, abund