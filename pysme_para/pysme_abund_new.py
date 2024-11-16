from pysme.sme import SME_Structure
from pysme.abund import Abund
from pysme.synthesize import synthesize_spectrum
from pysme.solve import solve

from . import pysme_synth

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, time, pickle
from contextlib import redirect_stdout
from tqdm.notebook import tqdm
from copy import copy

def get_sensitive_synth(wave, R, teff, logg, m_h, vmic, vmac, vsini, line_list, abund, ele_fit, ion_fit):

    # Generate the synthetic spectra using current parameters
    sme = SME_Structure()
    sme.teff, sme.logg, sme.monh, sme.vmic, sme.vmac, sme.vsini = teff, logg, m_h, vmic, vmac, vsini
    sme.iptype = 'gauss'
    sme.ipres = R
    sme.abund = abund
    sme.wave = wave
    spec_syn_all = pysme_synth.batch_synth_line_range(sme, line_list, parallel=True, n_jobs=10)

    # Calculate the sensitive spectra for the fitting elements.
    spec_syn = {}
    i = 0
    for ele in tqdm(ele_fit):
        sme.abund = abund
        sme.abund[ele] += 0.1
        spec_syn_plus = pysme_synth.batch_synth_line_range(sme, line_list, parallel=True, n_jobs=10)
        sme.abund = abund
        sme.abund[ele] -= 0.2
        spec_syn_minus = pysme_synth.batch_synth_line_range(sme, line_list, parallel=True, n_jobs=10)
        sme.abund[ele] += 0.1

        spec_syn_ion = {}
        for ion in ion_fit:
            try:
                spec_syn_ion[ion] = pysme_synth.batch_synth_line_range(sme, line_list[line_list['species'] == f'{ele} {ion}'], parallel=True, n_jobs=10)[1]
            except:
                spec_syn_ion[ion] = np.ones_like(spec_syn_minus[0])

        spec_syn[ele] = {'minus':spec_syn_minus[1],
                            'plus':spec_syn_plus[1],
                            'partial_derivative':np.abs(spec_syn_plus[1]-spec_syn_minus[1])/2, 
                            'ele_only':spec_syn_ion}
        if i == 0:
            total_partial_derivative = np.abs(spec_syn_plus[1]-spec_syn_minus[1])/2
        else:
            total_partial_derivative += np.abs(spec_syn_plus[1]-spec_syn_minus[1])/2
        i += 1

    spec_syn['total'] = {'wave':spec_syn_all[0],
                            'flux':spec_syn_all[1],
                            'minus':np.array([]),
                            'plus':np.array([]),
                            'partial_derivative':total_partial_derivative}
    
    return spec_syn

def select_lines(wave, spec_syn, ele_fit, ion_fit, line_list, v_broad, margin_ratio=2, sensitivity_dominance_thres=0.6, line_dominance_thres=0.6, max_line_num=10):

    '''
    Select lines
    '''
    
    # Get the line range for fitting
    fit_line_group = {}

    for ele in ele_fit:
        fit_line_group[ele] = {}
        for ion in ion_fit: 
            fit_line_group_single = []
            for i in line_list[line_list['species'] == f'{ele} {ion}'].index:
                line_wav = line_list._lines.loc[i, 'wlcent']
                fit_line_group_single.append([line_wav * (1-margin_ratio*v_broad / 3e5), line_wav * (1+margin_ratio*v_broad / 3e5)])
            
            # Remove the line group outside the observed spectra
            fit_line_group_single = [ele for ele in fit_line_group_single if ele[0] >= np.min(wave) and ele[1] <= np.max(wave) and len(wave[(wave>=ele[0]) & (wave<=ele[1])]) > 1]
            # Merge the line_group if they are connected
            fit_line_group_single.sort(key=lambda x: x[0])
            merged_ranges = []
            for current_range in fit_line_group_single:
                if not merged_ranges:
                    merged_ranges.append(current_range)
                else:
                    last_range = merged_ranges[-1]
                    # 检查当前范围的起始值是否小于等于前一个范围的结束值
                    if current_range[0] <= last_range[1]:
                        # 更新已合并范围的结束值为当前范围和前一个范围的结束值的最大者
                        last_range[1] = max(last_range[1], current_range[1])
                    else:
                        merged_ranges.append(current_range)
            fit_line_group[ele][ion] = [{'wav_s':ele[0], 'wav_e':ele[1]} for ele in merged_ranges]

    # Calculate the line range parameters
    for ele in ele_fit:
        for ion in ion_fit:
            for i in range(len(fit_line_group[ele][ion])):            
                wav_range = [fit_line_group[ele][ion][i]['wav_s'], fit_line_group[ele][ion][i]['wav_e']]
                
                indices = (spec_syn['total']['wave'] >= wav_range[0]) & (spec_syn['total']['wave'] <= wav_range[1]) 
                max_sensitivity = np.max(spec_syn[ele]['partial_derivative'][indices])
                sensitivity_dominance = np.sum(spec_syn[ele]['partial_derivative'][indices]) / np.sum(spec_syn['total']['partial_derivative'][indices]) 
                line_dominance = np.sum(1 - spec_syn[ele]['ele_only'][ion][indices]) / np.sum(1 - spec_syn['total']['flux'][indices]) 
                fit_line_group[ele][ion][i]['max_sensitivity'] = max_sensitivity
                fit_line_group[ele][ion][i]['sensitivity_dominance'] = sensitivity_dominance
                fit_line_group[ele][ion][i]['line_dominance'] = line_dominance
            if len(fit_line_group[ele][ion]) > 0:
                fit_line_group[ele][ion] = pd.DataFrame(fit_line_group[ele][ion])
            else:
                fit_line_group[ele][ion] = pd.DataFrame(pd.DataFrame(columns=['wav_s', 'wav_e', 'max_sensitivity', 'sensitivity_dominance', 'line_dominance']))

    # Select the lines
    for ele in ele_fit:
        for ion in ion_fit:
            indices = (fit_line_group[ele][ion]['sensitivity_dominance'] >= sensitivity_dominance_thres) & (fit_line_group[ele][ion]['line_dominance'] >= line_dominance_thres)
            fit_line_group[ele][ion] = fit_line_group[ele][ion][indices].sort_values('max_sensitivity', ascending=False)[:max_line_num].reset_index(drop=True)

    return fit_line_group

def abund_fit(ele, ion, wav, flux, flux_uncs, fit_range, R, teff, logg, m_h, vmic, vmac, vsini, abund, use_list, 
              spec_syn, synth_margin=2,
              save_path=None, plot=False, atmo=None, normalization=False, nlte=False):

    # Crop the spectra 
    indices = (wav >= fit_range[0]-synth_margin) & (wav <= fit_range[1]+synth_margin)
    wav = wav[indices]
    flux = flux[indices]
    flux_uncs = flux_uncs[indices]
    # Crop the line list
    use_list = use_list[(use_list['line_range_e'] > fit_range[0]-synth_margin) & (use_list['line_range_s'] < fit_range[1]+synth_margin)]
    
    sme_fit = SME_Structure()
    sme_fit.teff, sme_fit.logg, sme_fit.monh, sme_fit.vmic, sme_fit.vmac, sme_fit.vsini = teff, logg, m_h, vmic, vmac, vsini
    sme_fit.iptype = 'gauss'
    sme_fit.ipres = R
    sme_fit.abund = copy(abund)
    sme_fit.linelist = use_list
    indices = (wav >= fit_range[0]) & (wav <= fit_range[1])
    sme_fit.wave = wav
    sme_fit.spec = flux
    sme_fit.uncs = flux_uncs
    mask = np.zeros_like(sme_fit.wave[0])
    mask[indices] = 1

    if nlte:
        sme_fit.nlte.set_nlte(ele)

    if atmo is not None:
        sme_fit.atmo = atmo
        sme_fit.atmo.method = 'embedded'

    if normalization:
        sme_fit = synthesize_spectrum(sme_fit)
        con_level = np.median(sme_fit.spec[0] / sme_fit.synth[0])
        sme_fit.spec = sme_fit.spec[0] / con_level
        sme_fit.uncs = sme_fit.uncs[0] / con_level

    sme_fit = solve(sme_fit, [f'abund {ele}'])
    fit_flag = 'normal'
    
    # Calculate the EW
    EW_all = np.trapz(sme_fit.synth[0], sme_fit.wave[0])
    best_fit_synth = sme_fit.synth[0].copy()
    sme_fit.linelist = use_list[use_list['species'] == f'{ele} {ion}']
    sme_fit = synthesize_spectrum(sme_fit)
    EW = - (EW_all - np.trapz(sme_fit.synth[0], sme_fit.wave[0])) * 1000
    sme_fit.linelist = use_list
    if sme_fit.fitresults['fit_uncertainties'][0] < 8:
        sme_fit.abund[ele] += sme_fit.fitresults['fit_uncertainties'][0] - sme_fit.monh
        sme_fit = synthesize_spectrum(sme_fit)
        EW_plus = np.trapz(sme_fit.synth[0], sme_fit.wave[0])
        plus_fit_synth = sme_fit.synth[0].copy()
        sme_fit.abund[ele] -= 2*sme_fit.fitresults['fit_uncertainties'][0] + sme_fit.monh
        sme_fit = synthesize_spectrum(sme_fit)
        EW_minus = np.trapz(sme_fit.synth[0], sme_fit.wave[0])
        minus_fit_synth = sme_fit.synth[0].copy()
        diff_EW = np.abs(EW_plus-EW_minus)/2
        if EW <= 2*diff_EW:
            fit_flag = 'upper_limit'
    else:
        diff_EW = np.nan
        fit_flag = 'error'


    if plot:
        plt.figure(figsize=(10, 6))
        plt.subplot(211)
        indices = (spec_syn['total']['wave'] >= fit_range[0]-2) & (spec_syn['total']['wave'] <= fit_range[1]+2)
        plt.fill_between(spec_syn['total']['wave'][indices], spec_syn[ele]['minus'][indices], spec_syn[ele]['plus'][indices], label=f"Synthetic spectra with [{ele}/Fe]$\pm$0.1")
        plt.plot(spec_syn['total']['wave'][indices], spec_syn[ele]['ele_only'][ion][indices], c='C1', label=f"Synthetic spectra with {ele} {ion} line only")
        plt.axvspan(*fit_range, color='C1', alpha=0.2)
        plt.legend()
        plt.title(f'{ele} {ion} fitting')

        plt.subplot(212)
        plt.fill_between(sme_fit.wave[0], sme_fit.spec[0]-sme_fit.uncs[0], sme_fit.spec[0]+sme_fit.uncs[0], label='', alpha=0.3)
        plt.plot(sme_fit.wave[0], sme_fit.spec[0], label='Oberved spectra')
        plt.plot(sme_fit.wave[0], best_fit_synth, label='Synthesized spectra')
        if sme_fit.fitresults['fit_uncertainties'][0] < 8:
            plt.plot(sme_fit.wave[0], plus_fit_synth, label='', c='C1', ls='--')
            plt.plot(sme_fit.wave[0], minus_fit_synth, label='', c='C1', ls='--')
        
        plt.axvspan(*fit_range, color='C1', alpha=0.2)
        if sme_fit.fitresults['fit_uncertainties'][0] < 8:
            plt.title(f"Fitted A({ele})={sme_fit.fitresults['values'][0]:.3f}$\pm${sme_fit.fitresults['fit_uncertainties'][0]:.3f}, EW={EW:.2f}$\pm${diff_EW:.2f} m$\mathrm{{\AA}}$")
        else:
            plt.title(f"Fitted A({ele})={sme_fit.fitresults['values'][0]:.3f}$\pm${sme_fit.fitresults['fit_uncertainties'][0]:.3f}, bad fitting")
        plt.legend()
        plt.xlabel('Wavelength ($\mathrm{\AA}$)')
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(f"{save_path}_{fit_range[0]:.3f}-{fit_range[1]:.3f}_line_fit.pdf")
            plt.close()
    
    fitresults = copy(sme_fit.fitresults)
    del sme_fit
    return (fitresults, EW, diff_EW, fit_flag)

def plot_average_abun(ele, fit_line_group_ele, ion_fit, result_folder, standard_value=None):

    plt.figure(figsize=(13, 4))

    for ion in ion_fit:
        indices = fit_line_group_ele['fit_result']['ioni_state'] == ion
        plt.scatter(fit_line_group_ele['fit_result'].index[indices], fit_line_group_ele['fit_result'].loc[indices, f'A({ele})'], zorder=2, label=f'{ele} {ion} line')
    plt.ylim(plt.ylim())
    plt.errorbar(fit_line_group_ele['fit_result'].index, fit_line_group_ele['fit_result'][f'A({ele})'], 
                 yerr=fit_line_group_ele['fit_result'][f'err_A({ele})'], fmt='.', zorder=1, color='gray', alpha=0.5)
    
    if standard_value is not None:
        plt.axhline(standard_value, c='C3', label=f'Standard value: {standard_value:.2f}', ls='--')
    plt.axhline(fit_line_group_ele['average_abundance'], label='Fitted value', ls='--')
    plt.axhspan(fit_line_group_ele['average_abundance']-fit_line_group_ele['average_abundance_err'], fit_line_group_ele['average_abundance']+fit_line_group_ele['average_abundance_err'], alpha=0.2, label='Fitted std')
    
    plt.xticks(fit_line_group_ele['fit_result'].index, ['{:.3f}-{:.3f}$\mathrm{{\AA}}$'.format(fit_line_group_ele['fit_result'].loc[i, 'wav_s'], fit_line_group_ele['fit_result'].loc[i, 'wav_s']) for i in fit_line_group_ele['fit_result'].index], rotation=-90);
    plt.legend(fontsize=7)
    plt.ylabel(f'A({ele})')
    plt.title(f"A({ele})={fit_line_group_ele['average_abundance']:.2f}$\pm${fit_line_group_ele['average_abundance_err']:.2f}")
    plt.tight_layout()
    plt.grid()
    
    print('{result_folder}/{ele}/{ele}-fit.pdf')
    plt.savefig(f'{result_folder}/{ele}/{ele}-fit.pdf')
    plt.close()

def pysme_abund(wave, flux, flux_err, R, teff, logg, m_h, vmic, vmac, vsini, line_list, ele_fit, ion_fit=[1, 2], result_folder=None, line_mask=None, abund=None, atmo=None, plot=False, standard_values=None, max_N=None, abund_record=None, save=False, overwrite=False, line_margin=2, central_depth_thres=0.01, cal_central_depth=True):
    '''
    The main function for determining abundances using pysme.
    Input: observed wavelength, normalized flux, teff, logg, [M/H], vmic, vmac, vsini, line_list, pysme initial abundance list, line mask of wavelength to be removed.
    ele_fit have to be either list or string.
    Output: the fitted abundances and reports on the abundances. Can be more than one element, but we do not do parallal computing, and the elements should be fit in sequence.

    Parameters
    ----------
    wave : 

    Returns
    -------
    abund_record : 
        
    abund : 
    '''

    if abund is None:
        abund = Abund.solar() 
        abund.monh = m_h

    if abund_record is None:
        abund_record = {}

    if type(ele_fit) == str:
        ele_fit = [ele_fit]
    elif type(ele_fit) != list:
        raise TypeError('ele_fit have to be either list or string.')

    if plot or save:
        # Create sub-folders for the star.
        os.makedirs(f"{result_folder}/", exist_ok=True)
    if result_folder is None:
        log_file = '/dev/null'
    else:
        log_file = f"{result_folder}/pysme-abun.log"
    with redirect_stdout(open(log_file, 'w')):

        time_select_line_s = time.time()

        v_broad = np.sqrt(vmic**2 + vsini**2 + (3e5/R)**2)

        if 'central_depth' not in line_list.columns or 'line_range_s' not in line_list.columns or 'line_range_e' not in line_list.columns or cal_central_depth:
            # Calculate the central_depth and line_range, if required or no such column
            sme = SME_Structure()
            sme.teff, sme.logg, sme.monh, sme.vmic, sme.vmac, sme.vsini = teff, logg, m_h, vmic, vmac, vsini
            line_list = pysme_synth.get_cdepth_range(sme, line_list, parallel=True, n_jobs=10)
        line_list = line_list[line_list['central_depth'] > central_depth_thres]

        # Generate synthetic and sensitive spectra using current parameters
        spec_syn = get_sensitive_synth(wave, R, teff, logg, m_h, vmic, vmac, vsini, line_list, abund, ele_fit, ion_fit)
        fit_line_group = select_lines(wave, spec_syn, ele_fit, ion_fit, line_list, v_broad)

        time_select_line_e = time.time()

        # Iterate for all the elements
        ele_fit_count = 0
        for ele in ele_fit:
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
            for ion in ion_fit:
                fit_result = []
                if len(fit_line_group[ele][ion]) == 0:
                    fit_line_group[ele][ion] = pd.DataFrame(pd.DataFrame(columns=list(fit_line_group[ele][ion].columns) + [f'A({ele})', f'err_A({ele})', 'EW', 'diff_EW', 'flag']))
                for i in fit_line_group[ele][ion].index:
                    fit_range = [fit_line_group[ele][ion].loc[i, 'wav_s'],  fit_line_group[ele][ion].loc[i, 'wav_e']]
                    fitresults, EW, diff_EW, fit_flag = abund_fit(ele, ion, wave, flux, flux_err, fit_range, R, teff, logg, m_h, vmic, vmac, vsini, abund, line_list, spec_syn,
                                                                    save_path=f"{result_folder}/{ele}/{ele}_{ion}", atmo=None, plot=plot)
                    fit_result.append({f'A({ele})':fitresults.values[0], f'err_A({ele})':fitresults.fit_uncertainties[0], 'EW':EW, 'diff_EW':diff_EW, 'flag':fit_flag})
                fit_line_group[ele][ion] = pd.concat([fit_line_group[ele][ion], pd.DataFrame(fit_result)], axis=1)

            abun_all = np.concatenate([fit_line_group[ele][i].loc[fit_line_group[ele][i]['flag'] == 'normal', f'A({ele})'].values for i in ion_fit])
            abun_err_all = np.concatenate([fit_line_group[ele][i].loc[fit_line_group[ele][i]['flag'] == 'normal', f'err_A({ele})'].values for i in ion_fit])
            weights = 1 / abun_err_all**2
            average_values = np.average(abun_all, weights=weights/np.sum(weights))
            average_error = np.average((abun_all-average_values)**2, weights=weights/np.sum(weights))
            average_error = np.sqrt(average_error + 1 / np.sum(weights))
            fit_line_group[ele]['average_abundance'] = average_values
            fit_line_group[ele]['average_abundance_err'] = average_error

            i = 0
            for ion in ion_fit:
                if i == 0:
                    fit_result_df = fit_line_group[ele][ion]
                    fit_result_df['element'] = ele
                    fit_result_df['ioni_state'] = ion
                else:
                    temp = fit_line_group[ele][ion]
                    temp['element'] = ele
                    temp['ioni_state'] = ion
                    fit_result_df = pd.concat([fit_result_df, temp])
                del fit_line_group[ele][ion]
                i += 1
            fit_line_group[ele]['fit_result'] = fit_result_df.sort_values('wav_s').reset_index(drop=True)

            if plot:
                if standard_values is not None:
                    plot_average_abun(ele, fit_line_group[ele], ion_fit, result_folder, standard_value=standard_values[0][ele_fit_count])
                else:
                    plot_average_abun(ele, fit_line_group[ele], ion_fit, result_folder)
            ele_fit_count += 1

            time_chi2_e = time.time()
            fit_line_group[ele]['line_selection_time'] = time_select_line_e - time_select_line_s
            fit_line_group[ele]['chi2_time'] = time_chi2_e - time_select_line_e

    if plot:
        # Plot the final abundance and comparison
        plt.figure(figsize=(14, 3))
        plot_x = []
        label_func1 = lambda x: 'standard abunds' if x == 0 else ''
        label_func2 = lambda x: 'pysme abunds' if x == 0 else ''
        
        if standard_values is not None:
            plt.scatter(range(len(ele_fit)), standard_values[0])
        plt.scatter(range(len(ele_fit)), [fit_line_group[ele]['average_abundance'] for ele in ele_fit])
        plt.ylim(plt.ylim())

        j = 0
        for ele in ele_fit:
            plot_x.append(j)
            if standard_values is not None:
                plt.errorbar(j, standard_values[0][j], yerr=standard_values[1][j], fmt='.', alpha=0.7, label=label_func1(j), color='C0')
            plt.errorbar(j, fit_line_group[ele]['average_abundance'], yerr=fit_line_group[ele]['average_abundance_err'], fmt='.', alpha=0.7, label=label_func2(j), color='C1')
            j += 1
        plt.xticks(plot_x, ele_fit)
        plt.legend()
        plt.ylabel('A(X)')
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'{result_folder}/abund-result.pdf')
        plt.close()

        plt.figure(figsize=(14, 3))
        plot_x = []
        label_func1 = lambda x: 'standard abunds error' if x == 0 else ''
        label_func2 = lambda x: 'pysme abunds error' if x == 0 else ''

        if standard_values is not None:
            plt.scatter(range(len(ele_fit)), np.array([fit_line_group[ele]['average_abundance'] for ele in ele_fit]) - np.array(standard_values[0]), zorder=3)
            plt.ylim(plt.ylim())
            
            j = 0
            for ele in ele_fit:
                plot_x.append(j)
                if standard_values is not None:
                    plt.errorbar(range(len(ele_fit)), np.array([fit_line_group[ele]['average_abundance'] for ele in ele_fit]) - np.array(standard_values[0]),
                                yerr=standard_values[1], fmt='.', alpha=0.7, label=label_func1(j), color='C0', zorder=1)
                plt.errorbar(range(len(ele_fit)), np.array([fit_line_group[ele]['average_abundance'] for ele in ele_fit]) - np.array(standard_values[0]), 
                            yerr=[fit_line_group[ele]['average_abundance_err'] for ele in ele_fit], fmt='.', alpha=0.7, label=label_func2(j), color='C1', zorder=2)
                j += 1
            plt.axhline(0, ls='--', color='brown')
            plt.xticks(plot_x, ele_fit)
            plt.ylabel('A(X)$_\mathrm{measure}$ - A(X)$_\mathrm{standard}$')
            plt.tight_layout()
            plt.grid(zorder=0)
        else:
            plt.title('No standard value.')
        plt.savefig(f'{result_folder}/diff-result.pdf')
        plt.close()

    if save:
        pickle.dump(fit_line_group, open(f'{result_folder}/abun_res.pkl', 'wb'))
        abun_res_df = pd.DataFrame({'element':ele_fit, 
                                    'A(X)':[fit_line_group[ele]['average_abundance'] for ele in ele_fit],
                                    'err_A(X)':[fit_line_group[ele]['average_abundance_err'] for ele in ele_fit], 
                                    'line_selection_time':[fit_line_group[ele]['line_selection_time'] for ele in ele_fit],
                                    'chi2_time':[fit_line_group[ele]['chi2_time'] for ele in ele_fit]})
        # abun_res_df.columns = ['A(X)', 'err_A(X)', 'time_line_selection' ,'time_chi2']
        abun_res_df.to_csv(f'{result_folder}/abun_fit.csv')

    return fit_line_group