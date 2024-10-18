import numpy as np
import pandas as pd
from scipy.signal import convolve
import matplotlib.pyplot as plt

from pysme.sme import SME_Structure
from pysme.synthesize import synthesize_spectrum
from pysme.solve import solve
from copy import copy

def select_lines(spectra, Teff, vald, purity_crit, fwhm, SNR, verbose=False, select_mode='depth'):

    '''
    input:
    ** spectra: pandas.DataFrame containing a column 'wave' with the wavelengths, 'flux_all' with the full spectrum of a star (all the elements, all the molecules, blends etc) and 'flux_el' with the spectrum of a given element only, computed in a similar way as flux_all (in order to have the same continuum opacities)
    ** Teff: Effective Teff, used in Boltzmann equation. 
    ** vald: pandas Dataframe containing the vald line-parameters for the target element only (ll, Echi, loggf) -- (Echi in eV)
    ** purity_crit: minimum required purity to keep the line
    ** fwhm: in A. Resolution element of the spectrograph
    ** SNR : minimum SNR per resolution element (used for line detection)
    ** sampling: spectrum's sampling (in A)
    ** verbose (optional): print information while running
    
    returns: 
    one panda sdata frame, with the following columns:
    ** wlcent : central wavelength where either side of the line has a putiry higher than purity_crit
    ** Blueratio: Purity of the line, defined as the ratio of the element spectrum and the full spectrum at lambda0-1.5xFWHM. 
    ** Redratio : Purity of the line, defined as the ratio of the element spectrum and the full spectrum at lambda0+1.5xFWHM. 
    ** Fullratio: Purity of the line, defined as the ratio of the element spectrum and the full spectrum at lambda0+/-1.5xFWHM. 
    ** Maxratio:  max between the right and the left blend of the line.
    ** fmin: depth of the core of line (as identified by the algorithm) for the element spectrum
    ** fmin_sp: depth of the full spectrum at the position of the core of the line. 
    ** width: width in which the ratio has been computed
    ** BlueFlag: Number of pixels that have a flux>0.9 within 1.5 times the FWHM (resolution element) 
    ** RedFlag: Number of pixels that have a flux>0.9 within 1.5 times the FWHM (resolution element)
    
    
    Steps: 
    1) Identifies the centers of the lines by computing the first two derivatives of the element spectrum.  
    
    2) Does a x-match with the vald linelist. 
    When several VALD lines are within +/- 1 pixel from the derived line core, 
    the line that has: 
      a) the highest line centeral_depth (select_mode is set to 'depth'), or
      b) the highest value of the boltzmann equation  (select_mode is set to 'boltz'),
        is selected.
        ==>log(Boltzmann): log(N)=log(A) -(E_chi/kT)+log(gf)
            log(A) is a constant we can neglect
            loggf is in vald
            T is the temperature of the star
            E_chi is the excitation potential 
        
        ==> Caution: By using Boltzmann equation to select the lines,we assume that for a given element, 
            all of the lines correspond to the same ionisation level. If this is not the case, 
            we need to involve Saha's equation too. This is not implemented yet. 
        ==> Additional Caution: when there is hyperfine structure, then the lambda of Vald that we will 
            find is not necessarily the center of the line we will be seeing

    3) Estimates the depth of the line and compares it to Careyl's formula. sigma_fmin = 1.5/SNR_resol 
    If the depth of the line is large enough to be seen at a given SNR, then the line is selected. 
    
    
    4) We estimate the width of the line as the pixel in which the flux of the element itself is close enough to the continuum. 
    
    Once the line is selected, we compute the ratio between the element spectrum and the full spectrum. 
    Note: we require that if ratio<0.8 then we must have at least two pixels of the total spectrum with flux>0.9 within 1.5 FWHM,
    
    History: 
    04 Oct. 2024: modify the code to support line selection by line central_depth. - MJ
    10 Jun. 2024: modify the code to support pysme VALD linelist format input (not compatible to pandas.DataFrame). - MJ
    20 Apr. 2023: replaced np.argmin (deprecated) with idxmin, that caused code to crash for machines with updated numpy - GK
    10 Feb. 2023: Curated the Code - GK
    04 Feb. 2023: Cleaned the readme. - GK
    
    Contact: Georges Kordopatis - georges.kordopatis -at- oca.eu
             Mingjie Jian - mingjie.jian -at- astro.su.se
    '''

    def _consecutive(data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    
    # Assign the wavelength array to ll.
    ll = spectra['wave'].values

    depth = 1.-3.*(1.5/SNR) # for a 3sigma detection. Based on Careyl's 1988 formula

    # Blindly identify the line's position based on the derivatives. 
    # Take the derivative to find the zero crossings which correspond to
    # the peaks (positive or negative)
    kernel = [1, 0, -1]
    dY = convolve(spectra['flux_el'], kernel, 'same')
    # Use sign flipping to determine direction of change
    S = np.sign(dY)
    ddS = convolve(S, kernel, 'same')
#     print('DERIVATIVES COMPUTED')

    # Find all the indices that appear to be part of a negative peak (absorption lines)
    candidates = np.where(dY < 0)[0] + (len(kernel) - 1)
    line_inds = sorted(set(candidates).intersection(np.where(ddS == 2)[0] + 1))

    # Now group them and find the max highest point.
    line_inds_grouped = _consecutive(line_inds, stepsize=1)

    if len(line_inds_grouped[0]) > 0:
        #absorption_inds = [np.argmin(spectra['flux_el'][inds]) for inds in line_inds_grouped]
        absorption_inds = [spectra['flux_el'][inds].idxmin() for inds in line_inds_grouped]
    else:
        absorption_inds = []
    absorption_ind = np.array(absorption_inds)
    
    # We select the lines that are deep enough to be detected
    zz0 = np.where((spectra['flux_el'].iloc[absorption_ind]<=depth)) [0]  
    zz = absorption_ind[zz0]
    
    # BOLTZMANN METHOD 
    kboltzmann = 8.61733034e-5 # in eV/K
    vald_centers_preliminary = []
    
    # contains the wavelengths (at the pixels) where the first derivative is null and the second is positive
    
    for j in range(0, len(zz)):
        search = np.abs(ll[zz[j]] - vald['wlcent'])
        myvald = np.where((vald['wlcent'] >= ll[zz[j]] - 0.5*fwhm) & (vald['wlcent'] <= ll[zz[j]] + 0.5*fwhm))[0]
        
        if len(myvald) > 1:
            if select_mode == 'boltz':
                myBoltzmann = -vald['excit'][myvald] / (kboltzmann*Teff) + vald['gflog'][myvald]
                mysel = np.where(myBoltzmann == np.max(myBoltzmann))[0]
                vald_centers_preliminary.append(vald['wlcent'][myvald[mysel[0]]])
                if verbose: print(ll[zz[j]],'len(myvald)>1', vald['wlcent'][myvald[mysel[0]]])
            elif select_mode == 'depth':
                mylist = vald._lines.loc[(vald._lines['wlcent'] >= ll[zz[j]] - 0.5*fwhm) & (vald._lines['wlcent'] <= ll[zz[j]] + 0.5*fwhm)]
                idx = mylist['central_depth'].idxmax()
                vald_centers_preliminary.append(mylist.loc[idx, 'wlcent'])
            else:
                raise ValueError("'select_mode' must be either 'depth' or 'boltz'.")
        elif len(myvald) == 1:
            if verbose: print(ll[zz[j]],'-->',len(myvald))
            myvald = np.where(search == np.min(search))[0] # Note that this allows the center of the line to be out of the sampling. 
            vald_centers_preliminary.append(vald['wlcent'][myvald[0]])
            if verbose: print(len(myvald),vald['wlcent'][myvald[0]])
        else: 
            if verbose: print(ll[zz[j]],'-->',len(myvald), ', skip.')
    vald_unique, vald_unique_index = np.unique(np.array(vald_centers_preliminary), return_index=True)

    centers_index = zz[vald_unique_index]
    centers_ll = np.array(vald_unique) 
    
    #Integration of the fluxes in the element spectrum and the full spectrum
    n_lines = len(centers_ll)
    Fratio = np.zeros(n_lines) * np.nan
    Fratio_all = np.zeros(n_lines) * np.nan
    Fratio_blue = np.zeros(n_lines) * np.nan
    Fratio_red = np.zeros(n_lines) * np.nan
    
    width_blue = np.zeros(n_lines) * np.nan
    width_red = np.zeros(n_lines) * np.nan
    
    flag_blue = np.empty(n_lines, dtype=int) * 0
    flag_red = np.empty(n_lines, dtype=int) * 0
    
    half_window_width = 1.5 * fwhm # the total window is 3 fwhm
    
    for j in range(0, n_lines):
        # two selections: blue (left) part of the line, red (right) part of the line
        window_sel_blue = np.where((ll >= centers_ll[j] - half_window_width) & (ll <= centers_ll[j]))[0]
        window_sel_red = np.where((ll <= centers_ll[j] + half_window_width) & (ll >= centers_ll[j]) )[0]
        if len(window_sel_blue) > 0:
            width_blue[j] = ll[window_sel_blue[0]] # this will be overwritten if criteria below are fulfilled.
        else:
            width_blue[j] = 0
        if len(window_sel_red) > 0:
            width_red[j] = ll[window_sel_red[-1]] # this will be overwritten if criteria below are fulfilled.
        else:
            width_red[j] = 0

        for ww in range(0,2): # loop on blue and red wing of the line
            if ww==0: mywindow=window_sel_blue #blue window
            if ww==1: mywindow=window_sel_red # red window

            cont_crit = (1 - np.min(spectra['flux_el'][mywindow])*0.02) #(We are back to the continuum levels more or less 2% of the depth of the line)
            cont_search = np.where(spectra['flux_el'][mywindow] >= cont_crit)[0]
            
            full_continumm_search=np.where(spectra['flux_all'][mywindow]>=0.9)[0] # in order to establish the flags. We want the full spectrum to have a flux >0.9. And we search in a range of +/-1.5FWHM and not the width of the line. 

            if len(cont_search)>=1:
                if ww==0:
                    width_blue[j]=np.max(ll[mywindow[cont_search]])
                    window_sel_blue=np.where((ll>=width_blue[j]) & (ll<=centers_ll[j]))[0]
                    mywindow=window_sel_blue
                if ww==1: 
                    width_red[j]=np.min(ll[mywindow[cont_search]])
                    window_sel_red=np.where((ll<=width_red[j]) & (ll>=centers_ll[j]))[0]
                    mywindow=window_sel_red
                    
            myflux_element=np.sum(1-spectra['flux_el'][mywindow])
            myflux_full_spectrum=np.sum(1-spectra['flux_all'][mywindow])
            myline_flux_ratio=myflux_element/myflux_full_spectrum
                
            if ww==0: 
                Fratio_blue[j]=np.round(myline_flux_ratio,3)
                flag_blue[j]=len(full_continumm_search)
            if ww==1: 
                Fratio_red[j]=np.round(myline_flux_ratio,3)
                flag_red[j]=len(full_continumm_search)
                

        full_window_sel=np.append(window_sel_blue,window_sel_red) # this now contains the full width of the line
        flux_element=np.sum(1-spectra['flux_el'][full_window_sel])
        flux_full_spectrum=np.sum(1-spectra['flux_all'][full_window_sel])
        line_flux_ratio=flux_element/flux_full_spectrum

        Fratio_all[j]=np.round(line_flux_ratio,3)
        Fratio[j]=max([Fratio_blue[j],Fratio_red[j]])
        #print(line_flux_ratio, line_flux_ratio1,line_flux_ratio2,Fratio[j])
            
    keep=np.where(Fratio>purity_crit)[0]
    
    myresult=pd.DataFrame()
    myresult['wlcent'] = np.round(centers_ll[keep],5)
    myresult['Bluewidth']=np.round(width_blue[keep],5)
    myresult['Redwidth']=np.round(width_red[keep],5)
    myresult['Maxratio']=Fratio[keep]
    myresult['fmin']=np.round(spectra['flux_el'][centers_index[keep]].values,3)
    myresult['fmin_sp']=np.round(spectra['flux_all'][centers_index[keep]].values,3)
    myresult['Blueratio']=Fratio_blue[keep]
    myresult['Redratio']=Fratio_red[keep]
    myresult['Fullratio']=Fratio_all[keep]
    myresult['Blueflag']=flag_blue[keep]
    myresult['Redflag']=flag_red[keep]
    
    if verbose: 
        print(centers_ll)
        print('N lines found:',len(vald_unique), ', N lines kept:', len(keep) )

    return(myresult)

def pysme_para_main(wav_obs, flux_obs, flux_err_obs, R, s_n, line_list, teff_init, logg_init, monh_init, vmic_init, vmac_init, vsini_init, ion_list=['Fe 1', 'Fe2'], spec_margin=0.2, linelist_margin=2):
    
    # Find all isolated Fe I and Fe II lines
    wav_start, wav_end = np.min(wav_obs), np.max(wav_obs)
    sme = SME_Structure()

    sme.teff, sme.logg, sme.monh, sme.vmic, sme.vmac, sme.vsini = teff_init, logg_init, monh_init, vmic_init, vmac_init, vsini_init
    sme.iptype = 'gauss'
    sme.ipres = R
    wave_synth_array = np.arange(wav_start, wav_end, 0.05)
    sme.wave = wave_synth_array

    indices = (line_list['wlcent'] >= wav_start) & (line_list['wlcent'] <= wav_end)

    sme.linelist = line_list[indices]
    sme = synthesize_spectrum(sme)
    wav_all, flux_all = copy(sme.wave[0]), copy(sme.synth[0])

    indices_use_paras = wav_obs < 0
    indices_linelist_use_paras = line_list['wlcent'] < 0

    for ion in ion_list:
        fe1_indices = (line_list['species'] == ion)
        sme.linelist = line_list[indices & fe1_indices]
        sme = synthesize_spectrum(sme)
        wav_fe, flux_fe = copy(sme.wave[0]), copy(sme.synth[0])
        spectra = pd.DataFrame({'wave':wav_all, 'flux_all':flux_all, 'flux_el':flux_fe})
        selected_lines_fe1 = select_lines(spectra, teff_init, 
                                    line_list[indices & fe1_indices], 
                                    0.7, 0.2, s_n)
    
    # Select the Fe1 and Fe2 spectral regions, also the sub line list
    
    indices_use_paras = wav_obs < 0
    indices_linelist_use_paras = line_list['wlcent'] < 0
    for i in selected_lines_fe1.index:
        wav_chunk_start, wav_chunk_end = selected_lines_fe1.loc[i, ['Bluewidth', 'Redwidth']].values
        indices_use_paras = indices_use_paras | ((wav_obs >= wav_chunk_start-spec_margin) & (wav_obs <= wav_chunk_end+spec_margin))
        indices_linelist_use_paras = indices_linelist_use_paras | (((line_list['wlcent'] >= wav_chunk_start-linelist_margin) & (line_list['wlcent'] <= wav_chunk_end+linelist_margin)))
    for i in selected_lines_fe2.index:
        wav_chunk_start, wav_chunk_end = selected_lines_fe2.loc[i, ['Bluewidth', 'Redwidth']].values
        indices_use_paras = indices_use_paras | ((wav_obs >= wav_chunk_start-spec_margin) & (wav_obs <= wav_chunk_end+spec_margin))
        indices_linelist_use_paras = indices_linelist_use_paras | (((line_list['wlcent'] >= wav_chunk_start-linelist_margin) & (line_list['wlcent'] <= wav_chunk_end+linelist_margin)))

    wav_obs_use_paras, flux_obs_use_paras = wav_obs[indices_use_paras], flux_obs[indices_use_paras]
    line_list_use_paras = line_list[indices_linelist_use_paras]
    
    plt.figure(figsize=(14, 3), dpi=150)
    plt.plot(wav_obs, flux_obs)
    
    for i in selected_lines_fe1.index:
        plt.axvspan(*selected_lines_fe1.loc[i, ['Bluewidth', 'Redwidth']].values, alpha=0.5, color='C1')
    for i in selected_lines_fe2.index:
        plt.axvspan(*selected_lines_fe2.loc[i, ['Bluewidth', 'Redwidth']].values, alpha=0.5, color='C2')

    plt.scatter(wav_obs_use_paras, flux_obs_use_paras, s=5, c='red', zorder=5)

    sme_fit = SME_Structure()
    sme_fit.teff, sme_fit.logg, sme_fit.monh, sme_fit.vmic, sme_fit.vmac, sme_fit.vsini = teff_init, logg_init, monh_init, vmic_init, vmac_init, vsini_init

    sme_fit.iptype = 'gauss'
    sme_fit.ipres = R

    sme_fit.linelist = line_list_use_paras
    sme_fit.wave = wav_obs_use_paras
    sme_fit.spec = flux_obs_use_paras
    sme_fit.uncs = flux_err_obs

    sme_fit = solve(sme_fit, ['teff', 'logg', 'monh', 'vmic', 'vsini'])

    if sme_fit.vsini > 15:
        print('Large Vsini, second minimization.')
        sme_fit_2 = SME_Structure()
        sme_fit_2.teff, sme_fit_2.logg, sme_fit_2.monh, sme_fit_2.vmic, sme_fit_2.vsini = sme_fit.teff, sme_fit.logg, sme_fit.monh, sme_fit.vmic, sme_fit.vsini
        sme_fit_2.vmac = 0
        sme_fit_2.iptype = 'gauss'
        sme_fit_2.ipres = R
        sme_fit_2.abund = copy(sme_fit.abund)

        sme_fit_2.linelist = sme_fit.linelist
        sme_fit_2.wave = sme_fit.wave
        sme_fit_2.spec = sme_fit.spec
        sme_fit_2.uncs = sme_fit.uncs

        sme_fit_2.accft, sme_fit_2.accgt, sme_fit_2.accxt = 0.1*sme_fit.accft, 0.1*sme_fit.accgt, 0.1*sme_fit.accxt

        sme_fit_2 = solve(sme_fit_2, ['teff', 'logg', 'monh', 'vmic'])
        
        return sme_fit_2
    
    plt.plot(sme_fit.wave[0], sme_fit.synth[0])
    
    return sme_fit