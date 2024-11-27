import numpy as np
import spectres, scipy

def log_wav_raster(wave):
    a = wave[1] / wave[0]
    length = int(np.ceil(np.log10(wave[-1]/wave[0]) / np.log10(a)))
    return wave[0] * a ** np.arange(length)

def log_wav2rv(wav):
    '''
    Convert the log wavelangth scale to RV.

    :param wav:
        A numpy.array, containing the logarithmic wavelength scale of the spectra.
    :return:
        The RV array. 
    '''
    
    # check if the a value is constant for all the pixels
    if not np.all(wav):
        raise ValueError('The wavelength array is not in constant log scale.')
        
    c = 299792458.0 / 1000
    a = wav[1] / wav[0]
    if len(wav) % 2 == 1:
        # Odd case
        multiplicative_shift = a**(np.arange(1, (len(wav)-1) // 2 + 1) / 1)
        rv_array = c * (a**np.arange(1, len(wav) // 2 + 1) - 1)
        rv_array = c * (multiplicative_shift**2 - 1) / (multiplicative_shift**2 + 1)
        rv_array = np.concatenate([-rv_array[::-1], [0], rv_array])
    else:
        # Even case
        wav_temp = np.append(wav, wav[0] * a**len(wav))
        rv_array = log_wav2rv(wav_temp)[:-1]
            
    return rv_array

def interpolation_quadratic_dcf(x_vals, y_vals):
    """
    Use quadratic interpolation to find the sub-pixel position of the peak of the CCF. 
    This is Dominic's version which uses analytic solution, modified by Mingjie to do the fitting in lambda scale.

    :param x_vals:
        If the CCF is y(x), this is the array of the x values of the data points supplied to interpolate.
    :param y_vals:
        If the CCF is y(x), this is the array of the y values of the data points supplied to interpolate.
    :return:
        Our best estimate of the position of the peak.
    """

    assert len(x_vals) == 3  # This analytic solution only works with three input points
    assert x_vals[1] == 0  # Three points must be centred around x==0

    def quadratic_peak_x(p):
        return -p[1] / (2 * p[0])

    p0 = (y_vals[0] + y_vals[2]) / (x_vals[0]**2 - x_vals[0]*x_vals[2]) + y_vals[1] / (x_vals[0]*x_vals[2])

    p1 = -(y_vals[0]*x_vals[2]) / (x_vals[0]**2 - x_vals[0]*x_vals[2]) 
    p1 -= y_vals[1]*(x_vals[0]+x_vals[2]) / (x_vals[0]*x_vals[2])
    p1 -= x_vals[0]*y_vals[2] / (x_vals[2]**2 - x_vals[0]*x_vals[2])
    
    p2 = y_vals[1]

    peak_x = quadratic_peak_x(p=(p0, p1, p2))

    return p0, p1, p2, peak_x

def measure_rv_from_ccf(rv_raster, cross_correlation, interpolation_pixels=3):
    
    max_position = np.where(cross_correlation == np.max(cross_correlation))[0][0]
    N = len(rv_raster)
    
    # Now make three points which straddle the maximum
    x_min = max_position - interpolation_pixels // 2
    x_max = x_min + interpolation_pixels - 1
    x_indices = np.array(range(x_min, x_max + 1))
    x_vals = rv_raster[x_indices]
    y_vals = cross_correlation[x_indices]

    # Put peak close to zero for numerical stability
    x_vals = x_vals - rv_raster[max_position]
    y_peak = y_vals[1]

    # Do interpolation
    if len(x_vals) == 3:
        dcf_para = interpolation_quadratic_dcf(x_vals=x_vals, y_vals=y_vals)
        peak_x = dcf_para[-1] + rv_raster[max_position]
        rv_err = np.sqrt(- 1 / N * y_peak / (dcf_para[0] * 2) * (1 - y_peak**2) / y_peak**2)
    return peak_x, rv_err

def measure_rv(wav_template, flux_template, wav_spec, flux_spec, ccf_out=False):
    log_wave = log_wav_raster(wav_template)
    flux_rv_template = spectres.spectres(log_wave, wav_template, flux_template, fill=1)
    flux_rv_template -= np.nanmean(flux_rv_template)
    flux_rv_template /= np.nanstd(flux_rv_template)
    flux_rv_template[np.isnan(flux_rv_template)] = 0
    rv_raster = log_wav2rv(log_wave)

    flux_rv_obs = spectres.spectres(log_wave, wav_spec, flux_spec, fill=1)
    flux_rv_obs -= np.nanmean(flux_rv_obs)
    flux_rv_obs/= np.nanstd(flux_rv_obs)
    flux_rv_obs[np.isnan(flux_rv_obs)] = 0
    ccf = scipy.signal.correlate(flux_rv_obs, flux_rv_template, mode='same') / len(log_wave)
    star_rv, rv_err = measure_rv_from_ccf(rv_raster, ccf)
    if ccf_out:
        return star_rv, rv_err, rv_raster, ccf
    else:
        return star_rv, rv_err