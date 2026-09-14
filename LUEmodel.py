import numpy as np
import pandas as pd


def simple_lue_model(
    input_data,
    f_pot,
    temp_lim_func=None,
    temp_lim_params=None,
    rad_lim_func=None,
    rad_lim_params=None,
    vpd_lim_func=None,
    vpd_lim_params=None,
    sm_lim_func=None,
    sm_lim_params=None,
    ci_lim_func=None,
    ci_lim_params=None,
    lim_floor=0.01,
    **kwargs
):

    if temp_lim_func is None:
        temp_lim_func = f_temp_horn
    temp_lim_params = temp_lim_params or {}
    
    if rad_lim_func is None:
        rad_lim_func = lightresp
    rad_lim_params = rad_lim_params or {}
    
    if vpd_lim_func is None:
        vpd_lim_func = f_vpd
    vpd_lim_params = vpd_lim_params or {}
    
    if sm_lim_func is None:
        sm_lim_func = f_water_horn
    sm_lim_params = sm_lim_params or {}
    
    if ci_lim_func is None:
        ci_lim_func = f_cloud_index_exp
    ci_lim_params = ci_lim_params or {}
    
    ta = input_data['TA']
    sw_in = input_data['SW_IN']
    sw_in_pot = input_data['SW_IN_POT']
    vpd = input_data['VPD']
    sm = input_data['PAW']
    # co2 = input_data.get('CO2_MLO_NOAA', None)
    
    tair_lim = temp_lim_func(ta, **temp_lim_params)
    rg_lim = rad_lim_func(sw_in, **rad_lim_params)
    vpd_lim, *_ = vpd_lim_func(vpd, **vpd_lim_params) 
    sm_lim = sm_lim_func(sm, **sm_lim_params)
    ci_lim, ci = ci_lim_func(sw_in=sw_in, sw_in_pot=sw_in_pot, **ci_lim_params)

    tair_lim = np.clip(tair_lim, lim_floor, None)
    rg_lim = np.clip(rg_lim, lim_floor, None)
    vpd_lim = np.clip(vpd_lim, lim_floor, None)
    sm_lim = np.clip(sm_lim, lim_floor, None)
    ci_lim = np.clip(ci_lim, lim_floor, None)

    gpp = f_pot * tair_lim * rg_lim * vpd_lim * sm_lim * ci_lim
    
    result = pd.DataFrame({
        'GPP': gpp,
        'TA_lim': tair_lim,
        'RAD_lim': rg_lim,
        'VPD_lim': vpd_lim,
        'SM_lim': sm_lim,
        'CI_lim': ci_lim  
    }, index=input_data.index)
    assert (result.drop(columns='GPP') > 0).all().all(), "zero component after rescale"
    return result, ci

def f_temp_horn(temp, t_opt, k_t, alpha_ft):
    """
    calculate partial sensitivity function for temperature
    based on Horn and Schulz (2011)
    ref: https://doi.org/10.5194/bg-8-999-2011

    parameters:
    temp (array): temperature timeseries (degC)
    t_opt (float): parameter to calculate fT
                   (Optimal temperature in degC)
    k_t (float): parameter to calculate fT
                    (Sensitivity to temperature changes; degC-1)
    alpha_ft (float): lag parameter to calculate fT (dimensionless)

    returns:
    ft_horn (array): partial sensitivity function values for temperature
    """

    lag_step = 1  # number of previous timestep to be considered for the lag function

    # initialize t_f array
    t_f = np.zeros_like(temp)
    # calculate t_f
    for idx, tair in enumerate(temp):
        if idx == 0:
            t_f[idx] = (1.0 - alpha_ft) * tair + alpha_ft * tair
        else:
            t_f[idx] = (1.0 - alpha_ft) * tair + alpha_ft * t_f[idx - lag_step]

    # calculate ft_Horn
    # the scalar in numerator has been changed from 4 to 2 and
    # in denominator e(x**2) has been changed to (e(x))**2 to make the fT values between 0 and 1
    ft_eval_exp_num = -(t_f - t_opt) / k_t 
    ft_eval_exp_deno = -(t_f - t_opt) / k_t 
    ft_horn = (2.0 * np.exp(ft_eval_exp_num)) / (
        1.0 + (np.exp(ft_eval_exp_deno)) ** 2.0
    )

    return ft_horn

def lightresp(I, alpha):
    y = 1 - np.exp(-alpha * I)
    return y / np.nanmax(y)

def f_vpd(vpd, epsilon):
    '''VPD in hPa'''
    low_vpd_mask = vpd <= 10
    fvpd = np.exp(epsilon*(vpd-10))
    fvpd[low_vpd_mask] = 1 
    return (fvpd, )

def f_water_horn(sm, w_i, k_w, alpha):
    """
    calculate sensitivity function for moisture stress
    based on Horn and Schulz (2011)
    ref: https://doi.org/10.5194/bg-8-999-2011

    parameters:
    sm (array): soil moisture proxies
    w_i (float): parameter to calculate fW
                (half-saturation point)
    k_w (float): parameter to calculate fW
                 (Sensitivity to soil moisture changes; dimensionless)
    alpha (float): lag parameter to calculate fW (dimensionless)

    returns:
    fw (array): partial sensitivity function values for moisture stress
    """

    # k_w = (
    #     -k_w
    # )  # negative sign is added to k_w as the bounds and initial are made positive
    lag_step = 1  # number of previous timestep to be considered for the lag function

    # initialize w_f array
    w_f = np.zeros_like(sm)
    # calculate w_f
    for idx, wai_nor_val in enumerate(sm):
        if idx == 0:
            w_f[idx] = (1.0 - alpha) * wai_nor_val + alpha * wai_nor_val
        else:
            w_f[idx] = (1.0 - alpha) * wai_nor_val + alpha * w_f[idx - lag_step]
    
    fw_eval_exp = k_w * (w_f - w_i)  
    fw_horn = 1.0 / (1.0 + np.exp(fw_eval_exp))  # calculate fw_Horn

    return fw_horn/fw_horn.max()

def f_cloud_index_exp(mu_fci, sw_in=None, sw_in_pot=None, ci=None):
    """
    calculate sensitivity function for cloudiness index
    based on Bao et al. (2022)
    ref: https://doi.org/10.1016/j.agrformet.2021.108708

    parameters:
    mu_fci (float): parameter to calculate fCI
                    (Sensitivity to cloudiness index changes; dimensionless)
    sw_in (array): incoming shortwave radiation (W m-2)
    sw_in_pot (array): potential incoming shortwave radiation (W m-2)
    ci (array): cloudiness index (dimensionless)
    
    either ci or sw_in and sw_in_pot should be provided to calculate fCI
    
    returns:
    fci (array): partial sensitivity function values for cloudiness index
    ci (array): cloudiness index (dimensionless)
    """
    # when ci is not supplied, calculate it from sw_in and sw_in_pot
    if (ci is None) and (sw_in is not None) and (sw_in_pot is not None):

        # ignore the runtime warning produced by ZeroDivisionError
        # and later deal with the inf and nan values
        with np.errstate(divide="ignore", invalid="ignore"):
            ci = 1.0 - (sw_in / sw_in_pot)

        # set inf (1.0/0) and nan (0.0/0.0) values produced by
        # ZeroDivisionError (when sw_in_pot is 0.0) to 0.0
        sw_in_pot_zero_mask = sw_in_pot == 0.0
        ci = np.where(sw_in_pot_zero_mask, 0.0, ci)

        # physically implausible case (sw_in > sw_in_pot) is also set to 0
        high_sw_in_mask = sw_in > sw_in_pot
        ci = np.where(high_sw_in_mask, 0.0, ci)

    # when ci is supplied, it can be directly used to calculate fCI
    elif ci is not None:
        pass
    else: # when ci is not supplied and sw_in and sw_in_pot are also not supplied
        raise ValueError(
            "Either ci or sw_in and sw_in_pot should be provided to calculate fCI"
        )
    ci = np.clip(ci, 0.05, 1)
    # calculate fCI
    fci = ci**mu_fci

    return fci/fci.max(), ci

def add_noise(gpp, sigma=0.3, floor_frac=0.05, seed=1):
    rng = np.random.default_rng(seed)
    g = np.asarray(gpp, dtype=float)
    floor = floor_frac * g      

    out = g + rng.normal(0, g * sigma, len(g))
    bad = out < floor
    for _ in range(20):                        
        if not bad.any():
            break
        out[bad] = g[bad] + rng.normal(0, g[bad] * sigma, bad.sum())
        bad = out < floor
    out[bad] = floor[bad]                       
    return out
