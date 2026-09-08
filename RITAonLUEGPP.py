import numpy as np
import pandas as pd
from RITA import (build_RITAxgb, build_RITAgam, EquDistSampling, XGB_addSHAP, unified_dependence_plot, train_XGB)
import fluxcom as flx
from fluxcom.transformers.temporal import StackTime
from LUEmodel import simple_lue_model, f_SM_bellshape, fL, add_noise, f_vpd
from collections import defaultdict
from scipy import stats
import json
import sys, os
import xarray as xr
from sklearn.preprocessing import MinMaxScaler
import glob
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import joblib
from sklearn.model_selection import KFold
import xgboost as xgb
import hashlib
from functions import LE_to_ET, calc_paw

# ── Paths ────────────────────────────────────────────────────────────────────
TOPT_CSV   = "Topt.csv"
OUTPUT_DIR = "output_dir"

# ── Shared XGBoost parameter base ────────────────────────────────────────────
xgb_params = dict(
    colsample_bytree=0.8,
    colsample_bynode=1,
    learning_rate=0.05,
    max_depth=6,
    num_parallel_tree=1,
    objective="reg:squarederror",
    subsample=0.8,
    min_child_weight=10,
    nthread=-1,
    tree_method="auto",
    random_state=42,
)

def to_regular_dict(d):
    if isinstance(d, dict):
        return {k: to_regular_dict(v) for k, v in d.items()}
    return d

# ── Helpers ───────────────────────────────────────────────────────────────────

def evaluate_RCfit(feature_map, shap_values, targets, x):
    """Compute Pearson-r, NSE, MAE and RMSE between recovered SHAP components and true limits."""
    rlt = {}
    for var, var_lim in feature_map.items():
        if isinstance(var, str):
            _, idx = EquDistSampling(x[var], npt=1000)
            label = var
        elif isinstance(var, tuple):
            idx = []
            for i in var:
                _, idx_i = EquDistSampling(x[i], npt=1000)
                idx.extend(idx_i)
            label = "x".join(var)
        else:
            raise ValueError(f"Unsupported feature type: {type(var)}")

        true_full  = targets[var_lim]
        pred_full  = shap_values[label]
        true_smpl  = true_full.iloc[idx]
        pred_smpl  = pred_full.iloc[idx]

        rlt[label] = {
            "pearsonr_ts_full":        stats.pearsonr(true_full, pred_full).statistic,
            "NSE_ts_full":             r2_score(true_full, pred_full),
            "pearsonr_ts_equDistSmpl": stats.pearsonr(true_smpl, pred_smpl).statistic,
            "NSE_ts_equDistSmpl":      r2_score(true_smpl, pred_smpl),
            "MAE_ts_full":             mean_absolute_error(true_full, pred_full),
            "MAE_ts_equDistSmpl":      mean_absolute_error(true_smpl, pred_smpl),
            "RMSE_ts_full":            root_mean_squared_error(true_full, pred_full),
            "RMSE_ts_equDistSmpl":     root_mean_squared_error(true_smpl, pred_smpl)
        }
    return rlt

def gapfill_le(LE, SW_IN):
    le = LE.interpolate(limit=6, limit_direction='both')
    m = le.notna() & SW_IN.notna()
    if m.sum() > 100:
        k = np.polyfit(SW_IN[m], le[m], 1)
        le = le.fillna(pd.Series(np.polyval(k, SW_IN), index=le.index))

    le = le.fillna(le.groupby([le.index.dayofyear, le.index.hour]).transform('mean'))
    le = le.fillna(le.groupby(le.index.dayofyear).transform('mean'))
    return le.fillna(le.mean())

# ── Main ───────────────────────────────────────────────

def main(site: str):
    # ── 1. Load data ──────────────────────────────────────────────────────────
    ec_prov = flx.providers.ERA5FilledEddyProvider(site=[site], transforms=[StackTime()], version="FLUXNET2015")
    ds = ec_prov.get_data(ec_prov.variables)

    rs_prov = flx.providers.MCD43A(site=[site])
    v_rs = rs_prov.get_data([flx.Variable("NDVI")])
    ndvi_raw = xr.where((v_rs['NDVI'] >= 0) & (v_rs['NDVI'] <= 1), v_rs['NDVI'], np.nan)
    # ndvi_smooth, dndvi = smooth_and_differentiate(ndvi_raw)
    ndvi = ndvi_raw.broadcast_like(ds).ffill(dim="time")
    ds = xr.merge([ds, ndvi])

    ip_data = ds[["TA", "TA_QC", "VPD", "VPD_QC", "SW_IN", "SW_IN_QC", "SW_IN_POT", "NDVI", 'LE', 'P']]\
                .sel(site=site).reset_coords(drop=True).to_dataframe()

    # calculate soil moisture proxies
    required = ['LE', 'P', 'TA']
    avail = {v: ip_data[v].dropna() for v in required}

    missing = [v for v, s in avail.items() if s.empty]
    if missing:
        raise ValueError(f"{site}: no data for {missing}")

    trange = [max(s.index.min() for s in avail.values()),
            min(s.index.max() for s in avail.values())]
    if trange[0] >= trange[1]:
        raise ValueError(f"{site}: no overlapping period across {required}")

    data = ip_data.loc[trange[0]:trange[1], :].copy()
    
    le = gapfill_le(data.LE, data.SW_IN)
    ET = LE_to_ET(data['TA'], le) * 3600
    ET = pd.Series(np.asarray(ET), index=data.index)  
    ET = ET.clip(lower=0).fillna(0.0)
    
    data['P'].fillna(0.0, inplace=True)
    # SMs_value, SMs_label = calcWBvars(data['P'].values, ET, surf_caps=[150], iniCWD=150)
    # data[SMs_label] = SMs_value
    assert np.isfinite(data['P']).all() and np.isfinite(ET).all()
    data['PAW'] = calc_paw(data.P, ET, cap=150, n_cycles=2)

    data = data[
        (data["TA_QC"] == 1) & (data["VPD_QC"] == 1) &
        (data["SW_IN_QC"] == 1) & (data["SW_IN"] <= data["SW_IN_POT"]) &
        (data.SW_IN>10)
    ]
    data.dropna(subset=["TA", "VPD", "SW_IN", "PAW", "NDVI", "SW_IN_POT"], inplace=True)

    # ── 2. Generate synthetic GPP ──────────────────────────────────────────────
    Topt_sites = pd.read_csv(TOPT_CSV, index_col='SITE_ID')
    t_opt = Topt_sites.loc[site, 'Topt_LiPan']
    
    def prng(site, name):
        key = f"{site}:{name}".encode()
        return np.random.default_rng(int(hashlib.md5(key).hexdigest()[:8], 16))

    R = lambda n: prng(site, n)
    
    def jitter(lo, hi, rng):
        return float(rng.uniform(lo, hi))
    
    sm_lim_params = {'w_i': jitter(45.0, 75.0,  R('w_i')), 'k_w': -jitter(0.035, 0.07, R('k_w')), 'alpha': 0,} 
    t_lim_params  = {"t_opt": t_opt, "k_t": jitter(7.0,  13.0,  R('k_t')), "alpha_ft": 0}
    vpd_lim_params = {'epsilon': -jitter(0.035, 0.07, R('eps'))}
    rad_lim_params = {"gamma": jitter(0.0003, 0.0015, R('gamma')), "NDVI": data["NDVI"], "a": jitter(2.0,  3.2,   R('a')), "k": jitter(0.40, 0.70,  R('k'))}
    ci_lim_params = {'mu_fci': jitter(0.05, 0.20,  R('mu'))}
    
    Amax = float(ds['GPP'].quantile(0.95).values) * 2
    GPP_est, CI = simple_lue_model(
        data, f_pot=Amax,
        temp_lim_params=t_lim_params,
        vpd_lim_func=f_vpd, vpd_lim_params=vpd_lim_params,
        rad_lim_func=fL, rad_lim_params=rad_lim_params,
        sm_lim_params=sm_lim_params,
        ci_lim_params=ci_lim_params,
    )

    training_data = data.copy()
    # training_data["noisyGPP"] = add_noise(GPP_est["GPP"], noise_type="heteroscedastic+normal", sigma=0.3, seed=6)
    training_data["noisyGPP"] = add_noise(GPP_est["GPP"], seed=4)
    training_data["CI"]  = CI
    # training_data["LAI"] = 10 * (data["NDVI"] ** a)

    dt_idx = training_data.loc[
        (training_data.noisyGPP > 0) & (training_data['CI'].notnull())
    ].index
    training_data = training_data.loc[dt_idx]
    targets = GPP_est.loc[dt_idx]

    FEATURE_LIM   = {
        "TA":               "TA_lim",
        "VPD":              "VPD_lim",
        ("SW_IN", "NDVI"):  "RAD_lim",
        "CI":               "CI_lim",
        "PAW":              "SM_lim",
    }
    
    INPUT_LABEL   = ["TA", "VPD", "SW_IN", "PAW", "CI", "NDVI"]
    X = training_data[INPUT_LABEL]
    X_scaled = pd.DataFrame((X-X.mean(axis=0))/X.std(axis=0), index=X.index, columns=X.columns)
    y = training_data["noisyGPP"]
    
    if len(X) < 500:
        print(f"[SKIP] {site}: insufficient data ({len(X)} samples)")
        return
    
    
    # ── 3. RITA ──────────────────────────────────────────────
    baseline = XGB_addSHAP(X, y, xgb_params)
    baseline['SW_INxNDVI'] = baseline['SW_IN']+baseline['NDVI']
    
    psi_baseline_log, _, _ = build_RITAxgb(X, y, xgb_params)
    psi_baseline_log['SW_INxNDVI'] = psi_baseline_log['SW_IN']*psi_baseline_log['NDVI']
    psi_baseline_log_n = psi_baseline_log/psi_baseline_log.quantile(1, axis=0)
    
    mc_xgb = dict(VPD=-1, SW_IN=1, PAW=1, CI=1, TA=2, NDVI=1)
    ic = [['TA'], ['VPD'], ['SW_IN', 'NDVI'], ['PAW'], ['CI']]
    psi_xgb, peaks, RITAxgb = build_RITAxgb(X, y, xgb_params, monotone_constraints=mc_xgb, interaction_constraints=ic, highlycorrelatedVars=[['TA', 'VPD']])
    psi_xgb['SW_INxNDVI'] = psi_xgb['SW_IN']*psi_xgb['NDVI']
    psi_xgb_norm = (psi_xgb/psi_xgb.quantile(1, axis=0)).clip(0, 1)
    
    ic_nointer = [['TA'], ['VPD'], ['SW_IN'], ['NDVI'], ['PAW'], ['CI']]
    psi_xgb_nointer, peaks_nointer, RITAxgb_nointer = build_RITAxgb(X, y, xgb_params, monotone_constraints=mc_xgb, interaction_constraints=ic_nointer, highlycorrelatedVars=[['TA', 'VPD']])
    psi_xgb_nointer['SW_INxNDVI'] = psi_xgb_nointer['SW_IN']*psi_xgb_nointer['NDVI']
    psi_xgb_nointer_norm = (psi_xgb_nointer/psi_xgb_nointer.quantile(1, axis=0)).clip(0, 1)
    
    ic_allinter = [['TA', 'VPD', 'SW_IN', 'NDVI', 'PAW', 'CI']]
    psi_xgb_allinter, peaks_allinter, RITAxgb_allinter = build_RITAxgb(X, y, xgb_params, monotone_constraints=mc_xgb, interaction_constraints=ic_allinter, highlycorrelatedVars=[['TA', 'VPD']])
    psi_xgb_allinter['SW_INxNDVI'] = psi_xgb_allinter['SW_IN']*psi_xgb_allinter['NDVI']
    psi_xgb_allinter_norm = (psi_xgb_allinter/psi_xgb_allinter.quantile(1, axis=0)).clip(0, 1)
    
    mc_gam = {
        'VPD':  'monotonic_dec',
        'SW_IN': 'monotonic_inc',
        'PAW':  'monotonic_inc',
        'CI':   'monotonic_inc',
        'TA':   'concave',
        'NDVI': 'monotonic_inc'
    }

    psi_gam, RITAgam = build_RITAgam(X_scaled, y, monotone_constraints=mc_gam, interaction_constraints=ic)
    psi_gam_norm = (psi_gam/psi_gam.quantile(1, axis=0)).clip(0, 1)
    print(f'Summary of RITAgam with true interaction constraints\n{RITAgam.summary()}')
    
    psi_gam_nointer, RITAgam_nointer = build_RITAgam(X_scaled, y, monotone_constraints=mc_gam, interaction_constraints=ic_nointer)
    psi_gam_nointer['SW_INxNDVI'] = psi_gam_nointer['SW_IN']*psi_gam_nointer['NDVI']
    psi_gam_nointer_norm = (psi_gam_nointer/psi_gam_nointer.quantile(1, axis=0)).clip(0, 1)
    
    # ── 4. Cross validation ────────────────────────────────────────────────────
    nfolds = 5
    y_cv_gam, y_cv_xgb, y_cv_baseline, y_cv_baselinelog = np.empty(np.size(y)), np.empty(np.size(y)), np.empty(np.size(y)), np.empty(np.size(y))
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(X, y):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        Xscaled_train_fold, Xscaled_test_fold = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        # baseline xval
        baselinemodel = train_XGB(X_train_fold, y_train_fold, xgb_params)
        d_test = xgb.DMatrix(X_test_fold, label=y_test_fold)
        y_cv_baseline[test_index] = baselinemodel.predict(d_test)
        
        # baseline_log xval
        baseline_log_model = train_XGB(X_train_fold, np.log(y_train_fold), xgb_params)
        y_cv_baselinelog[test_index] = np.exp(baseline_log_model.predict(xgb.DMatrix(X_test_fold)))
        
        # RITA_XGB xval
        shap, peaks, xgbmodel = build_RITAxgb(X_train_fold, y_train_fold, xgb_params, mc_xgb, ic, highlycorrelatedVars=[['TA', 'VPD']])
        X['TA_left'] = X['TA'].where(X['TA'] <= peaks['TA'], peaks['TA'])
        X['TA_right'] = X['TA'].where(X['TA'] > peaks['TA'], peaks['TA'])
        # X['WAI_left'] = X['WAI'].where(X['WAI'] <= peaks['WAI'], peaks['WAI'])
        # X['WAI_right'] = X['WAI'].where(X['WAI'] > peaks['WAI'], peaks['WAI'])
        X_scaled_tmp = (X - X.mean(axis=0))/X.std(axis=0)
        X_scaled_test_fold = X_scaled_tmp[shap.columns.drop(['TA'])].iloc[test_index]
        dtest = xgb.DMatrix(X_scaled_test_fold)
        y_cv_xgb[test_index] = np.exp(xgbmodel.model.predict(dtest))

        # RITA_GAM xval
        _, gammodel = build_RITAgam(Xscaled_train_fold, y_train_fold, monotone_constraints=mc_gam, interaction_constraints=ic)
        y_cv_gam[test_index] = gammodel.predict(Xscaled_test_fold)

    nse_xval_gam_noisyGPP = r2_score(y, y_cv_gam)
    nse_xval_xgb_noisyGPP = r2_score(y, y_cv_xgb)
    nse_xval_baseline_noisyGPP = r2_score(y, y_cv_baseline)
    nse_xval_baselinelog_noisyGPP = r2_score(y, y_cv_baselinelog)
    
    nse_xval_gam_trueGPP = r2_score(targets['GPP'], y_cv_gam)
    nse_xval_xgb_trueGPP = r2_score(targets['GPP'], y_cv_xgb)
    nse_xval_baseline_trueGPP = r2_score(targets['GPP'], y_cv_baseline)
    nse_xval_baselinelog_trueGPP = r2_score(targets['GPP'], y_cv_baselinelog)
    
    # ── 4. Collect metrics ────────────────────────────────────────────────────
    results = defaultdict(lambda: defaultdict(dict))
    r = results[site]
    r['y_cv_gam'] = y_cv_gam
    r['y_cv_xgb'] = y_cv_xgb
    r['y_cv_baseline'] = y_cv_baseline
    r['y_cv_baselinelog'] = y_cv_baselinelog
    
    r['nse_xval_gam_trueGPP'] = nse_xval_gam_trueGPP
    r['nse_xval_xgb_trueGPP'] = nse_xval_xgb_trueGPP
    r['nse_xval_baseline_trueGPP'] = nse_xval_baseline_trueGPP
    r['nse_xval_baselinelog_trueGPP'] = nse_xval_baselinelog_trueGPP
    
    r['nse_xval_gam_noisyGPP'] = nse_xval_gam_noisyGPP
    r['nse_xval_xgb_noisyGPP'] = nse_xval_xgb_noisyGPP
    r['nse_xval_baseline_noisyGPP'] = nse_xval_baseline_noisyGPP
    r['nse_xval_baselinelog_noisyGPP'] = nse_xval_baselinelog_noisyGPP
    
    r['y_hat_xgb'] = RITAxgb.model.predict(xgb.DMatrix(RITAxgb.X_scaled[RITAxgb.input_var_labels]))
    r['y_hat_gam'] = RITAgam.predict(X_scaled.values)
    r['psi_baseline'] = baseline
    
    r['psi_baseline_log'] = psi_baseline_log
    
    r['psi_xgb_orig'] = {
        'xgb_nointer':psi_xgb_nointer,
        'xgb_trueinter':psi_xgb,
        'xgb_allinter':psi_xgb_allinter,
    }
    
    r['psi_gam_orig'] = {
        'GAM_nointer': psi_gam_nointer,
        'GAM_trueinter': psi_gam
    }
    
    r['training_data'] = training_data
    r['target']=targets

    r['LUEparams'] = dict(
        Amax=Amax,
        temp_lim_params=t_lim_params,
        sm_lim_params=sm_lim_params,
        vpd_lim_params=vpd_lim_params,
        rad_lim_params=rad_lim_params,
        ci_lim_params=ci_lim_params,
    )
    
    r['Topt_SHAP'] = {
        'xgb_nointer': peaks_nointer['TA'],
        'xgb_trueinter':peaks['TA'],
        'xgb_allinter':peaks_allinter['TA']
    }
    r['Topt_GAM'] = {
        'GAM_trueinter':X.loc[psi_gam['TA'].idxmax(), 'TA'],
        'GAM_nointer':X.loc[psi_gam_nointer['TA'].idxmax(), 'TA']
    }

    r["RCmetrics_XGB_trueinter"] = evaluate_RCfit(FEATURE_LIM, psi_xgb_norm, targets, X)
    r["RCmetrics_XGB_nointer"]   = evaluate_RCfit(FEATURE_LIM, psi_xgb_nointer_norm, targets, X)
    r["RCmetrics_XGB_allinter"]  = evaluate_RCfit(FEATURE_LIM, psi_xgb_allinter_norm, targets, X)
    r["RCmetrics_GAM_trueinter"] = evaluate_RCfit(FEATURE_LIM, psi_gam_norm, targets, X)
    r["RCmetrics_GAM_nointer"]   = evaluate_RCfit(FEATURE_LIM, psi_gam_nointer_norm, targets, X)
    r['RCmetrics_baseline_log']  = evaluate_RCfit(FEATURE_LIM, psi_baseline_log_n, targets, X)
    
    targets_mm  = pd.DataFrame(MinMaxScaler().fit_transform(targets), columns=targets.columns, index=targets.index)
    baseline_mm = pd.DataFrame(MinMaxScaler().fit_transform(baseline), columns=baseline.columns, index=baseline.index)
    r['RCmetrics_baseline'] = evaluate_RCfit(FEATURE_LIM, baseline_mm, targets_mm, X)

    # ── 5. Save metrics ───────────────────────────────────────────────────────
    joblib.dump(to_regular_dict(results), f"{OUTPUT_DIR}/metrics/{site}.joblib")
    
    # ── 6. Plots ──────────────────────────────────────────────────────────────
    truth = targets.rename(columns={'TA_lim':'TA', 'VPD_lim':'VPD', 'CI_lim':'CI', 'RAD_lim':'SW_INxNDVI', 'SM_lim':'PAW'})
    unified_dependence_plot(
        X       = [X, X, X, X],                       
        Y       = [psi_xgb_allinter_norm, psi_xgb_nointer_norm, psi_xgb_norm, truth],                     
        vars_1d = ['TA', 'VPD', 'CI', 'PAW'],   
        vars_2d = [('SW_IN', 'NDVI'), ],       
        labels=['$\mathrm{RITA_{XGB-SHAP-allinter}}$', '$\mathrm{RITA_{XGB-SHAP-nointer}}$', '$\mathrm{RITA_{XGB-SHAP}}$', 'Truth'],
        plt_title = site, hist=False, ncols=4,
        show_marginal=False,
        cmap=['#0072B2', '#D55E00', '#009E73', '#CC79A7'],
        cmap_2d='viridis',
        vmin=0, vmax=1,
        savepath=f'{OUTPUT_DIR}/plots/{site}_RITAxgb_3IC.jpg',
        fontsize_base=14,
    )
    
    unified_dependence_plot(
        X       = [X, X, X, X],                       
        Y       = [baseline_mm, psi_baseline_log_n, psi_xgb_norm, truth],                     
        vars_1d = ['TA', 'VPD', 'CI', 'PAW'],   
        vars_2d = [('SW_IN', 'NDVI'), ],       
        labels=['Baseline', '$\mathrm{Baseline_{log}}$', '$\mathrm{RITA_{XGB-SHAP}}$', 'Truth'],
        plt_title = site, hist=False, ncols=4, 
        show_marginal=False,
        cmap=['#C8CDD2','#0277BD','#C2185B','#000000',],
        cmap_2d='viridis', vmin=0, vmax=1,
        savepath=f'{OUTPUT_DIR}/plots/{site}_RITAxgbVSbaseline.jpg',
        fontsize_base=14,
    )
    
    unified_dependence_plot(
        X       = [X, X, X, X],                       
        Y       = [baseline_mm, truth, psi_xgb_norm, psi_gam_norm],                     
        vars_1d = ['TA', 'VPD', 'CI', 'PAW'],   
        vars_2d = [('SW_IN', 'NDVI'), ],       
        labels=['Baseline', 'Truth', '$\mathrm{RITA_{XGB-SHAP}}$', '$\mathrm{RITA_{GAM}}$'],
        cmap=['#C8CDD2','#000000', '#C2185B','#00796B'],
        plt_title = site, hist=False, ncols=4,
        show_marginal=False,
        cmap_2d='viridis', vmin=0, vmax=1,
        savepath=f'{OUTPUT_DIR}/plots/{site}_RITAxgbVSRITAgam.jpg',
        fontsize_base=14,
    )

    unified_dependence_plot(
        X       = [X, X, X],                       
        Y       = [truth, psi_gam_norm, psi_gam_nointer_norm],                     
        vars_1d = ['TA', 'VPD', 'CI', 'PAW'],   
        vars_2d = [('SW_IN', 'NDVI'), ],       
        labels=['Truth', '$\mathrm{RITA_{GAM}}$', '$\mathrm{RITA_{GAM-nointer}}$'],
        plt_title = site, hist=False, ncols=4,
        show_marginal=False,
        cmap_2d='viridis', vmin=0, vmax=1,
        savepath=f'{OUTPUT_DIR}/plots/{site}_RITAgam_2IC.jpg',
        fontsize_base=14,
    )
    
if __name__ == "__main__":
    main(sys.argv[1])
