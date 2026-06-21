import numpy as np 
import shap
import pandas as pd 
import xgboost as xgb
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error
from functools import reduce
import operator
from pygam import GAM, s, te
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import binned_statistic_2d

class ThresholdProcessor:
    def __init__(self, threshold):
        self.threshold = threshold
        self.lt_idx = None
        self.gt_idx = None
        self.lt_label = None
        self.gt_label = None
        self.original_label = None
    
    def preprocess(self, data, op_labels=None):
        self.lt_idx = data < self.threshold
        self.gt_idx = data > self.threshold
        self.original_label = data.name
        self.lt_label = op_labels[0] if op_labels is not None else data.name+'_lt'
        self.gt_label = op_labels[1] if op_labels is not None else data.name+'_gt'
        
        results = {self.lt_label: data.where(self.lt_idx, self.threshold), 
                   self.gt_label: data.where(self.gt_idx, self.threshold)}
        return results
    
    def merge(self, data):
        result = {self.original_label:data[self.lt_label].where(self.lt_idx, data[self.gt_label]).values}
        return result


def train_XGB(X, y, params, weight=None):  
    if weight is not None:
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, weight, test_size=0.2, random_state=1)
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
        Dpred = xgb.DMatrix(X_test, label=y_test, weight=w_test)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        Dpred = xgb.DMatrix(X_test, label=y_test)
    xgb_model_tmp = xgb.train(params, dtrain, num_boost_round=10000, early_stopping_rounds=10, evals=[(Dpred, 'eval')], verbose_eval=False)
    Ddata = xgb.DMatrix(X, label=y, weight=weight)
    xgb_model_full = xgb.train(params, Ddata, num_boost_round=xgb_model_tmp.best_iteration, verbose_eval=False)
    return xgb_model_full


def sampleEquidist_idx(x,npt=100,pcts=[0,100]):
    xx=np.percentile(x,pcts)
    xout=np.linspace(xx[0], xx[1], num=npt)
    idxs = []
    for s in np.arange(npt):
        idx=np.argmin(np.abs(x-xout[s]))
        idxs.append(idx)
    return (xout,idxs)


# smooth unimodal response curves to get optimal point
def get_Opt(x, y, method='spline', smooth_param=None, verbose=True):
    from scipy.signal import find_peaks
    # sampling
    _, idx = sampleEquidist_idx(x, npt=1000)
    x_sample = x[idx]
    y_sample = y[idx]
    order = np.argsort(x_sample)
    x_sorted, y_sorted = x_sample[order], y_sample[order]

    x_grid = np.linspace(x_sorted.min(), x_sorted.max(), 1000)
    if method == 'spline':
        from scipy.interpolate import UnivariateSpline
        spline = UnivariateSpline(x_sorted, y_sorted, s=smooth_param)
        y_smooth = spline(x_grid)

    elif method == 'lowess':
        from statsmodels.nonparametric.smoothers_lowess import lowess
        y_loess = lowess(endog=y_sorted, exog=x_sorted, frac=smooth_param, return_sorted=False)
        y_smooth = np.interp(x_grid, x_sorted, y_loess)
    
    else:
        raise ValueError("Unsupported method")
    
    # peaks, _ = find_peaks(y_smooth)
    # if len(peaks) > 0:
    #     main_peak_idx = peaks[np.argmax(y_smooth[peaks])]
    # else:
    #     if verbose:
    #         print(f"⚠️ {method} 没有检测到明显峰值，使用全局最大值作为主峰")
    main_peak_idx = np.argmax(y_smooth)
    
    return x_grid[main_peak_idx]


class RITAExplainer:
    def __init__(self, X, y, xgb_params, normaliseX=True, config=None):
        self.X = X.copy()
        self.y = y.copy()
        self.lny = np.log(self.y) 
        self.xgb_params = xgb_params.copy()
        self.processed_monotonic_constraints = {}

        config = config or {}
        self.weight = config.get("weight", None)
        self.shap_normalize = config.get("shap_normalize", False)
        self.shap_cleaning = config.get("shap_cleaning", False)
        self.cleaning_method = config.get("cleaning_method", "default")  
        self.cleaning_params = config.get("cleaning_params", {})
        self.shapExplainerParams = config.get("shapExplainerParams", {})
        self.unimodal_split_points = config.get("unimodal_split_points", {}) 
        self.normaliseX = normaliseX
        # internal cache
        self.model = None
        self.shap_add_const = None
        self.shap_multip_const = None
        self.X_scaled = None
        self.shap_add = None
        self.shap_add_clean = None
        self.shap_multip = None
        self.local_importance = None
        self.global_importance = None
        self.input_var_labels = None

        self._handle_constraints()
        self._fit_and_explain()
        self.compute_multip_shap(return_const=True)
        if len(self.processed_monotonic_constraints)>0:
            for key, values in self.processed_monotonic_constraints.items():
                self.shap_multip[key] = self.shap_multip[values].prod(axis=1)

    def _handle_constraints(self):
        orig_mc = self.xgb_params.get("monotone_constraints", None)
        orig_ic = self.xgb_params.get("interaction_constraints", None)

        has_mc = bool(orig_mc)
        has_ic = bool(orig_ic)

        if (has_mc and not has_ic) or (has_ic and not has_mc) or (not has_mc and not has_ic):
            self.input_var_labels = list(self.X.columns)
            return

        if not hasattr(self, "processed_monotonic_constraints"):
            self.processed_monotonic_constraints = {}

        new_mc = {}
        input_var_labels = []
        replace_map = {}

        def is_top_level_singleton(feature, ic_list):
            if not ic_list:
                return False
            for entry in ic_list:
                if isinstance(entry, list) and len(entry) == 1 and entry[0] == feature:
                    return True
            return False

        for feature, value in orig_mc.items():
            if value == 2:
                independent = is_top_level_singleton(feature, orig_ic)
                if independent:
                    if feature not in self.unimodal_split_points:
                        raise ValueError(f"Missing unimodal split point for feature '{feature}'")

                    split = self.unimodal_split_points[feature]
                    left_feature = f"{feature}_left"
                    right_feature = f"{feature}_right"

                    self.X[left_feature] = self.X[feature].where(self.X[feature] <= split, split)
                    self.X[right_feature] = self.X[feature].where(self.X[feature] > split, split)

                    new_mc[left_feature] = 1
                    new_mc[right_feature] = -1

                    input_var_labels.extend([left_feature, right_feature])

                    replace_map[feature] = (left_feature, right_feature)
                    self.processed_monotonic_constraints[feature] = [left_feature, right_feature]
                else:
                    new_mc[feature] = 0
                    input_var_labels.append(feature)
            else:
                new_mc[feature] = value
                input_var_labels.append(feature)

        new_ic = []
        for entry in orig_ic:
            if isinstance(entry, list) and len(entry) == 1 and entry[0] in replace_map:
                left_f, right_f = replace_map[entry[0]]
                new_ic.extend([[left_f], [right_f]])
            else:
                new_ic.append(entry)

        self.xgb_params["monotone_constraints"] = new_mc
        self.xgb_params["interaction_constraints"] = new_ic
        self.input_var_labels = input_var_labels
        
    def _fit_and_explain(self):
        if self.normaliseX:
            self.X_scaled = pd.DataFrame(StandardScaler().fit_transform(self.X), columns=self.X.columns, index=self.X.index)
            inputs = self.X_scaled[self.input_var_labels]
        else:
            inputs = self.X[self.input_var_labels]
        
        self.model = train_XGB(inputs, self.lny, self.xgb_params, weight=self.weight)
        
        shap_params = self.shapExplainerParams.copy()
        if shap_params.get("feature_perturbation") == "interventional" and "data" not in shap_params:
            selected_years = np.random.choice(inputs.index.year.unique(), size=1, replace=False)
            shap_params["data"] = (
                inputs[inputs.index.year.isin(selected_years)]
                .astype("float64")
                .copy()
            )
        self.explainer = shap.TreeExplainer(self.model, **shap_params)
        self.explanation = self.explainer(inputs)
        self.shap_add = pd.DataFrame(
            self.explanation.values,
            columns=self.explanation.feature_names,
            index=inputs.index
        )
        self.shap_add_const = self.explainer.expected_value
    
    def clean_shap(self, data=None, cleaning_method=None, cleaning_params={}, save=True):
        if data is None:
            if self.shap_add is None:
                raise ValueError("SHAP additive values not computed yet.")
            data = self.shap_add.values
            
        if cleaning_method == "default":
            cleaned = pd.DataFrame(cleanEffectsMult(data, **cleaning_params), columns=self.shap_add.columns, index=self.shap_add.index)
            if save:
                self.shap_add_clean = cleaned
        elif cleaning_method == "jumps":
            pass
        else:
            raise NotImplementedError(f"Cleaning method '{cleaning_method}' not supported.")
        if save:
            self.shap_cleaning = True
            
        return cleaned

    def compute_multip_shap(self, data=None, return_const=False, save=True):
        # base = self.clean_shap_add() if self.shap_cleaning else self.shap_add
        if data is None:
            data = self.shap_add
            
        shap_multip = np.exp(data.values)

        df_shap_multip = pd.DataFrame(shap_multip, columns=data.columns, index=data.index)

        if save:
            self.shap_multip = df_shap_multip

        if return_const and self.shap_add_const is not None:
            self.shap_multip_const = np.exp(self.shap_add_const)
            return self.shap_multip_const, df_shap_multip
        else:
            return df_shap_multip

    def compute_local_importance(self, save=True, method='MaxNormalize'):
        if self.shap_multip is None:
            self.compute_multip_shap()
            
        if method == 'X-SHAP':
            local_I = np.maximum(self.shap_multip, 1 / self.shap_multip)
        elif method == 'MaxNormalize':
            local_I = RITAExplainer._max_normalize(self.shap_multip, shap_type='multiplicative')
        else:
            print('Please specify the method to compute local importance')
            
        if save:
            self.local_importance = local_I
        return local_I

    def compute_global_importance(self, data=None, save=False):
        if data is None:
            if self.local_importance is None:
                self.compute_local_importance()
            global_I = np.exp(np.log(self.local_importance).mean(axis=0))
        else:
            global_I = np.exp(np.log(data).mean(axis=0))
        if save:
            self.global_importance = global_I
        return global_I

    @staticmethod
    def _max_normalize(data, shap_type=None):
        norm_factor = np.nanmax(data, axis=0)
        if shap_type == 'additive':
            return (norm_factor, np.exp(data - norm_factor))
        elif shap_type == 'multiplicative':
            return (norm_factor, data / norm_factor)
        return data


def dependence_plot(X, Y, labels=None, plt_title=None, multiYaxis=False, ylim=(0, 1.05), ncols=2, hist=True, savepath=None, cmap=None, texts=None, text_pos=(0.05, 0.95), **kwargs):
    
    rc_params = {
        'font.family': 'Arial',
        'axes.titlesize': 32,
        'axes.labelsize': 26,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 12,
    }
    
    with plt.rc_context(rc_params):  # All rcParams changes are scoped here only
        nvar = len(X.columns) if isinstance(X, pd.DataFrame) else len(X[0].columns)
        nrows = math.ceil(nvar / ncols)

        fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                                figsize=(kwargs.get("figsize", (ncols*5, nrows * 5))))
        fig.suptitle(plt_title, fontsize=16)
        axes = axes.flatten()

        for i in range(nvar):
            ax = axes[i]
            var_name = X.columns[i] if isinstance(X, pd.DataFrame) else X[0].columns[i]

            if isinstance(Y, list) and isinstance(X, list):
                if multiYaxis == False:
                    if cmap is None:
                        colors = plt.cm.Set1.colors
                    elif hasattr(cmap, 'colors'):
                        colors = cmap.colors
                    elif isinstance(cmap, (list, tuple)):
                        colors = cmap
                    else:
                        raise ValueError("Invalid cmap format. Provide a matplotlib colormap or a list of color strings.")

                    for idx, (x, y) in enumerate(zip(X, Y)):
                        color = colors[idx % len(colors)]
                        ax.scatter(x.iloc[:, i], y.iloc[:, i], label=labels[idx] if labels is not None else str(idx), color=color, s=10, alpha=0.3)

                    ax.legend(
                        fontsize=max(8, 11 - nvar // 4),
                        markerscale=2.0,
                        handletextpad=0.4,
                        borderpad=0.4,
                        labelspacing=0.3,
                    )
                    ax.set_ylim(ylim)
                    ax.set_ylabel('Multiplicative contribution', fontsize=13)
                else:
                    ax_list = []
                    for idx, (x, y) in enumerate(zip(X, Y)):
                        if idx == 0:
                            ax_list.append(ax)
                        else:
                            ax_new = ax.twinx()
                            ax_list.append(ax_new)
                            ax_new.spines['right'].set_position(('outward', 60 * (idx - 1)))

                        ax_curr = ax_list[idx]
                        if cmap is None:
                            color = plt.cm.tab10(idx)
                        elif hasattr(cmap, 'colors'):
                            color = cmap(idx)
                        elif isinstance(cmap, (list, tuple)):
                            color = cmap[idx]
                        else:
                            raise ValueError("Invalid cmap format. Provide a matplotlib colormap or a list of color strings.")
                        ax_curr.scatter(x.iloc[:, i], y.iloc[:, i], label=labels[idx] if labels is not None else str(idx), color=color, s=10, alpha=0.3)
                        ax_curr.set_ylabel(f'{labels[idx]}', fontsize=13, color=color)
                        ax_curr.tick_params(axis='y', labelcolor=color)
                        if isinstance(ylim, tuple):
                            ax_curr.set_ylim(ylim)
                        elif isinstance(ylim, list):
                            ax_curr.set_ylim(ylim[idx])

                ax.set_xlabel(x.columns[i], fontsize=16)

            else:
                ax.scatter(X.iloc[:, i], Y.iloc[:, i], s=10, alpha=0.3)
                ax.set_ylim(ylim)
                ax.set_ylabel(f'SHAP_x value for {Y.columns[i]}', fontsize=13)
                ax.set_xlabel(X.columns[i], fontsize=16)

            if texts is not None:
                text = texts.get(var_name)
                if text is not None:
                    ax.text(text_pos[0], text_pos[1], text, transform=ax.transAxes, va='top', ha='left',
                            bbox=dict(facecolor='white', alpha=0.8), fontsize=9)

            ax.ticklabel_format(style='sci', axis='x', scilimits=(-2, 3))

            if hist:
                hist_data = x.iloc[:, i] if isinstance(X, list) else X.iloc[:, i]
                ax_hist = ax.twinx()
                if multiYaxis and isinstance(X, list):
                    n_y_axes = len(X)
                    ax_hist.spines['right'].set_position(('outward', 60 * (n_y_axes - 1)))
                ax_hist.hist(hist_data, bins=100, alpha=0.3, color='gray', density=True)
                ax_hist.set_ylim(0, ax_hist.get_ylim()[1])
                ax_hist.set_yticks([])

        for j in range(nvar, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if savepath is not None:
            plt.savefig(savepath)
        plt.show()

def unified_dependence_plot(
    X,
    Y,
    vars_1d=None,
    vars_2d=None,
    labels=None,
    plt_title=None,
    multiYaxis=False,
    ylim=(0, 1.05),
    ncols=2,
    hist=True,
    savepath=None,
    cmap=None,
    texts=None,
    text_pos=(0.05, 0.95),
    bins_2d=50,
    vmin=None,
    vmax=None,
    cmap_2d='RdBu',
    show_marginal=True,  
    **kwargs
):
    """
    Unified dependence plot.

    Parameters
    ----------
    X : pd.DataFrame or list of pd.DataFrame
        Input features.
    Y : pd.DataFrame, list of pd.DataFrame, or dict of {label: pd.DataFrame/Series}
        Output / SHAP values.
    vars_1d : list of str, optional
        Column names in X to plot as 1D dependence panels.
        e.g. ['SW_IN', 'NDVI', 'Tair']
    vars_2d : list of tuple, optional
        Pairs (x_var, y_var) to plot as 2D heatmap + marginal panels.
        e.g. [('SW_IN', 'NDVI'), ('Tair', 'VPD')]

    labels : list of str, optional
        Labels for multiple Y series (1D multi-series + 2D marginal legends).
    plt_title : str, optional
        Figure suptitle.
    multiYaxis : bool
        Use multiple y-axes for 1D multi-series scatter.
    ylim : tuple
        Default y-limits for 1D plots.
    ncols : int
        Number of subplot columns.
    hist : bool
        Overlay histogram on 1D plots.
    savepath : str, optional
        Save path for the figure.
    cmap : colormap or list
        Colour source for 1D multi-series scatter.
    texts : dict, optional
        {var_name: annotation_string} placed on 1D panels.
    text_pos : tuple
        Axes-fraction position for text annotations.
    bins_2d : int
        Bins for 2D binned statistic.
    vmin, vmax : float
        Global colour limits for 2D heatmaps.
    colorbar_label : str
        Default colourbar label.
    cmap_2d : str or Colormap
        Default colourmap for 2D heatmaps.
    """

    vars_1d  = vars_1d or []
    vars_2d  = vars_2d or []

    rc_params = {
        'font.family'      : 'Arial',
        'axes.titlesize'   : 14,
        'axes.labelsize'   : 11,
        'xtick.labelsize'  : 9,
        'ytick.labelsize'  : 9,
        'legend.fontsize'  : 9,
        'figure.titlesize' : 14,
    }
    panel_labels = 'abcdefghijklmnopqrstuvwxyz'

    # ── total panels = 1D vars + 2D pairs ──────────────────────────────
    if isinstance(Y, pd.DataFrame):
        nvar  = len(vars_1d) + len(vars_2d)
    if isinstance(Y, (list, dict)):
        nvar  = len(vars_1d) + len(vars_2d)*len(Y)
    
    nrows = math.ceil(nvar / ncols)
    

    # ================================================================
    # Utility helpers
    # ================================================================
    def _psi_label(x_var, y_var=None):
        def _fmt(v):
            # keep _ visible by using \_ inside \mathrm
            safe = v.replace('_', r'\_')
            return f'\\mathrm{{{safe}}}'
        
        if y_var is None:
            return f'$\\psi_{{{_fmt(x_var)}}}$'
        else:
            return f'$\\psi_{{{_fmt(x_var)} \\times {_fmt(y_var)}}}$'



    def _get_colors_1d(n):
        if cmap is None:
            return [plt.cm.Set1(k % 9) for k in range(n)]
        elif hasattr(cmap, 'colors'):
            return [cmap.colors[k % len(cmap.colors)] for k in range(n)]
        elif isinstance(cmap, (list, tuple)):
            return [cmap[k % len(cmap)] for k in range(n)]
        else:
            raise ValueError("Invalid cmap.")

    # ================================================================
    # 1D panel
    # ================================================================
    def _plot_1d(ax, var, panel_idx):
        local_ylim = ylim

        if isinstance(Y, list) and isinstance(X, list):
            # col_idx = X[0].columns.get_loc(var)
            if not multiYaxis:
                colors = _get_colors_1d(len(X))
                for idx, (x, y) in enumerate(zip(X, Y)):
                    ax.scatter(
                        x.loc[:, var], y.loc[:, var],
                        label=labels[idx] if labels is not None else str(idx),
                        color=colors[idx], s=10, alpha=0.3,
                    )
                ax.legend(markerscale=2.0, handletextpad=0.4,
                          borderpad=0.4, labelspacing=0.3)
                ax.set_ylim(local_ylim)
                ax.set_ylabel(f'$\\psi_{{{var}}}$', fontsize=11)
            else:
                ax_list = []
                for idx, (x, y) in enumerate(zip(X, Y)):
                    if idx == 0:
                        ax_list.append(ax)
                    else:
                        ax_new = ax.twinx()
                        ax_new.spines['right'].set_position(('outward', 60 * (idx - 1)))
                        ax_list.append(ax_new)
                    ax_curr = ax_list[idx]
                    color   = plt.cm.tab10(idx) if cmap is None else _get_colors_1d(len(X))[idx]
                    ax_curr.scatter(
                        x.loc[:, var], y.loc[:, var],
                        label=labels[idx] if labels is not None else str(idx),
                        color=color, s=10, alpha=0.3,
                    )
                    ax_curr.set_ylabel(
                        labels[idx] if labels is not None else str(idx),
                        fontsize=11, color=color,
                    )
                    ax_curr.tick_params(axis='y', labelcolor=color)
                    _ylim = local_ylim[idx] if isinstance(local_ylim, list) else local_ylim
                    ax_curr.set_ylim(_ylim)

            ax.set_xlabel(var, fontsize=11)
            hist_data = X[0].loc[:, var]

        else:
            # col_idx = X.columns.get_loc(var)
            ax.scatter(X.loc[:, var], Y.loc[:, var], s=10, alpha=0.3)
            ax.set_ylim(local_ylim)
            ax.set_ylabel(f'$\\psi_{{{var}}}$', fontsize=11)
            ax.set_xlabel(var, fontsize=11)
            hist_data = X.loc[:, var]

        ax.set_title(var)

        if texts is not None and var in texts:
            ax.text(
                text_pos[0], text_pos[1], texts[var],
                transform=ax.transAxes, va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.8), fontsize=9,
            )

        ax.ticklabel_format(style='sci', axis='x', scilimits=(-2, 3))

        if hist:
            ax_hist = ax.twinx()
            if multiYaxis and isinstance(X, list):
                ax_hist.spines['right'].set_position(('outward', 60 * (len(X) - 1)))
            ax_hist.hist(hist_data, bins=100, alpha=0.3, color='gray', density=True)
            ax_hist.set_ylim(0, ax_hist.get_ylim()[1])
            ax_hist.set_yticks([])

        ax.text(
            -0.08, 1.0, f'({panel_labels[panel_idx]})',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            va='bottom', ha='right',
        )

    # ================================================================
    # 2D panel
    # ================================================================
    def _plot_2d(ax, x, y, z, x_var, y_var, panel_idx, title=None, marginal_x=None, marginal_y=None):
            stat, x_edge, y_edge, _ = binned_statistic_2d(
                x, y, z,
                statistic='mean', bins=bins_2d,
            )
            z_label = _psi_label(x_var, y_var)
            if marginal_x is not None and marginal_y is not None:
                # ── axes layout ────────────────────────────────────────────────
                divider  = make_axes_locatable(ax)
                ax_top   = divider.append_axes('top',    size='22%', pad=0.08, sharex=ax)
                ax_right = divider.append_axes('right',  size='22%', pad=0.08, sharey=ax)
                ax_cb    = divider.append_axes('bottom', size='5%',  pad=0.55)

                # ── heatmap ────────────────────────────────────────────────────
                pcm = ax.pcolormesh(
                    x_edge, y_edge, stat.T,
                    cmap=cmap_2d, shading='auto',
                    vmin=vmin, vmax=vmax,
                )
                plt.colorbar(pcm, cax=ax_cb, orientation='horizontal', label=z_label)
                ax.set_xlabel(x_var, fontsize=11)
                ax.set_ylabel(y_var, fontsize=11)
                ax.ticklabel_format(style='sci', axis='both', scilimits=(-2, 3))
                
                # ── marginal curves ────────────────────────────────────────────
                cm         = mpl.colormaps.get_cmap(cmap_2d)
                color_top  = cm(0.15)    
                color_right= cm(0.85)   
                ax_top.scatter(x, marginal_x, color=color_top, s=5, alpha=0.5)
                ax_right.scatter(marginal_y, y, color=color_right, s=5, alpha=0.5)

                ax_top.set_ylim(0,1.01)
                ax_right.set_xlim(0,1.01)

                # ── top marginal style ─────────────────────────────────────────
                ax_top.set_ylabel(_psi_label(x_var), fontsize=8, labelpad=2)
                ax_top.tick_params(axis='x', labelbottom=False)
                ax_top.tick_params(labelsize=7)
                ax_top.ticklabel_format(style='sci', axis='y', scilimits=(-2, 3))
                ax_top.spines[['right', 'top']].set_visible(False)

                # ── right marginal style ───────────────────────────────────────
                ax_right.set_xlabel(_psi_label(y_var), fontsize=8, labelpad=2)
                ax_right.tick_params(axis='y', labelleft=False)
                ax_right.tick_params(labelsize=7)
                ax_right.ticklabel_format(style='sci', axis='x', scilimits=(-2, 3))
                ax_right.spines[['right', 'top']].set_visible(False)

                # ── panel label and title ────────────────────────────────────────────────
                ax_top.text(
                    -0.12, 1.0, f'({panel_labels[panel_idx]})',
                    transform=ax_top.transAxes, fontsize=11, fontweight='bold',
                    va='bottom', ha='right',
                )
                ax_top.set_title(title)
            else:
                divider = make_axes_locatable(ax)
                ax_cb   = divider.append_axes('bottom', size='5%', pad=0.55)
                pcm = ax.pcolormesh(
                    x_edge, y_edge, stat.T,
                    cmap=cmap_2d, shading='auto',
                    vmin=vmin, vmax=vmax,
                )
                plt.colorbar(pcm, cax=ax_cb, orientation='horizontal', label=z_label)
                ax.set_xlabel(x_var, fontsize=11)
                ax.set_ylabel(y_var, fontsize=11)
                ax.ticklabel_format(style='sci', axis='both', scilimits=(-2, 3))
                ax.set_title(title)

                # ── panel label and title ────────────────────────────────────────────────
                ax.text(
                    -0.12, 1.0, f'({panel_labels[panel_idx]})',
                    transform=ax.transAxes, fontsize=11, fontweight='bold',
                    va='bottom', ha='right',
                )
            
    # ================================================================
    # Build figure
    # ================================================================
    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(
            nrows, ncols,
            squeeze=False,
            figsize=kwargs.get('figsize', (ncols * 4.5, nrows * 4.5)),
        )
        fig.suptitle(plt_title, fontsize=16, y=1.01)
        axes_flat = axes.flatten()

        # 1D panels first, then 2D panels  (or interleave by listing order)
        panel_idx = 0
        for var in vars_1d:
            _plot_1d(axes_flat[panel_idx], var, panel_idx)
            panel_idx += 1

        for (x_var, y_var) in vars_2d:
            if isinstance(Y, list) and isinstance(X, list):
                for (x, y), lbl in zip(zip(X, Y), labels):
                    x_val = x[x_var]
                    y_val = x[y_var]
                    z_val = y[f'{x_var}x{y_var}']
                    
                    if show_marginal and x_var in y.columns and y_var in y.columns:
                        marginal_x = y[x_var]
                        marginal_y = y[y_var]
                    else:
                        marginal_x = None
                        marginal_y = None
                    
                    _plot_2d(axes_flat[panel_idx], x_val, y_val, z_val, x_var, y_var, panel_idx, title=lbl, marginal_x=marginal_x, marginal_y=marginal_y)
                    panel_idx += 1

        for j in range(nvar, len(axes_flat)):
            fig.delaxes(axes_flat[j])

        plt.tight_layout(rect=[0, 0, 1, 0.99])
        if savepath is not None:
            plt.savefig(savepath, bbox_inches='tight', dpi=300)
        plt.show()
        
        
def XGB_CV(X, y, nfolds, model_params):
    y_cv = np.zeros(np.size(y)) + np.nan
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(X, y):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        # Split training set into train and validation
        X_train, X_val, y_train, y_val = train_test_split(X_train_fold, y_train_fold, test_size=0.2,
                                                          random_state=42)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        evallist = [(dval, 'eval')]
        model = xgb.train(model_params, dtrain, num_boost_round=10000, evals=evallist,
                          early_stopping_rounds=20, verbose_eval=False)
        best_iteration = model.best_iteration

        dtrain_fold = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dtest_fold = xgb.DMatrix(X_test_fold)
        bst = xgb.train(model_params, dtrain_fold, num_boost_round=best_iteration)
        y_cv[test_index] = bst.predict(dtest_fold)

    return y_cv


def XGB_addSHAP(X, y, xgb_params):
    xgb_model = train_XGB(X, y, xgb_params)
    explainer = shap.TreeExplainer(xgb_model)
    explanation = explainer(X)
    shap_values = explanation.values
    df_shap_add = pd.DataFrame(shap_values, columns=explanation.feature_names, index=X.index)
    return df_shap_add

def rita_params(base: dict, mono: dict, interaction: list) -> dict:
    """Return a full XGB param dict with the given monotone / interaction overrides."""
    return {**base, "monotone_constraints": mono, "interaction_constraints": interaction}


def relax_constraints(mono_constraints, interaction_constraints, highlycorrelatedVars):
    """
    If a variable has mono_constraint == 2 AND is highly correlated with other variables,
    relax it: set mono_constraint to 0 and allow interaction with its correlated variables.

    Args:
        mono_constraints        (dict):       e.g. {'TA': 2, 'VPD': -1, 'SWC': 1}
        interaction_constraints (list[list]): e.g. [['TA'], ['VPD'], ['SWC']]
        highlycorrelatedVars    (list[list]): e.g. [['TA', 'VPD']]

    Returns:
        new_mono_constraints        (dict)
        new_interaction_constraints (list[list])
    """
    new_mono = mono_constraints.copy()
    new_ic = [g.copy() for g in interaction_constraints]

    unimodal_vars = [var for var, val in mono_constraints.items() if val == 2]

    for var in unimodal_vars:
        new_mono[var] = 0
        if highlycorrelatedVars is not None and len(highlycorrelatedVars)>0:
            correlated_group = next(
                (g for g in highlycorrelatedVars if var in g), None
            )
            if correlated_group is None:
                continue

            groups_to_merge = [g for g in new_ic if any(v in g for v in correlated_group)]
            merged = []
            for g in groups_to_merge:
                new_ic.remove(g)
                for v in g:
                    if v not in merged:
                        merged.append(v)
            new_ic.append(merged)

    return new_mono, new_ic


def _estimate_turning_points(X, shap_norm, mono_constraints):
    """
    For every variable with monotone_constraint == 2 (unimodal),
    estimate its optimal turning point from the first-pass SHAP values.

    Args:
        X               (pd.DataFrame): feature matrix
        shap_norm       (pd.DataFrame): normalised multiplicative SHAP values
        mono_constraints (dict):        e.g. {'TA': 2, 'VPD': -1}

    Returns:
        dict: {var: Topt} for each unimodal variable
    """
    unimodal_vars = [var for var, val in mono_constraints.items() if val == 2]
    if not unimodal_vars:
        return {}

    return {
        var: get_Opt(
            X[var].values,
            shap_norm[var].values,
            method="lowess",
            smooth_param=0.25,
            verbose=False,
        )
        for var in unimodal_vars
    }


def GAM_partial_dependence(gam, X):
    out = []
    for i in range(len(gam.terms)-1): 
        response = gam.partial_dependence(term=i, X=X)
        out.append(response)
    return np.column_stack(out)


def build_RITAgam(X, y, monotone_constraints, interaction_constraints):
    gam_func = []
    for i in interaction_constraints:
        if isinstance(i, str):
            idx = X.columns.get_loc(i)
            gam_func.append(s(idx, constraints=monotone_constraints.get(i, None), n_splines=15))
        elif isinstance(i, list):
            if len(i) == 1:
                idx = X.columns.get_loc(i[0])
                gam_func.append(s(idx, constraints=monotone_constraints.get(i[0], None), n_splines=15))
            else:
                idxs = [X.columns.get_loc(col) for col in i]
                gam_func.append(
                    te(*idxs, constraints=[monotone_constraints.get(c, None) for c in i], n_splines=[10 for c in i])
                )

    formula = reduce(operator.add, gam_func)
    lam = np.logspace(-3, 5, 21)
    RITAgam = GAM(formula, distribution='poisson', link='log').gridsearch(X.values, y, lam=lam)
    phi = GAM_partial_dependence(RITAgam, X)
    psi = np.exp(phi)
    psi_gam = pd.DataFrame(psi, index=X.index, columns=['x'.join(i) for i in interaction_constraints])
    return psi_gam, RITAgam

def build_RITAxgb(X, y, xgb_params, monotone_constraints=None, interaction_constraints=None, highlycorrelatedVars=None):
    """
    Two-stage RITA build:
      Stage 1 — first-pass model to estimate turning point(s) for unimodal variables
               (those with monotone_constraint == 2).
      Stage 2 — final model incorporating the estimated split points.

    Returns:
        shap  (pd.DataFrame): multiplicative SHAP values with SW_INxNDVI interaction term
        Topts (dict):         {var: turning_point} for each unimodal variable
    """
    has_unimodal = (
        monotone_constraints is not None
        and any(v == 2 for v in monotone_constraints.values())
    )

    # Stage 1 — estimate turning points for unimodal variables
    if has_unimodal:
        relaxed_mono, relaxed_ic = relax_constraints(
            monotone_constraints, interaction_constraints, highlycorrelatedVars
        )
        first_pass_params = rita_params(xgb_params, relaxed_mono, relaxed_ic)

        tmp = RITAExplainer(
            X, y, first_pass_params, normaliseX=True,
            config={"shapExplainerParams": {"feature_perturbation": "interventional"}},
        )
        _, shap_norm_tmp = RITAExplainer._max_normalize(tmp.shap_multip, shap_type="multiplicative")
        peaks = _estimate_turning_points(X, shap_norm_tmp, monotone_constraints)
    else:
        peaks = {}

    # Stage 2 — final model with unimodal split points
    final_params = rita_params(xgb_params, monotone_constraints, interaction_constraints)
    rita = RITAExplainer(
        X, y, final_params, normaliseX=True,
        config={
            "unimodal_split_points": peaks,
            # "shapExplainerParams": {"feature_perturbation": "interventional"},
        },
    )

    psi = rita.shap_multip.copy()

    return psi, peaks, rita.model
