"""
MASTER COMPARISON SCRIPT
=========================
Aggregates validation outputs from all 5 parcellation methods and produces
cross-method statistical comparisons, publication-ready tables, and figures.

Prerequisites:
    All 5 per-method validation scripts must have been run first:
        schaefer_validation.py  → Outputs/Method_0_Schaefer/Validation/
        gMSHBM_validation.py    → Outputs/Method_1_MSHBM/Validation/
        agp_validation.py       → Outputs/Method_2_AGP/Validation/
        slic_validation.py      → Outputs/Method_3_SLIC_F/Validation/
                                   Outputs/Method_4_SLIC_C/Validation/

OUTPUTS — all to Outputs/Master_Comparison/:
    Comparison_All_Subjects.xlsx   — full Excel workbook, one sheet per metric
    Comparison_Summary.csv         — grand means and SDs, all metrics × methods
    Comparison_Statistics.csv      — pairwise t-tests + Cohen's d, all metric pairs
    Comparison_PubTable.csv        — publication-ready summary table (LaTeX-friendly)
    Figures/
        Boxplot_{metric}.png       — grouped boxplot across methods per metric
        Radar_Method_Profiles.png  — radar chart of method profiles (z-scored)
        Heatmap_Pairwise_Cohen.png — Cohen's d heatmap for all method pairs
        Scatter_Homog_vs_TRT.png   — homogeneity vs test-retest scatter
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from itertools import combinations

warnings.filterwarnings('ignore')
import config

plt.style.use('seaborn-v0_8-whitegrid')

# =============================================================================
# CONFIGURATION
# =============================================================================

METHODS = {
    'Schaefer':  config.METHOD_0_DIR / 'Validation' / 'Schaefer_Scientific_Metrics.csv',
    'gMSHBM':   config.METHOD_1_DIR / 'Validation' / 'gMSHBM_Scientific_Metrics.csv',
    'AGP':       config.METHOD_2_DIR / 'Validation' / 'AGP_Scientific_Metrics.csv',
    'SLIC_F':    config.METHOD_3_DIR / 'Validation' / 'SLIC_F_Scientific_Metrics.csv',
    'SLIC_C':    config.METHOD_4_DIR / 'Validation' / 'SLIC_C_Scientific_Metrics.csv',
}

# Method display order (group → individual)
METHOD_ORDER = ['Schaefer', 'gMSHBM', 'AGP', 'SLIC_F', 'SLIC_C']

# Colour palette — one per method
METHOD_COLORS = {
    'Schaefer': '#4477AA',
    'gMSHBM':  '#EE6677',
    'AGP':      '#228833',
    'SLIC_F':   '#CCBB44',
    'SLIC_C':   '#AA3377',
}

# Core scientific metrics for cross-method comparison
CORE_METRICS = [
    'Homogeneity_Mean',
    'Homogeneity_SD',
    'Parcel_Size_CV',
    'Parcel_Size_Gini',
    'Contiguity_Pct_Combined',
    'N_Fragmented_LH',
    'N_Fragmented_RH',
    'TestRetest_R',
    'InterSubject_Dice_Mean',
]

# Human-readable labels for figures and tables
METRIC_LABELS = {
    'Homogeneity_Mean':          'Functional Homogeneity (Mean)',
    'Homogeneity_SD':            'Functional Homogeneity (SD)',
    'Parcel_Size_CV':            'Parcel Size CV',
    'Parcel_Size_Gini':          'Parcel Size Gini Coefficient',
    'Contiguity_Pct_Combined':   'Spatial Contiguity (%)',
    'N_Fragmented_LH':           'Fragmented Parcels LH (n)',
    'N_Fragmented_RH':           'Fragmented Parcels RH (n)',
    'TestRetest_R':              'Test-Retest Reliability (r)',
    'InterSubject_Dice_Mean':    'Inter-Subject Dice (Mean)',
}

# Higher-is-better (+1) or lower-is-better (-1)
METRIC_DIRECTION = {
    'Homogeneity_Mean':         +1,
    'Homogeneity_SD':           -1,   # lower = more uniform
    'Parcel_Size_CV':           -1,   # lower = more equal sizes
    'Parcel_Size_Gini':         -1,
    'Contiguity_Pct_Combined':  +1,
    'N_Fragmented_LH':          -1,
    'N_Fragmented_RH':          -1,
    'TestRetest_R':             +1,
    'InterSubject_Dice_Mean':   +1,
}

OUT_DIR = config.OUTPUTS_DIR / 'Master_Comparison'
FIG_DIR = OUT_DIR / 'Figures'


# =============================================================================
# 1. LOAD DATA
# =============================================================================

def load_all_methods():
    """
    Load Scientific_Metrics CSVs for all 5 methods.
    Returns:
        dfs      : dict {method_name: DataFrame}
        missing  : list of method names whose CSV was not found
    """
    dfs = {}
    missing = []
    for name, path in METHODS.items():
        if path.exists():
            df = pd.read_csv(path)
            df['Method'] = name
            dfs[name] = df
            print(f'  [Loaded] {name}: {df.shape[0]} subjects, {df.shape[1]} columns')
        else:
            missing.append(name)
            print(f'  [Missing] {name}: {path}')
    return dfs, missing


# =============================================================================
# 2. MERGE INTO LONG AND WIDE FORMATS
# =============================================================================

def build_long_table(dfs):
    """
    Stack all method dataframes into a single long-format table.
    Returns DataFrame with columns: Subject_ID, Method, <all metrics>
    """
    frames = []
    for name in METHOD_ORDER:
        if name in dfs:
            frames.append(dfs[name])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


def build_wide_table(long_df, metric):
    """
    Pivot long table to wide format for a single metric.
    Rows = subjects, columns = methods.
    """
    available = [m for m in METHOD_ORDER if m in long_df['Method'].values]
    wide = long_df.pivot_table(index='Subject_ID', columns='Method', values=metric)
    return wide[[m for m in available if m in wide.columns]]


# =============================================================================
# 3. SUMMARY STATISTICS TABLE
# =============================================================================

def build_summary_table(long_df):
    """
    Grand means and SDs for every metric × method combination.
    Returns DataFrame: rows = method × metric, columns = mean, std, median, min, max, n
    """
    rows = []
    for method in METHOD_ORDER:
        mdf = long_df[long_df['Method'] == method]
        if mdf.empty:
            continue
        for metric in CORE_METRICS:
            if metric not in mdf.columns:
                continue
            vals = mdf[metric].dropna()
            if vals.empty:
                continue
            rows.append({
                'Method':  method,
                'Metric':  metric,
                'Label':   METRIC_LABELS.get(metric, metric),
                'N':       int(len(vals)),
                'Mean':    round(float(vals.mean()), 4),
                'SD':      round(float(vals.std()), 4),
                'SEM':     round(float(vals.sem()), 4),
                'Median':  round(float(vals.median()), 4),
                'Min':     round(float(vals.min()), 4),
                'Max':     round(float(vals.max()), 4),
            })
    return pd.DataFrame(rows)


# =============================================================================
# 4. PAIRWISE STATISTICAL TESTS
# =============================================================================

def cohens_d(a, b):
    """
    Compute Cohen's d for two paired samples.

    Returns np.nan when std(diff) < 1e-10 (degenerate case: all paired
    differences are identical, e.g. comparing a fixed atlas metric against
    itself or another fixed metric). This is distinct from d=0 (no effect)
    and is flagged as undefined rather than artificially inflated by
    floating-point noise.
    """
    diff = np.array(a) - np.array(b)
    sd = float(np.std(diff, ddof=1))
    if sd < 1e-10:
        return float('nan')
    return float(np.mean(diff) / sd)


def interpret_d(d):
    if np.isnan(d):
        return 'undefined'
    ad = abs(d)
    if ad < 0.2:
        return 'negligible'
    elif ad < 0.5:
        return 'small'
    elif ad < 0.8:
        return 'medium'
    else:
        return 'large'


def run_pairwise_stats(long_df):
    """
    For every metric × method pair, run a paired t-test and compute Cohen's d.

    Uses only subjects present in BOTH methods (matched pairs).

    Returns DataFrame with columns:
        Metric, Method_A, Method_B, N_Pairs,
        Mean_A, Mean_B, Mean_Diff,
        t_stat, p_value, p_fdr, Cohen_d, Effect_Size_Interp
    """
    available_methods = [m for m in METHOD_ORDER if m in long_df['Method'].values]
    rows = []

    for metric in CORE_METRICS:
        if metric not in long_df.columns:
            continue

        for m_a, m_b in combinations(available_methods, 2):
            df_a = long_df[long_df['Method'] == m_a][['Subject_ID', metric]].dropna()
            df_b = long_df[long_df['Method'] == m_b][['Subject_ID', metric]].dropna()

            # Matched subjects only
            common = set(df_a['Subject_ID']) & set(df_b['Subject_ID'])
            if len(common) < 3:
                continue

            vals_a = df_a.set_index('Subject_ID').loc[list(common), metric].values.astype(float)
            vals_b = df_b.set_index('Subject_ID').loc[list(common), metric].values.astype(float)

            t, p = stats.ttest_rel(vals_a, vals_b)
            d = cohens_d(vals_a, vals_b)

            rows.append({
                'Metric':             metric,
                'Label':              METRIC_LABELS.get(metric, metric),
                'Method_A':           m_a,
                'Method_B':           m_b,
                'N_Pairs':            len(common),
                'Mean_A':             round(float(np.mean(vals_a)), 4),
                'SD_A':               round(float(np.std(vals_a)), 4),
                'Mean_B':             round(float(np.mean(vals_b)), 4),
                'SD_B':               round(float(np.std(vals_b)), 4),
                'Mean_Diff':          round(float(np.mean(vals_a - vals_b)), 4),
                'SD_Diff':            round(float(np.std(vals_a - vals_b)), 4),
                't_stat':             round(float(t), 4),
                'p_value':            round(float(p), 6),
                'Cohen_d':            round(d, 4),
                'Effect_Size_Interp': interpret_d(d),
                'Significant_p05':    bool(p < 0.05),
            })

    df_stats = pd.DataFrame(rows)

    # FDR correction (Benjamini-Hochberg) across all tests
    if not df_stats.empty and 'p_value' in df_stats.columns:
        p_vals = df_stats['p_value'].values
        n = len(p_vals)
        order = np.argsort(p_vals)
        fdr = np.empty(n)
        fdr[order] = p_vals[order] * n / (np.arange(n) + 1)
        # Enforce monotonicity
        for i in range(n - 2, -1, -1):
            fdr[order[i]] = min(fdr[order[i]], fdr[order[i + 1]])
        fdr = np.minimum(fdr, 1.0)
        df_stats['p_fdr'] = np.round(fdr, 6)
        df_stats['Significant_FDR05'] = fdr < 0.05

    return df_stats


# =============================================================================
# 5. PUBLICATION-READY SUMMARY TABLE
# =============================================================================

def build_pub_table(summary_df):
    """
    Pivot summary_df into a compact publication table:
    Rows = metrics, Columns = methods (Mean ± SD format).
    """
    rows = []
    metrics_done = []
    for metric in CORE_METRICS:
        sub = summary_df[summary_df['Metric'] == metric]
        if sub.empty:
            continue
        label = METRIC_LABELS.get(metric, metric)
        row = {'Metric': label}
        for method in METHOD_ORDER:
            m_sub = sub[sub['Method'] == method]
            if m_sub.empty:
                row[method] = 'N/A'
            else:
                mean = m_sub['Mean'].values[0]
                sd = m_sub['SD'].values[0]
                row[method] = f'{mean:.3f} ± {sd:.3f}'
        rows.append(row)
        metrics_done.append(metric)

    pub_df = pd.DataFrame(rows)
    pub_df.columns = ['Metric'] + [m for m in METHOD_ORDER if m in pub_df.columns]
    return pub_df


# =============================================================================
# 6. FIGURES
# =============================================================================

def plot_grouped_boxplot(long_df, metric, out_path):
    """Grouped boxplot: one box per method, for a single metric."""
    available = [m for m in METHOD_ORDER if m in long_df['Method'].values]
    data = [long_df[long_df['Method'] == m][metric].dropna().values for m in available]
    colors = [METHOD_COLORS[m] for m in available]

    fig, ax = plt.subplots(figsize=(max(7, len(available) * 1.4), 5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Overlay individual data points (jittered)
    for i, (d, color) in enumerate(zip(data, colors)):
        jitter = np.random.uniform(-0.15, 0.15, len(d))
        ax.scatter(np.full_like(d, i + 1) + jitter, d,
                   color=color, alpha=0.6, s=15, zorder=3)

    ax.set_xticks(range(1, len(available) + 1))
    ax.set_xticklabels(available, fontsize=10, rotation=15, ha='right')
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=11)
    ax.set_title(f'Cross-Method Comparison: {METRIC_LABELS.get(metric, metric)}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_radar(summary_df, out_path):
    """
    Radar (spider) chart of method profiles.
    Metrics are z-scored (direction-adjusted) so that
    'outer = better' for all axes.
    """
    available = [m for m in METHOD_ORDER if m in summary_df['Method'].values]
    metrics_to_use = [m for m in CORE_METRICS if m in summary_df['Metric'].values]
    if not metrics_to_use:
        return

    # Build matrix: rows = methods, cols = metrics
    matrix = np.zeros((len(available), len(metrics_to_use)))
    for j, metric in enumerate(metrics_to_use):
        sub = summary_df[summary_df['Metric'] == metric]
        for i, method in enumerate(available):
            m_sub = sub[sub['Method'] == method]
            matrix[i, j] = m_sub['Mean'].values[0] if not m_sub.empty else 0.0

    # Z-score columns and flip lower-is-better
    for j, metric in enumerate(metrics_to_use):
        col = matrix[:, j]
        if np.std(col) > 0:
            col = (col - np.mean(col)) / np.std(col)
        else:
            col = col - np.mean(col)
        col *= METRIC_DIRECTION.get(metric, 1)
        matrix[:, j] = col

    n_metrics = len(metrics_to_use)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    labels = [METRIC_LABELS.get(m, m).replace('(', '\n(') for m in metrics_to_use]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, method in enumerate(available):
        vals = matrix[i].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=METHOD_COLORS[method], linewidth=2, label=method)
        ax.fill(angles, vals, color=METHOD_COLORS[method], alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title('Method Profiles (z-scored, outer = better)',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_cohens_d_heatmap(stats_df, metric, out_path):
    """
    Cohen's d heatmap for all method pairs for a single metric.
    """
    sub = stats_df[stats_df['Metric'] == metric]
    if sub.empty:
        return
    available = [m for m in METHOD_ORDER if m in sub['Method_A'].values or m in sub['Method_B'].values]
    n = len(available)
    mat = np.zeros((n, n))
    annot = np.full((n, n), '', dtype=object)

    idx = {m: i for i, m in enumerate(available)}
    for _, row in sub.iterrows():
        i, j = idx.get(row['Method_A']), idx.get(row['Method_B'])
        if i is None or j is None:
            continue
        d = row['Cohen_d']
        p = row['p_value']
        sig = '*' if p < 0.05 else ''
        mat[i, j] = d
        mat[j, i] = -d
        annot[i, j] = f'{d:.2f}{sig}'
        annot[j, i] = f'{-d:.2f}{sig}'

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(mat, annot=annot, fmt='', cmap='RdBu_r', center=0,
                vmin=-2, vmax=2, xticklabels=available, yticklabels=available,
                ax=ax, linewidths=0.5, cbar_kws={'label': "Cohen's d"})
    ax.set_title(f'Pairwise Cohen\'s d — {METRIC_LABELS.get(metric, metric)}\n(* = p < 0.05)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_homog_vs_trt(long_df, out_path):
    """
    Scatter plot: Functional Homogeneity vs Test-Retest Reliability,
    coloured by method.
    """
    cols_needed = ['Homogeneity_Mean', 'TestRetest_R', 'Method']
    sub = long_df[cols_needed].dropna()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    for method in METHOD_ORDER:
        m_sub = sub[sub['Method'] == method]
        if m_sub.empty:
            continue
        ax.scatter(m_sub['Homogeneity_Mean'], m_sub['TestRetest_R'],
                   color=METHOD_COLORS[method], label=method, s=55, alpha=0.8, zorder=3)

    ax.set_xlabel('Functional Homogeneity (Mean)', fontsize=11)
    ax.set_ylabel('Test-Retest Reliability (r)', fontsize=11)
    ax.set_title('Homogeneity vs Test-Retest Reliability\n(all subjects, all methods)',
                 fontsize=12, fontweight='bold')
    ax.legend(title='Method', fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_metric_correlation_matrix(long_df, method, out_path):
    """
    Correlation matrix between core metrics for a single method.
    """
    mdf = long_df[long_df['Method'] == method][CORE_METRICS].dropna(how='all', axis=1)
    if mdf.shape[0] < 3 or mdf.shape[1] < 2:
        return
    corr = mdf.corr()
    short_labels = [METRIC_LABELS.get(c, c)[:25] for c in corr.columns]
    fig, ax = plt.subplots(figsize=(8, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                annot=True, fmt='.2f', ax=ax,
                xticklabels=short_labels, yticklabels=short_labels,
                cbar_kws={'label': 'Pearson r'}, linewidths=0.3)
    ax.set_title(f'{method} — Metric Correlation Matrix', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# 7. EXCEL WORKBOOK
# =============================================================================

def save_excel_workbook(long_df, summary_df, stats_df, pub_df, out_path):
    """
    Write a multi-sheet Excel workbook.

    Sheets:
        README            — description of each sheet
        Per_Metric_{name} — all subjects × all methods for each core metric
        Summary           — grand means, SDs, SEM, median, min, max
        Pairwise_Stats    — paired t-tests + Cohen's d
        Pub_Table         — publication-ready 'Mean ± SD' table
        Raw_All           — full long-format table (all metrics, all subjects)
    """
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:

        # --- README sheet ---
        readme = pd.DataFrame({
            'Sheet': [
                'README',
                'Per_Metric_*',
                'Summary',
                'Pairwise_Stats',
                'Pub_Table',
                'Raw_All',
            ],
            'Description': [
                'This file: sheet descriptions',
                'One sheet per core metric. Rows = subjects, Columns = methods. '
                'Suitable for pasting into SPSS/R for further analysis.',
                'Grand means, SDs, SEM, median, min, max for every metric × method.',
                'Pairwise paired t-tests and Cohen\'s d between every method pair '
                'for every metric. Includes FDR-corrected p-values.',
                'Publication-ready table in Mean ± SD format. '
                'Copy directly into Word or use as basis for LaTeX \\table.',
                'Full long-format raw data: every subject × method × all metrics.',
            ]
        })
        readme.to_excel(writer, sheet_name='README', index=False)
        _autofit(writer.sheets['README'])

        # --- Per-metric sheets ---
        for metric in CORE_METRICS:
            if metric not in long_df.columns:
                continue
            wide = build_wide_table(long_df, metric)
            if wide.empty:
                continue
            sheet_name = f'Metric_{metric[:25]}'  # Excel 31-char limit
            wide.to_excel(writer, sheet_name=sheet_name)
            _autofit(writer.sheets[sheet_name])

        # --- Summary sheet ---
        if not summary_df.empty:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            _autofit(writer.sheets['Summary'])

        # --- Pairwise stats sheet ---
        if not stats_df.empty:
            stats_df.to_excel(writer, sheet_name='Pairwise_Stats', index=False)
            _autofit(writer.sheets['Pairwise_Stats'])

        # --- Publication table sheet ---
        if not pub_df.empty:
            pub_df.to_excel(writer, sheet_name='Pub_Table', index=False)
            _autofit(writer.sheets['Pub_Table'])

        # --- Raw data sheet ---
        if not long_df.empty:
            long_df.to_excel(writer, sheet_name='Raw_All', index=False)
            _autofit(writer.sheets['Raw_All'])

    print(f'  [Saved] {out_path.name}')


def _autofit(ws):
    """Auto-fit column widths in an openpyxl worksheet."""
    for col in ws.columns:
        max_len = 0
        col_letter = col[0].column_letter
        for cell in col:
            try:
                cell_len = len(str(cell.value)) if cell.value else 0
                max_len = max(max_len, cell_len)
            except Exception:
                pass
        ws.column_dimensions[col_letter].width = min(max_len + 3, 40)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print('=' * 60)
    print('MASTER COMPARISON — All 5 Methods')
    print('=' * 60)

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    print('\n--- Loading per-method CSVs ---')
    dfs, missing = load_all_methods()
    if not dfs:
        print('ERROR: No method CSVs found. Run per-method validation scripts first.')
        raise SystemExit(1)
    if missing:
        print(f'\nWARNING: {len(missing)} method(s) missing — comparison will be partial:')
        for m in missing:
            print(f'  - {m}')

    # ------------------------------------------------------------------
    # 2. Merge
    # ------------------------------------------------------------------
    print('\n--- Merging data ---')
    long_df = build_long_table(dfs)
    print(f'  Long table: {long_df.shape[0]} rows × {long_df.shape[1]} columns')
    print(f'  Methods present: {long_df["Method"].unique().tolist()}')
    print(f'  Subjects: {sorted(long_df["Subject_ID"].unique().tolist())}')

    # ------------------------------------------------------------------
    # 3. Summary statistics
    # ------------------------------------------------------------------
    print('\n--- Computing summary statistics ---')
    summary_df = build_summary_table(long_df)
    print(f'  Summary: {summary_df.shape[0]} rows')

    # Print preview
    pivot = summary_df.pivot_table(index='Label', columns='Method', values='Mean')
    pivot = pivot[[m for m in METHOD_ORDER if m in pivot.columns]]
    print(f'\nGrand Means per Metric × Method:\n{pivot.round(4).to_string()}')

    # ------------------------------------------------------------------
    # 4. Pairwise statistics
    # ------------------------------------------------------------------
    print('\n--- Running pairwise statistical tests ---')
    stats_df = run_pairwise_stats(long_df)
    n_sig = int(stats_df['Significant_p05'].sum()) if not stats_df.empty else 0
    n_sig_fdr = int(stats_df['Significant_FDR05'].sum()) if 'Significant_FDR05' in stats_df.columns else 0
    print(f'  {len(stats_df)} pairwise tests computed')
    print(f'  Significant at p < 0.05: {n_sig}')
    print(f'  Significant at FDR < 0.05: {n_sig_fdr}')

    # ------------------------------------------------------------------
    # 5. Publication table
    # ------------------------------------------------------------------
    print('\n--- Building publication table ---')
    pub_df = build_pub_table(summary_df)
    print(f'\nPublication Table:\n{pub_df.to_string(index=False)}')

    # ------------------------------------------------------------------
    # 6. Figures
    # ------------------------------------------------------------------
    print('\n--- Generating figures ---')

    # Grouped boxplot per metric
    for metric in CORE_METRICS:
        if metric not in long_df.columns:
            continue
        plot_grouped_boxplot(long_df, metric,
                             FIG_DIR / f'Boxplot_{metric}.png')
    print(f'  [Saved] {len(CORE_METRICS)} metric boxplots')

    # Radar chart
    plot_radar(summary_df, FIG_DIR / 'Radar_Method_Profiles.png')
    print('  [Saved] Radar_Method_Profiles.png')

    # Cohen's d heatmaps for key metrics
    key_metrics = ['Homogeneity_Mean', 'TestRetest_R', 'Contiguity_Pct_Combined',
                   'InterSubject_Dice_Mean']
    for metric in key_metrics:
        if metric in stats_df['Metric'].values:
            plot_cohens_d_heatmap(stats_df, metric,
                                  FIG_DIR / f'CohenD_{metric}.png')
    print(f"  [Saved] {len(key_metrics)} Cohen's d heatmaps")

    # Homogeneity vs test-retest scatter
    plot_homog_vs_trt(long_df, FIG_DIR / 'Scatter_Homog_vs_TRT.png')
    print('  [Saved] Scatter_Homog_vs_TRT.png')

    # Per-method metric correlation matrices
    for method in METHOD_ORDER:
        if method in long_df['Method'].values:
            plot_metric_correlation_matrix(
                long_df, method,
                FIG_DIR / f'MetricCorr_{method}.png')
    print(f'  [Saved] {len(METHOD_ORDER)} metric correlation matrices')

    # ------------------------------------------------------------------
    # 7. Save all tables
    # ------------------------------------------------------------------
    print('\n--- Saving tables ---')

    # Full CSVs
    summary_df.to_csv(OUT_DIR / 'Comparison_Summary.csv', index=False)
    print(f'  [Saved] Comparison_Summary.csv')

    if not stats_df.empty:
        stats_df.to_csv(OUT_DIR / 'Comparison_Statistics.csv', index=False)
        print(f'  [Saved] Comparison_Statistics.csv')

    pub_df.to_csv(OUT_DIR / 'Comparison_PubTable.csv', index=False)
    print(f'  [Saved] Comparison_PubTable.csv')

    long_df.to_csv(OUT_DIR / 'Comparison_Raw_All.csv', index=False)
    print(f'  [Saved] Comparison_Raw_All.csv')

    # Excel workbook (master deliverable)
    save_excel_workbook(
        long_df, summary_df, stats_df, pub_df,
        OUT_DIR / 'Comparison_All_Subjects.xlsx'
    )

    # ------------------------------------------------------------------
    # 8. Print final summary
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('FINAL COMPARISON SUMMARY')
    print('=' * 60)
    print(f'\nMethods compared:  {", ".join(long_df["Method"].unique())}')
    print(f'Subjects per method (approx): {long_df.groupby("Method")["Subject_ID"].count().to_dict()}')
    print(f'\nStatistically significant differences (p < 0.05, uncorrected):')

    if not stats_df.empty:
        sig = stats_df[stats_df['Significant_p05']].copy()
        sig = sig[sig['Cohen_d'].notna()]  # exclude degenerate undefined cases
        sig = sig.sort_values('Cohen_d', key=lambda x: x.abs(), ascending=False)
        for _, row in sig.head(15).iterrows():
            print(f'  {row["Label"][:35]:<35} {row["Method_A"]} vs {row["Method_B"]}'
                  f'  |d|={abs(row["Cohen_d"]):.2f} ({row["Effect_Size_Interp"]})'
                  f'  p={row["p_value"]:.4f}')

    print(f'\nAll outputs saved to: {OUT_DIR}')
    print('--- MASTER COMPARISON COMPLETE ---')