"""
SCHAEFER VALIDATION — Method 0
================================
Standardized validation for the Schaefer 200-parcel group atlas.

TECHNICAL QC (atlas-level, computed once):
    1. All 200 parcels present and non-empty
    2. Internal consistency: Parcel 1 vertices correlate with parcel mean
    3. Hemisphere flip check: L-Visual vs R-Visual vs R-Motor

SCIENTIFIC METRICS (per subject):
    - Functional homogeneity (mean ± SD across parcels)
    - Parcel size statistics (mean, SD, CV, min, max, Gini)
      NOTE: Parcel sizes are FIXED for Schaefer (same atlas every subject)
    - Spatial contiguity statistics
      NOTE: Contiguity is FIXED for Schaefer
    - Test-retest reliability (Pearson r, REST1 vs REST2 FC matrices)

GROUP METRICS:
    - Inter-subject Dice = 1.0 by definition (same atlas for all subjects)
    - Homogeneity and test-retest statistics across subjects

OUTPUTS (all to Outputs/Method_0_Schaefer/Validation/):
    - Schaefer_Technical_QC.csv
    - Schaefer_Scientific_Metrics.csv
    - Schaefer_Summary_Stats.csv
    - Schaefer_PerParcel_Homogeneity.csv
    - Figures/  (FC matrix, parcel sizes, homogeneity, test-retest)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import config
from utils import (
    get_valid_vertices, load_and_filter_atlas,
    calculate_homogeneity_detailed, calculate_parcel_sizes,
    calculate_contiguity_stats, calculate_fc_testretest,
    calculate_intersubject_dice, load_pickle
)

plt.style.use('seaborn-v0_8-whitegrid')
METHOD_NAME = 'Schaefer'
N_PARCELS = 200


# =============================================================================
# TECHNICAL QC — Atlas-level (run once, not per subject)
# =============================================================================

def run_technical_qc(atlas_L, atlas_R, adj_L, adj_R):
    """
    Run atlas-level technical checks. Returns list of result dicts.
    These checks verify the atlas itself, not subject-specific data.
    """
    results = []
    atlas_combined = np.concatenate([atlas_L, atlas_R])

    # --- Check 1: All parcels present and non-empty ---
    sizes = {}
    empty = []
    for pid in range(1, N_PARCELS + 1):
        count = int(np.sum(atlas_combined == pid))
        sizes[pid] = count
        if count == 0:
            empty.append(pid)

    passed = len(empty) == 0
    results.append({
        'check': 'All_Parcels_Present',
        'passed': passed,
        'value': f'{N_PARCELS - len(empty)}/{N_PARCELS}',
        'detail': 'PASS: All 200 parcels have data' if passed
                  else f'FAIL: Empty parcels — {empty}'
    })

    # --- Check 2: Parcel size range sanity ---
    sz_vals = list(sizes.values())
    passed2 = min(sz_vals) >= 5
    results.append({
        'check': 'Parcel_Size_Sanity',
        'passed': passed2,
        'value': f'min={min(sz_vals)}, max={max(sz_vals)}',
        'detail': f'PASS' if passed2 else f'FAIL: {sum(1 for s in sz_vals if s < 5)} tiny parcels'
    })

    # --- Check 3: Spatial contiguity of the atlas ---
    if adj_L is not None and adj_R is not None:
        cont_L = calculate_contiguity_stats(atlas_L, adj_L)
        cont_R = calculate_contiguity_stats(atlas_R, adj_R)
        pct = (cont_L['pct_contiguous'] + cont_R['pct_contiguous']) / 2
        passed3 = pct >= 95.0
        results.append({
            'check': 'Atlas_Contiguity',
            'passed': passed3,
            'value': f'LH={cont_L["pct_contiguous"]:.1f}%, RH={cont_R["pct_contiguous"]:.1f}%',
            'detail': f'PASS: {pct:.1f}% contiguous' if passed3 else f'FAIL: {pct:.1f}%'
        })
    else:
        results.append({
            'check': 'Atlas_Contiguity',
            'passed': None,
            'value': 'N/A',
            'detail': 'SKIP: Adjacency graph not available'
        })

    # --- Check 4: LH and RH label ranges are correct ---
    lh_ids = set(np.unique(atlas_L[atlas_L > 0]).tolist())
    rh_ids = set(np.unique(atlas_R[atlas_R > 0]).tolist())
    expected_lh = set(range(1, 101))
    expected_rh = set(range(101, 201))
    passed4 = (lh_ids == expected_lh) and (rh_ids == expected_rh)
    results.append({
        'check': 'Label_Ranges',
        'passed': passed4,
        'value': f'LH IDs: {min(lh_ids)}-{max(lh_ids)}, RH IDs: {min(rh_ids)}-{max(rh_ids)}',
        'detail': 'PASS: LH=1-100, RH=101-200' if passed4 else 'FAIL: Unexpected label IDs'
    })

    return results


def run_subject_technical_qc(subj, parcel_df, atlas_combined, dense_data, n_cortex):
    """Per-subject QC checks using the parcellated timeseries CSV."""
    results = []

    # --- Check: Internal consistency (Parcel 1 vertices vs parcel mean) ---
    p1_idx = np.where(atlas_combined == 1)[0]
    cortex_data = dense_data[:, :n_cortex]
    p1_voxels = cortex_data[:, p1_idx]
    p1_mean = parcel_df.iloc[:, 0].values
    corrs = [float(np.corrcoef(p1_voxels[:, i], p1_mean)[0, 1])
             for i in range(p1_voxels.shape[1])]
    avg_r = float(np.mean(corrs))
    passed = avg_r > 0.4
    results.append({
        'check': 'Internal_Consistency_Parcel1',
        'passed': passed,
        'value': f'{avg_r:.3f}',
        'detail': f'PASS: r={avg_r:.3f}' if passed else f'WARNING: r={avg_r:.3f} (possible misalignment)'
    })

    # --- Check: Hemisphere flip ---
    ts_lv = parcel_df.iloc[:, 0].values      # Parcel 1 — L Visual
    ts_rv = parcel_df.iloc[:, 100].values    # Parcel 101 — R Visual (homologous)
    ts_rm = parcel_df.iloc[:, 107].values    # Parcel 108 — R Motor (unrelated)
    r_homo = float(np.corrcoef(ts_lv, ts_rv)[0, 1])
    r_unrel = float(np.corrcoef(ts_lv, ts_rm)[0, 1])
    passed2 = r_homo > r_unrel
    results.append({
        'check': 'Hemisphere_Flip',
        'passed': passed2,
        'value': f'r_homologous={r_homo:.3f}, r_unrelated={r_unrel:.3f}',
        'detail': 'PASS: Hemispheres correctly aligned' if passed2 else 'FAIL: Possible L/R flip'
    })

    return results


# =============================================================================
# FIGURES
# =============================================================================

def plot_fc_matrix(parcel_df, title, out_path):
    """200×200 FC matrix heatmap."""
    corr = parcel_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, ax=ax, cbar_kws={'label': 'Pearson r'})
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Parcel ID')
    ax.set_ylabel('Parcel ID')
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return corr


def plot_fc_testretest(fc1_flat, fc2_flat, r_val, title, out_path):
    """Scatter plot of REST1 vs REST2 FC values."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(fc1_flat, fc2_flat, s=0.5, alpha=0.3, color='steelblue', rasterized=True)
    lims = [min(fc1_flat.min(), fc2_flat.min()), max(fc1_flat.max(), fc2_flat.max())]
    ax.plot(lims, lims, 'r--', linewidth=1, label='identity')
    ax.set_xlabel('REST1 FC (r)', fontsize=11)
    ax.set_ylabel('REST2 FC (r)', fontsize=11)
    ax.set_title(f'{title}\nTest-Retest r = {r_val:.3f}', fontsize=12, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_homogeneity_parcels(per_parcel_dict, title, out_path):
    """Per-parcel homogeneity bar chart."""
    pids = sorted(per_parcel_dict.keys())
    vals = [per_parcel_dict[p] for p in pids]
    colors = ['#4477AA' if p <= 100 else '#EE6677' for p in pids]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(pids)), vals, color=colors, width=1.0, edgecolor='none')
    ax.axvline(99.5, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax.axhline(np.nanmean(vals), color='orange', linewidth=1.5,
               linestyle='--', label=f'Mean = {np.nanmean(vals):.3f}')
    ax.set_xlabel('Parcel ID', fontsize=11)
    ax.set_ylabel('Homogeneity', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend()
    # Hemisphere labels
    ax.text(50, ax.get_ylim()[0], 'LH', ha='center', va='bottom',
            fontsize=9, color='#4477AA')
    ax.text(150, ax.get_ylim()[0], 'RH', ha='center', va='bottom',
            fontsize=9, color='#EE6677')
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_parcel_sizes(sizes, title, out_path):
    """Parcel size distribution histogram."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sizes, bins=30, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.axvline(np.mean(sizes), color='red', linestyle='--',
               label=f'Mean = {np.mean(sizes):.0f}')
    ax.axvline(np.median(sizes), color='orange', linestyle='--',
               label=f'Median = {np.median(sizes):.0f}')
    ax.set_xlabel('Parcel size (vertices)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    stats_text = (f'N={len(sizes)}  Mean={np.mean(sizes):.0f}  '
                  f'SD={np.std(sizes):.0f}  CV={np.std(sizes)/np.mean(sizes):.3f}')
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=8,
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_group_boxplot(values_dict, ylabel, title, out_path):
    """Boxplot across subjects for a single metric."""
    data = pd.DataFrame({'Value': list(values_dict.values())})
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.boxplot(data['Value'].dropna(), widths=0.5, patch_artist=True,
               boxprops=dict(facecolor='steelblue', alpha=0.7))
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticklabels([METHOD_NAME])
    ax.set_title(title, fontsize=12, fontweight='bold')
    mean_v = data['Value'].dropna().mean()
    ax.axhline(mean_v, color='red', linestyle='--', alpha=0.7,
               label=f'Mean = {mean_v:.3f}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    val_dir = config.METHOD_0_DIR / 'Validation'
    fig_dir = val_dir / 'Figures'
    agg_dir = config.METHOD_0_DIR / 'Aggregated'
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print('=' * 60)
    print(f'SCHAEFER VALIDATION — Method 0')
    print('=' * 60)

    # ------------------------------------------------------------------
    # SETUP: Load shared resources
    # ------------------------------------------------------------------
    ref_path = config.get_brain_path(config.SUBJECT_IDS[0], config.RUN_IDS[0])
    valid_L = get_valid_vertices(ref_path, 'CIFTI_STRUCTURE_CORTEX_LEFT')
    valid_R = get_valid_vertices(ref_path, 'CIFTI_STRUCTURE_CORTEX_RIGHT')
    n_L, n_R = len(valid_L), len(valid_R)

    atlas_L, atlas_R = load_and_filter_atlas(config.SCHAEFER_200_FILE, valid_L, valid_R)
    atlas_combined = np.concatenate([atlas_L, atlas_R])
    n_cortex = len(atlas_combined)

    # Load adjacency graphs (borrowed from AGP output)
    adj_L_path = config.METHOD_2_DIR / 'Adjacency_Graph_L.pkl'
    adj_R_path = config.METHOD_2_DIR / 'Adjacency_Graph_R.pkl'
    adj_L = load_pickle(adj_L_path) if adj_L_path.exists() else None
    adj_R = load_pickle(adj_R_path) if adj_R_path.exists() else None

    # ------------------------------------------------------------------
    # ATLAS-LEVEL TECHNICAL QC (once, not per subject)
    # ------------------------------------------------------------------
    print('\n--- PHASE 1: Atlas-Level Technical QC ---')
    atlas_qc = run_technical_qc(atlas_L, atlas_R, adj_L, adj_R)
    for r in atlas_qc:
        sym = '✓' if r['passed'] else ('✗' if r['passed'] is False else '⚠')
        print(f'  [{sym}] {r["check"]}: {r["detail"]}')

    # Save atlas QC
    pd.DataFrame(atlas_qc).to_csv(val_dir / f'{METHOD_NAME}_Atlas_QC.csv', index=False)

    # ------------------------------------------------------------------
    # ATLAS-LEVEL FIXED METRICS (parcel sizes, contiguity)
    # ------------------------------------------------------------------
    size_stats = calculate_parcel_sizes(atlas_combined)
    cont_L = calculate_contiguity_stats(atlas_L, adj_L)
    cont_R = calculate_contiguity_stats(atlas_R, adj_R)

    # Save atlas parcel size figure (fixed for all subjects)
    plot_parcel_sizes(size_stats['sizes'],
                      f'{METHOD_NAME} — Parcel Size Distribution (Atlas)',
                      fig_dir / f'{METHOD_NAME}_Parcel_Size_Distribution.png')
    print('\n  [Saved] Parcel size figure')

    # ------------------------------------------------------------------
    # PER-SUBJECT: SCIENTIFIC METRICS + SUBJECT-LEVEL QC
    # ------------------------------------------------------------------
    print('\n--- PHASE 2: Per-Subject Scientific Metrics ---')

    metrics_rows = []
    qc_rows = []
    per_parcel_rows = []
    homog_values = {}
    trt_values = {}

    for subj in config.SUBJECT_IDS:
        print(f'\n  Subject: {subj}')

        # Load dense data
        dense_path = config.SHARED_DATA_DIR / 'Aggregated' / f'{subj}_ALL_Dense.npy'
        parcel_path = agg_dir / f'{subj}_ALL_Schaefer200.csv'

        if not dense_path.exists():
            print(f'    [Skip] Dense data missing')
            continue

        dense_all = np.load(dense_path)
        surface_data = dense_all[:, :n_cortex]

        # Homogeneity
        homog_mean, homog_per_parcel = calculate_homogeneity_detailed(
            surface_data, atlas_combined)
        valid_vals = [v for v in homog_per_parcel.values() if not np.isnan(v)]
        homog_sd = float(np.std(valid_vals)) if valid_vals else float('nan')
        homog_values[subj] = homog_mean
        print(f'    Homogeneity: {homog_mean:.4f} ± {homog_sd:.4f}')

        # Test-retest reliability
        trt = calculate_fc_testretest(atlas_L, atlas_R, n_L, n_R, subj)
        trt_values[subj] = trt
        print(f'    Test-Retest r: {trt:.3f}' if trt is not None else '    Test-Retest: N/A')

        # Compute combined contiguity (fixed for atlas — both hemispheres 100%)
        pct_L = cont_L['pct_contiguous']
        pct_R = cont_R['pct_contiguous']
        if pct_L is not None and pct_R is not None:
            n_cont = (cont_L['n_contiguous'] or 0) + (cont_R['n_contiguous'] or 0)
            n_tot = (cont_L['n_total'] or 0) + (cont_R['n_total'] or 0)
            pct_comb = round(100.0 * n_cont / n_tot, 2) if n_tot > 0 else None
        else:
            pct_comb = None

        # Collect metrics row
        row = {
            'Subject_ID': subj,
            'All_QC_Passed': None,  # Updated below after subject QC runs
            'Homogeneity_Mean': round(homog_mean, 4),
            'Homogeneity_SD': round(homog_sd, 4),
            'Parcel_Size_Mean': round(size_stats['mean'], 2),
            'Parcel_Size_SD': round(size_stats['std'], 2),
            'Parcel_Size_CV': round(size_stats['cv'], 4),
            'Parcel_Size_Min': size_stats['min'],
            'Parcel_Size_Max': size_stats['max'],
            'Parcel_Size_Gini': round(size_stats['gini'], 4),
            'Contiguity_Pct_LH': round(pct_L, 2) if pct_L is not None else None,
            'Contiguity_Pct_RH': round(pct_R, 2) if pct_R is not None else None,
            'Contiguity_Pct_Combined': pct_comb,
            'N_Fragmented_LH': cont_L['n_fragmented'],
            'N_Fragmented_RH': cont_R['n_fragmented'],
            'Mean_Components_LH': round(cont_L['mean_components'], 3) if cont_L['mean_components'] is not None else None,
            'Mean_Components_RH': round(cont_R['mean_components'], 3) if cont_R['mean_components'] is not None else None,
            'TestRetest_R': round(trt, 4) if trt is not None else None,
            'InterSubject_Dice_Mean': 1.0,   # Trivially 1.0 for fixed atlas
            'InterSubject_Dice_SD': 0.0,
        }
        metrics_rows.append(row)

        # Per-parcel homogeneity (for export)
        for pid, val in homog_per_parcel.items():
            per_parcel_rows.append({
                'Subject_ID': subj, 'Parcel_ID': pid,
                'Hemisphere': 'LH' if pid <= 100 else 'RH',
                'Homogeneity': round(val, 4) if not np.isnan(val) else None
            })

        # Subject-level QC (needs parcel CSV and dense data)
        if parcel_path.exists():
            parcel_df = pd.read_csv(parcel_path)
            subj_qc = run_subject_technical_qc(subj, parcel_df, atlas_combined, dense_all, n_cortex)
            for r in subj_qc:
                r['Subject_ID'] = subj
                qc_rows.append(r)
                sym = '✓' if r['passed'] else '✗'
                print(f'    [{sym}] {r["check"]}: {r["detail"]}')
            # Update All_QC_Passed in the already-appended row
            all_passed = all(r['passed'] for r in subj_qc if r['passed'] is not None)
            metrics_rows[-1]['All_QC_Passed'] = all_passed

        # Per-subject figures
        # FC matrix
        if parcel_path.exists():
            parcel_df = pd.read_csv(parcel_path)
            plot_fc_matrix(parcel_df, f'{subj} — {METHOD_NAME} FC Matrix',
                           fig_dir / f'{subj}_{METHOD_NAME}_FC_Matrix.png')

        # Homogeneity per parcel
        plot_homogeneity_parcels(homog_per_parcel,
                                 f'{subj} — {METHOD_NAME} Homogeneity per Parcel',
                                 fig_dir / f'{subj}_{METHOD_NAME}_Homogeneity_PerParcel.png')

        # Test-retest scatter
        if trt is not None:
            p1 = config.SHARED_DATA_DIR / 'Aggregated' / f'{subj}_REST1_Dense.npy'
            p2 = config.SHARED_DATA_DIR / 'Aggregated' / f'{subj}_REST2_Dense.npy'
            if p1.exists() and p2.exists():
                from utils import parcellate_dense
                ts1, _ = parcellate_dense(np.load(p1), atlas_L, atlas_R, n_L, n_R)
                ts2, _ = parcellate_dense(np.load(p2), atlas_L, atlas_R, n_L, n_R)
                fc1 = np.corrcoef(ts1.T)
                fc2 = np.corrcoef(ts2.T)
                idx = np.triu_indices_from(fc1, k=1)
                plot_fc_testretest(fc1[idx], fc2[idx], trt,
                                   f'{subj} — {METHOD_NAME}',
                                   fig_dir / f'{subj}_{METHOD_NAME}_TestRetest_Scatter.png')

    # ------------------------------------------------------------------
    # GROUP-LEVEL FIGURES
    # ------------------------------------------------------------------
    print('\n--- PHASE 3: Group-Level Outputs ---')

    if homog_values:
        plot_group_boxplot(homog_values, 'Homogeneity',
                           f'{METHOD_NAME} — Homogeneity Across Subjects',
                           fig_dir / f'{METHOD_NAME}_Group_Homogeneity_Boxplot.png')

    if trt_values:
        plot_group_boxplot({k: v for k, v in trt_values.items() if v is not None},
                           'Test-Retest r',
                           f'{METHOD_NAME} — Test-Retest Reliability Across Subjects',
                           fig_dir / f'{METHOD_NAME}_Group_TestRetest_Boxplot.png')

    # ------------------------------------------------------------------
    # SAVE ALL TABLES
    # ------------------------------------------------------------------

    # Scientific metrics CSV
    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(val_dir / f'{METHOD_NAME}_Scientific_Metrics.csv', index=False)
    print(f'  [Saved] {METHOD_NAME}_Scientific_Metrics.csv  ({df_metrics.shape})')

    # Subject-level QC CSV
    if qc_rows:
        df_qc = pd.DataFrame(qc_rows)
        df_qc.to_csv(val_dir / f'{METHOD_NAME}_Subject_QC.csv', index=False)
        print(f'  [Saved] {METHOD_NAME}_Subject_QC.csv')

    # Per-parcel homogeneity
    if per_parcel_rows:
        df_pp = pd.DataFrame(per_parcel_rows)
        df_pp.to_csv(val_dir / f'{METHOD_NAME}_PerParcel_Homogeneity.csv', index=False)
        print(f'  [Saved] {METHOD_NAME}_PerParcel_Homogeneity.csv')

    # Summary statistics
    numeric_cols = [c for c in df_metrics.columns if c not in ('Subject_ID', 'All_QC_Passed')]
    df_summary = df_metrics[numeric_cols].describe().T.round(4)
    df_summary.insert(0, 'Metric', df_summary.index)
    df_summary = df_summary.reset_index(drop=True)
    df_summary.to_csv(val_dir / f'{METHOD_NAME}_Summary_Stats.csv', index=False)
    print(f'  [Saved] {METHOD_NAME}_Summary_Stats.csv')

    print(f'\nSummary Statistics:\n{df_summary[["Metric","mean","std","min","max"]].to_string(index=False)}')
    print(f'\n--- {METHOD_NAME} VALIDATION COMPLETE ---')