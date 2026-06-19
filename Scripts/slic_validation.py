"""
SLIC VALIDATION — Methods 3 & 4
==================================
Standardized validation for SLIC-F (spatial+functional) and SLIC-C
(functional only) parcellations (Wang et al., 2016).

Handles both methods in a single script since they share the same label
format and validation logic.

TECHNICAL QC (per subject, per hemisphere, per method):
    1. All vertices assigned (no label = -1 remaining)
    2. Correct parcel count (100 per hemisphere)
    3. No empty parcels
    4. Parcel size distribution (warnings for extreme sizes)
    5. Spatial contiguity:
       - SLIC-F: expects mostly contiguous (< 25% fragmented = acceptable)
       - SLIC-C: fragmentation expected and quantified, not flagged as failure

SCIENTIFIC METRICS (per subject, per method):
    - Functional homogeneity (mean ± SD across parcels, both hemispheres)
    - Parcel size statistics (mean, SD, CV, min, max, Gini)
    - Spatial contiguity statistics (LH and RH separately)
    - Test-retest reliability (Pearson r, REST1 vs REST2 FC matrices)

GROUP METRICS:
    - Inter-subject Dice (mean pairwise, adjacency approach)
    - SLIC-F vs SLIC-C fragmentation comparison

OUTPUTS:
    Outputs/Method_3_SLIC_F/Validation/  — SLIC_F_*.csv + Figures/
    Outputs/Method_4_SLIC_C/Validation/  — SLIC_C_*.csv + Figures/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import config
from utils import (
    get_valid_vertices, load_and_filter_atlas, load_pickle,
    calculate_homogeneity_detailed, calculate_parcel_sizes,
    calculate_contiguity_stats, calculate_fc_testretest,
    calculate_intersubject_dice, parcellate_dense
)

plt.style.use('seaborn-v0_8-whitegrid')
N_PARCELS_PER_HEMI = 100


# =============================================================================
# TECHNICAL QC — Per subject, per hemisphere
# =============================================================================

def run_technical_qc_hemisphere(subj, labels, adj_list, hemi, method_name):
    """
    Five technical checks for one hemisphere. Returns list of result dicts.

    For SLIC, raw labels are 0-indexed (0 to n_clusters-1) so label -1
    indicates unassigned vertices.
    """
    results = []
    n_expected = N_PARCELS_PER_HEMI

    # --- Check 1: All vertices assigned ---
    n_unassigned = int(np.sum(labels < 0))
    passed = n_unassigned == 0
    results.append({
        'Subject_ID': subj, 'Hemisphere': hemi, 'check': 'All_Assigned',
        'passed': passed,
        'value': f'{len(labels) - n_unassigned}/{len(labels)}',
        'detail': 'PASS: All vertices assigned' if passed
                  else f'FAIL: {n_unassigned} unassigned vertices'
    })

    # --- Check 2: Correct parcel count ---
    unique_labels = np.unique(labels[labels >= 0])
    n_parcels = len(unique_labels)
    passed2 = n_parcels == n_expected
    results.append({
        'Subject_ID': subj, 'Hemisphere': hemi, 'check': 'Parcel_Count',
        'passed': passed2,
        'value': f'{n_parcels}/{n_expected}',
        'detail': f'PASS: {n_parcels} parcels' if passed2
                  else f'FAIL: Expected {n_expected}, got {n_parcels}'
    })

    # --- Check 3: No empty parcels ---
    sizes = np.array([np.sum(labels == pid) for pid in range(n_expected)])
    n_empty = int(np.sum(sizes == 0))
    passed3 = n_empty == 0
    results.append({
        'Subject_ID': subj, 'Hemisphere': hemi, 'check': 'No_Empty_Parcels',
        'passed': passed3,
        'value': f'empty={n_empty}',
        'detail': 'PASS: No empty parcels' if passed3
                  else f'FAIL: {n_empty} empty parcels'
    })

    # --- Check 4: Parcel size sanity ---
    valid_sizes = sizes[sizes > 0]
    mean_size = float(np.mean(valid_sizes)) if len(valid_sizes) > 0 else 0
    tiny = int(np.sum(valid_sizes < 5))
    huge = int(np.sum(valid_sizes > 3 * mean_size)) if mean_size > 0 else 0
    passed4 = (tiny == 0) and (huge <= 5)
    results.append({
        'Subject_ID': subj, 'Hemisphere': hemi, 'check': 'Parcel_Size_Sanity',
        'passed': passed4,
        'value': f'mean={mean_size:.0f}, tiny(<5)={tiny}, huge(>3x)={huge}',
        'detail': 'PASS' if passed4 else f'WARNING: {tiny} tiny, {huge} oversized'
    })

    # --- Check 5: Spatial contiguity ---
    if adj_list is not None:
        # Shift labels to be > 0 for contiguity function
        labels_shifted = labels + 1
        labels_shifted[labels < 0] = 0
        cont = calculate_contiguity_stats(labels_shifted, adj_list)
        pct = cont['pct_contiguous']
        n_frag = cont['n_fragmented']

        if method_name == 'SLIC_F':
            # Expect mostly contiguous
            passed5 = pct is not None and pct >= 75.0
            detail = (f'PASS: {pct:.1f}% contiguous ({n_frag} fragmented)' if passed5
                      else f'FAIL: Only {pct:.1f}% contiguous ({n_frag} fragmented)')
        else:  # SLIC_C — fragmentation expected
            passed5 = True  # Not a failure for SLIC-C
            detail = f'INFO: {pct:.1f}% contiguous ({n_frag} fragmented) — expected for SLIC-C'

        results.append({
            'Subject_ID': subj, 'Hemisphere': hemi, 'check': 'Spatial_Contiguity',
            'passed': passed5,
            'value': f'{cont["n_contiguous"]}/{cont["n_total"]} ({pct:.1f}%)',
            'detail': detail
        })
        return results, cont
    else:
        results.append({
            'Subject_ID': subj, 'Hemisphere': hemi, 'check': 'Spatial_Contiguity',
            'passed': None,
            'value': 'N/A',
            'detail': 'SKIP: Adjacency graph not available'
        })
        return results, None


# =============================================================================
# FIGURES
# =============================================================================

def plot_fc_matrix(parcel_df, title, out_path):
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
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(fc1_flat, fc2_flat, s=0.5, alpha=0.3, color='steelblue', rasterized=True)
    lims = [min(fc1_flat.min(), fc2_flat.min()), max(fc1_flat.max(), fc2_flat.max())]
    ax.plot(lims, lims, 'r--', linewidth=1)
    ax.set_xlabel('REST1 FC (r)', fontsize=11)
    ax.set_ylabel('REST2 FC (r)', fontsize=11)
    ax.set_title(f'{title}\nTest-Retest r = {r_val:.3f}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_homogeneity_parcels(per_parcel_dict, title, out_path):
    pids = sorted(per_parcel_dict.keys())
    vals = [per_parcel_dict[p] for p in pids]
    colors = ['#4477AA' if p <= 100 else '#EE6677' for p in pids]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(pids)), vals, color=colors, width=1.0, edgecolor='none')
    ax.axvline(99.5, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax.axhline(np.nanmean(vals), color='orange', linewidth=1.5, linestyle='--',
               label=f'Mean = {np.nanmean(vals):.3f}')
    ax.set_xlabel('Parcel ID', fontsize=11)
    ax.set_ylabel('Homogeneity', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_parcel_sizes(sizes, title, out_path):
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
            ha='right', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_group_boxplot(values_dict, ylabel, title, method_name, out_path):
    vals = [v for v in values_dict.values() if v is not None and not np.isnan(v)]
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.boxplot(vals, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor='steelblue', alpha=0.7))
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticklabels([method_name])
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axhline(np.mean(vals), color='red', linestyle='--', alpha=0.7,
               label=f'Mean = {np.mean(vals):.3f}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# PROCESS ONE METHOD
# =============================================================================

def process_method(method_name, method_dir, valid_L, valid_R, n_L, n_R,
                   atlas_L, atlas_R, adj_L, adj_R):
    """Run the full validation for one SLIC variant."""

    val_dir = method_dir / 'Validation'
    fig_dir = val_dir / 'Figures'
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print(f'\n{"=" * 60}')
    print(f'{method_name} VALIDATION')
    print(f'{"=" * 60}')

    metrics_rows = []
    qc_rows = []
    per_parcel_rows = []
    homog_values = {}
    trt_values = {}
    all_labels_list = []

    for subj in config.SUBJECT_IDS:
        print(f'\n  Subject: {subj}')

        ll_path = method_dir / f'{subj}_Labels_L.npy'
        lr_path = method_dir / f'{subj}_Labels_R.npy'
        if not (ll_path.exists() and lr_path.exists()):
            print(f'    [Skip] Label files missing')
            continue

        raw_L = np.load(ll_path)   # 0 to 99
        raw_R = np.load(lr_path)   # 0 to 99

        # Technical QC (on raw 0-indexed labels)
        qc_L, cont_stats_L = run_technical_qc_hemisphere(subj, raw_L, adj_L, 'LH', method_name)
        qc_R, cont_stats_R = run_technical_qc_hemisphere(subj, raw_R, adj_R, 'RH', method_name)
        qc_rows.extend(qc_L + qc_R)

        all_passed = all(r['passed'] for r in qc_L + qc_R
                         if r['passed'] is not None and
                         not (method_name == 'SLIC_C' and r['check'] == 'Spatial_Contiguity'))
        for r in qc_L + qc_R:
            sym = '✓' if r['passed'] else ('✗' if r['passed'] is False else '⚠')
            print(f'    [{sym}] {r["Hemisphere"]} {r["check"]}: {r["detail"]}')

        # Shift labels to valid-vertex space with unique IDs: LH=1-100, RH=101-200
        labels_L = raw_L + 1       # 1-100
        labels_R = raw_R + 101     # 101-200
        # Handle unassigned (-1 → 0)
        labels_L[raw_L < 0] = 0
        labels_R[raw_R < 0] = 0

        combined_labels = np.concatenate([labels_L, labels_R])
        all_labels_list.append(combined_labels.copy())

        # Load dense data
        dense_path = config.SHARED_DATA_DIR / 'Aggregated' / f'{subj}_ALL_Dense.npy'
        if not dense_path.exists():
            print(f'    [Skip] Dense data missing')
            continue

        dense_all = np.load(dense_path)
        surface_data = dense_all[:, :n_L + n_R]

        # Homogeneity
        homog_mean, homog_per_parcel = calculate_homogeneity_detailed(
            surface_data, combined_labels)
        valid_vals = [v for v in homog_per_parcel.values() if not np.isnan(v)]
        homog_sd = float(np.std(valid_vals)) if valid_vals else float('nan')
        homog_values[subj] = homog_mean
        print(f'    Homogeneity: {homog_mean:.4f} ± {homog_sd:.4f}')

        # Parcel sizes
        size_stats = calculate_parcel_sizes(combined_labels)

        # Contiguity (use shifted labels since our function needs > 0 for valid)
        cont_L = calculate_contiguity_stats(labels_L, adj_L)
        cont_R = calculate_contiguity_stats(labels_R, adj_R)
        pct_comb = None
        if cont_L['pct_contiguous'] is not None and cont_R['pct_contiguous'] is not None:
            n_cont = (cont_L['n_contiguous'] or 0) + (cont_R['n_contiguous'] or 0)
            n_tot = (cont_L['n_total'] or 0) + (cont_R['n_total'] or 0)
            pct_comb = round(100.0 * n_cont / n_tot, 2) if n_tot > 0 else None
        print(f'    Contiguity: LH={cont_L["pct_contiguous"]:.1f}%, RH={cont_R["pct_contiguous"]:.1f}%')

        # Test-retest
        trt = calculate_fc_testretest(labels_L, labels_R, n_L, n_R, subj)
        trt_values[subj] = trt
        print(f'    Test-Retest r: {trt:.3f}' if trt is not None else '    Test-Retest: N/A')

        row = {
            'Subject_ID': subj,
            'All_QC_Passed': all_passed,
            'Homogeneity_Mean': round(homog_mean, 4),
            'Homogeneity_SD': round(homog_sd, 4),
            'Parcel_Size_Mean': round(size_stats['mean'], 2),
            'Parcel_Size_SD': round(size_stats['std'], 2),
            'Parcel_Size_CV': round(size_stats['cv'], 4),
            'Parcel_Size_Min': size_stats['min'],
            'Parcel_Size_Max': size_stats['max'],
            'Parcel_Size_Gini': round(size_stats['gini'], 4),
            'Contiguity_Pct_LH': round(cont_L['pct_contiguous'], 2) if cont_L['pct_contiguous'] is not None else None,
            'Contiguity_Pct_RH': round(cont_R['pct_contiguous'], 2) if cont_R['pct_contiguous'] is not None else None,
            'Contiguity_Pct_Combined': pct_comb,
            'N_Fragmented_LH': cont_L['n_fragmented'],
            'N_Fragmented_RH': cont_R['n_fragmented'],
            'Mean_Components_LH': round(cont_L['mean_components'], 3) if cont_L['mean_components'] is not None else None,
            'Mean_Components_RH': round(cont_R['mean_components'], 3) if cont_R['mean_components'] is not None else None,
            'TestRetest_R': round(trt, 4) if trt is not None else None,
        }
        metrics_rows.append(row)

        for pid, val in homog_per_parcel.items():
            per_parcel_rows.append({
                'Subject_ID': subj, 'Parcel_ID': pid,
                'Hemisphere': 'LH' if pid <= 100 else 'RH',
                'Homogeneity': round(val, 4) if not np.isnan(val) else None
            })

        # Figures
        plot_homogeneity_parcels(
            homog_per_parcel,
            f'{subj} — {method_name} Homogeneity per Parcel',
            fig_dir / f'{subj}_{method_name}_Homogeneity_PerParcel.png')
        plot_parcel_sizes(
            size_stats['sizes'],
            f'{subj} — {method_name} Parcel Size Distribution',
            fig_dir / f'{subj}_{method_name}_Parcel_Size_Distribution.png')

        csv_path = method_dir / f'{subj}_{method_name}_Timeseries.csv'
        if csv_path.exists():
            plot_fc_matrix(pd.read_csv(csv_path),
                           f'{subj} — {method_name} FC Matrix',
                           fig_dir / f'{subj}_{method_name}_FC_Matrix.png')

        if trt is not None:
            p1 = config.SHARED_DATA_DIR / 'Aggregated' / f'{subj}_REST1_Dense.npy'
            p2 = config.SHARED_DATA_DIR / 'Aggregated' / f'{subj}_REST2_Dense.npy'
            if p1.exists() and p2.exists():
                ts1, _ = parcellate_dense(np.load(p1), labels_L, labels_R, n_L, n_R)
                ts2, _ = parcellate_dense(np.load(p2), labels_L, labels_R, n_L, n_R)
                fc1 = np.corrcoef(ts1.T)
                fc2 = np.corrcoef(ts2.T)
                idx2 = np.triu_indices_from(fc1, k=1)
                plot_fc_testretest(fc1[idx2], fc2[idx2], trt,
                                   f'{subj} — {method_name}',
                                   fig_dir / f'{subj}_{method_name}_TestRetest_Scatter.png')

    # Group level
    print('\n--- Group-Level Metrics ---')
    dice_mean, dice_std, _ = calculate_intersubject_dice(all_labels_list)
    print(f'  Inter-Subject Dice: {dice_mean:.4f} ± {dice_std:.4f}')

    for row in metrics_rows:
        row['InterSubject_Dice_Mean'] = round(dice_mean, 4) if not np.isnan(dice_mean) else None
        row['InterSubject_Dice_SD'] = round(dice_std, 4) if not np.isnan(dice_std) else None

    if homog_values:
        plot_group_boxplot(homog_values, 'Homogeneity',
                           f'{method_name} — Homogeneity Across Subjects',
                           method_name,
                           fig_dir / f'{method_name}_Group_Homogeneity_Boxplot.png')
    if trt_values:
        plot_group_boxplot({k: v for k, v in trt_values.items() if v is not None},
                           'Test-Retest r',
                           f'{method_name} — Test-Retest Reliability Across Subjects',
                           method_name,
                           fig_dir / f'{method_name}_Group_TestRetest_Boxplot.png')

    # Save tables
    if qc_rows:
        pd.DataFrame(qc_rows).to_csv(val_dir / f'{method_name}_Technical_QC.csv', index=False)
        print(f'  [Saved] {method_name}_Technical_QC.csv')

    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(val_dir / f'{method_name}_Scientific_Metrics.csv', index=False)
    print(f'  [Saved] {method_name}_Scientific_Metrics.csv  ({df_metrics.shape})')

    if per_parcel_rows:
        pd.DataFrame(per_parcel_rows).to_csv(
            val_dir / f'{method_name}_PerParcel_Homogeneity.csv', index=False)
        print(f'  [Saved] {method_name}_PerParcel_Homogeneity.csv')

    if not df_metrics.empty:
        numeric_cols = [c for c in df_metrics.columns
                        if c not in ('Subject_ID', 'All_QC_Passed')]
        df_summary = df_metrics[numeric_cols].describe().T.round(4)
        df_summary.insert(0, 'Metric', df_summary.index)
        df_summary = df_summary.reset_index(drop=True)
        df_summary.to_csv(val_dir / f'{method_name}_Summary_Stats.csv', index=False)
        print(f'  [Saved] {method_name}_Summary_Stats.csv')
        print(f'\nSummary:\n{df_summary[["Metric","mean","std","min","max"]].to_string(index=False)}')

    return df_metrics


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print('=' * 60)
    print('SLIC VALIDATION — Methods 3 & 4')
    print('=' * 60)

    # Setup
    ref_path = config.get_brain_path(config.SUBJECT_IDS[0], config.RUN_IDS[0])
    valid_L = get_valid_vertices(ref_path, 'CIFTI_STRUCTURE_CORTEX_LEFT')
    valid_R = get_valid_vertices(ref_path, 'CIFTI_STRUCTURE_CORTEX_RIGHT')
    n_L, n_R = len(valid_L), len(valid_R)

    atlas_L, atlas_R = load_and_filter_atlas(config.SCHAEFER_200_FILE, valid_L, valid_R)

    adj_L = load_pickle(config.METHOD_2_DIR / 'Adjacency_Graph_L.pkl') \
        if (config.METHOD_2_DIR / 'Adjacency_Graph_L.pkl').exists() else None
    adj_R = load_pickle(config.METHOD_2_DIR / 'Adjacency_Graph_R.pkl') \
        if (config.METHOD_2_DIR / 'Adjacency_Graph_R.pkl').exists() else None

    if adj_L is not None:
        print('\n✓ Adjacency graphs loaded — contiguity tests enabled')
    else:
        print('\n⚠ Adjacency graphs not found — contiguity tests skipped')

    methods = [
        ('SLIC_F', config.METHOD_3_DIR),
        ('SLIC_C', config.METHOD_4_DIR),
    ]

    results = {}
    for method_name, method_dir in methods:
        df = process_method(
            method_name, method_dir,
            valid_L, valid_R, n_L, n_R,
            atlas_L, atlas_R, adj_L, adj_R
        )
        results[method_name] = df

    # ------------------------------------------------------------------
    # SLIC-F vs SLIC-C fragmentation comparison (cross-method summary)
    # ------------------------------------------------------------------
    print('\n\n--- SLIC-F vs SLIC-C Cross-Method Summary ---')

    for key in ['Contiguity_Pct_Combined', 'N_Fragmented_LH', 'N_Fragmented_RH',
                'Homogeneity_Mean', 'TestRetest_R', 'InterSubject_Dice_Mean']:
        for mn in ['SLIC_F', 'SLIC_C']:
            if mn in results and key in results[mn].columns:
                vals = results[mn][key].dropna()
                if len(vals) > 0:
                    print(f'  {mn} {key}: {vals.mean():.4f} ± {vals.std():.4f}')

    print('\n--- SLIC VALIDATION COMPLETE ---')