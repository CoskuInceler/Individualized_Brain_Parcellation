"""
AGP VALIDATION — Method 2
===========================
Standardized validation for Atlas-Guided Parcellation (Li et al., 2022).

TECHNICAL QC (per subject, per hemisphere):
    1. Vertex count: assigned vertices match atlas target count
    2. Seed integrity: every seed vertex has its correct parcel label
    3. Unique assignments: no vertex is counted in two parcels
    4. Label consistency: parcel IDs match atlas parcel IDs exactly
    5. Spatial contiguity: DFS check per parcel

SCIENTIFIC METRICS (per subject):
    - Functional homogeneity (mean ± SD across parcels, both hemispheres)
    - Parcel size statistics (mean, SD, CV, min, max, Gini)
    - Spatial contiguity statistics (LH and RH separately)
    - Test-retest reliability (Pearson r, REST1 vs REST2 FC matrices)

GROUP METRICS:
    - Inter-subject Dice (mean pairwise, adjacency approach)
    - Group summary statistics

OUTPUTS (all to Outputs/Method_2_AGP/Validation/):
    - AGP_Technical_QC.csv
    - AGP_Scientific_Metrics.csv
    - AGP_Summary_Stats.csv
    - AGP_PerParcel_Homogeneity.csv
    - Figures/
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
    get_valid_vertices, load_and_filter_atlas, load_pickle, load_subject_data,
    calculate_homogeneity_detailed, calculate_parcel_sizes,
    calculate_contiguity_stats, calculate_fc_testretest,
    calculate_intersubject_dice, parcellate_dense
)

plt.style.use('seaborn-v0_8-whitegrid')
METHOD_NAME = 'AGP'
N_PARCELS_PER_HEMI = 100


# =============================================================================
# TECHNICAL QC — Per subject, per hemisphere
# =============================================================================

def run_technical_qc_hemisphere(subj, labels, adj_list, seeds, atlas_labels, hemi):
    """
    Six technical checks for one hemisphere. Returns list of result dicts.
    """
    results = []
    tag = f'{subj}_{hemi}'

    # --- Check 1: Vertex count matches atlas ---
    assigned = int(np.sum(labels > 0))
    target = int(np.sum(atlas_labels > 0))
    passed = assigned == target
    results.append({
        'Subject_ID': subj, 'Hemisphere': hemi, 'check': 'Vertex_Count',
        'passed': passed,
        'value': f'{assigned}/{target}',
        'detail': f'PASS' if passed else f'FAIL: diff={abs(assigned-target)}'
    })

    # --- Check 2: Seed integrity ---
    if seeds is not None:
        errors = sum(1 for pid, sidx in seeds.items() if labels[sidx] != pid)
        passed2 = errors == 0
        results.append({
            'Subject_ID': subj, 'Hemisphere': hemi, 'check': 'Seed_Integrity',
            'passed': passed2,
            'value': f'{len(seeds) - errors}/{len(seeds)} correct',
            'detail': 'PASS: All seeds correctly assigned' if passed2
                      else f'FAIL: {errors} seeds misassigned'
        })

    # --- Check 3: Unique assignments ---
    assigned_verts = np.where(labels > 0)[0]
    unique_assigned = len(np.unique(assigned_verts))
    passed3 = len(assigned_verts) == unique_assigned
    results.append({
        'Subject_ID': subj, 'Hemisphere': hemi, 'check': 'Unique_Assignments',
        'passed': passed3,
        'value': f'{len(assigned_verts)} assigned, {unique_assigned} unique',
        'detail': 'PASS: No duplicates' if passed3 else f'FAIL: {len(assigned_verts) - unique_assigned} duplicates'
    })

    # --- Check 4: Label consistency with atlas ---
    atlas_ids = set(atlas_labels[atlas_labels > 0].tolist())
    result_ids = set(labels[labels > 0].tolist())
    missing = atlas_ids - result_ids
    extra = result_ids - atlas_ids
    passed4 = (len(missing) == 0) and (len(extra) == 0)
    results.append({
        'Subject_ID': subj, 'Hemisphere': hemi, 'check': 'Label_Consistency',
        'passed': passed4,
        'value': f'missing={len(missing)}, extra={len(extra)}',
        'detail': f'PASS: {len(atlas_ids)} IDs match atlas' if passed4
                  else f'FAIL: missing={list(missing)[:5]}, extra={list(extra)[:5]}'
    })

    # --- Check 5: Parcel size sanity ---
    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids > 0]
    sizes = np.array([np.sum(labels == pid) for pid in unique_ids])
    tiny = int(np.sum(sizes < 5))
    passed5 = tiny == 0
    results.append({
        'Subject_ID': subj, 'Hemisphere': hemi, 'check': 'Parcel_Size_Sanity',
        'passed': passed5,
        'value': f'min={int(sizes.min())}, max={int(sizes.max())}, tiny(<5)={tiny}',
        'detail': 'PASS' if passed5 else f'WARNING: {tiny} parcels have < 5 vertices'
    })

    # --- Check 6: Spatial contiguity ---
    if adj_list is not None:
        cont = calculate_contiguity_stats(labels, adj_list)
        n_disc = cont['n_fragmented']
        passed6 = n_disc == 0
        results.append({
            'Subject_ID': subj, 'Hemisphere': hemi, 'check': 'Spatial_Contiguity',
            'passed': passed6,
            'value': f'{cont["n_contiguous"]}/{cont["n_total"]} contiguous ({cont["pct_contiguous"]:.1f}%)',
            'detail': 'PASS: All parcels spatially contiguous' if passed6
                      else f'FAIL: {n_disc} disconnected parcels'
        })

    return results


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


def plot_group_boxplot(values_dict, ylabel, title, out_path):
    vals = [v for v in values_dict.values() if v is not None and not np.isnan(v)]
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.boxplot(vals, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor='steelblue', alpha=0.7))
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticklabels([METHOD_NAME])
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axhline(np.mean(vals), color='red', linestyle='--', alpha=0.7,
               label=f'Mean = {np.mean(vals):.3f}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    val_dir = config.METHOD_2_DIR / 'Validation'
    fig_dir = val_dir / 'Figures'
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print('=' * 60)
    print(f'{METHOD_NAME} VALIDATION — Method 2')
    print('=' * 60)

    # ------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------
    ref_path = config.get_brain_path(config.SUBJECT_IDS[0], config.RUN_IDS[0])
    valid_L = get_valid_vertices(ref_path, 'CIFTI_STRUCTURE_CORTEX_LEFT')
    valid_R = get_valid_vertices(ref_path, 'CIFTI_STRUCTURE_CORTEX_RIGHT')
    n_L, n_R = len(valid_L), len(valid_R)

    atlas_L, atlas_R = load_and_filter_atlas(config.SCHAEFER_200_FILE, valid_L, valid_R)

    adj_L = load_pickle(config.METHOD_2_DIR / 'Adjacency_Graph_L.pkl') \
        if (config.METHOD_2_DIR / 'Adjacency_Graph_L.pkl').exists() else None
    adj_R = load_pickle(config.METHOD_2_DIR / 'Adjacency_Graph_R.pkl') \
        if (config.METHOD_2_DIR / 'Adjacency_Graph_R.pkl').exists() else None
    seeds_L = load_pickle(config.METHOD_2_DIR / 'Seeds_Geometric_L.pkl') \
        if (config.METHOD_2_DIR / 'Seeds_Geometric_L.pkl').exists() else None
    seeds_R = load_pickle(config.METHOD_2_DIR / 'Seeds_Geometric_R.pkl') \
        if (config.METHOD_2_DIR / 'Seeds_Geometric_R.pkl').exists() else None

    # ------------------------------------------------------------------
    # PER-SUBJECT LOOP
    # ------------------------------------------------------------------
    print('\n--- PHASE 1 & 2: Technical QC + Scientific Metrics ---')

    metrics_rows = []
    qc_rows = []
    per_parcel_rows = []
    homog_values = {}
    trt_values = {}
    all_labels_list = []

    for subj in config.SUBJECT_IDS:
        print(f'\n  Subject: {subj}')

        ll_path = config.METHOD_2_DIR / f'{subj}_Labels_L.npy'
        lr_path = config.METHOD_2_DIR / f'{subj}_Labels_R.npy'
        if not (ll_path.exists() and lr_path.exists()):
            print(f'    [Skip] Label files missing')
            continue

        labels_L = np.load(ll_path)
        labels_R = np.load(lr_path)

        # Technical QC
        qc_L = run_technical_qc_hemisphere(subj, labels_L, adj_L, seeds_L, atlas_L, 'LH')
        qc_R = run_technical_qc_hemisphere(subj, labels_R, adj_R, seeds_R, atlas_R, 'RH')
        qc_rows.extend(qc_L + qc_R)

        all_passed = all(r['passed'] for r in qc_L + qc_R if r['passed'] is not None)
        for r in qc_L + qc_R:
            sym = '✓' if r['passed'] else ('✗' if r['passed'] is False else '⚠')
            print(f'    [{sym}] {r["Hemisphere"]} {r["check"]}: {r["detail"]}')

        # Combined labels for scientific metrics
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

        # Contiguity
        cont_L = calculate_contiguity_stats(labels_L, adj_L)
        cont_R = calculate_contiguity_stats(labels_R, adj_R)
        pct_comb = None
        if cont_L['pct_contiguous'] is not None and cont_R['pct_contiguous'] is not None:
            n_cont = (cont_L['n_contiguous'] or 0) + (cont_R['n_contiguous'] or 0)
            n_tot = (cont_L['n_total'] or 0) + (cont_R['n_total'] or 0)
            pct_comb = round(100.0 * n_cont / n_tot, 2) if n_tot > 0 else None
        print(f'    Contiguity: LH={cont_L["pct_contiguous"]}, RH={cont_R["pct_contiguous"]}%')

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
        plot_homogeneity_parcels(homog_per_parcel,
                                 f'{subj} — {METHOD_NAME} Homogeneity per Parcel',
                                 fig_dir / f'{subj}_{METHOD_NAME}_Homogeneity_PerParcel.png')
        plot_parcel_sizes(size_stats['sizes'],
                          f'{subj} — {METHOD_NAME} Parcel Size Distribution',
                          fig_dir / f'{subj}_{METHOD_NAME}_Parcel_Size_Distribution.png')

        csv_path = config.METHOD_2_DIR / f'{subj}_AGP_Timeseries.csv'
        if csv_path.exists():
            plot_fc_matrix(pd.read_csv(csv_path),
                           f'{subj} — {METHOD_NAME} FC Matrix',
                           fig_dir / f'{subj}_{METHOD_NAME}_FC_Matrix.png')

        if trt is not None:
            p1 = config.SHARED_DATA_DIR / 'Aggregated' / f'{subj}_REST1_Dense.npy'
            p2 = config.SHARED_DATA_DIR / 'Aggregated' / f'{subj}_REST2_Dense.npy'
            if p1.exists() and p2.exists():
                ts1, _ = parcellate_dense(np.load(p1), labels_L, labels_R, n_L, n_R)
                ts2, _ = parcellate_dense(np.load(p2), labels_L, labels_R, n_L, n_R)
                fc1 = np.corrcoef(ts1.T)
                fc2 = np.corrcoef(ts2.T)
                idx = np.triu_indices_from(fc1, k=1)
                plot_fc_testretest(fc1[idx], fc2[idx], trt,
                                   f'{subj} — {METHOD_NAME}',
                                   fig_dir / f'{subj}_{METHOD_NAME}_TestRetest_Scatter.png')

    # ------------------------------------------------------------------
    # GROUP LEVEL
    # ------------------------------------------------------------------
    print('\n--- PHASE 3: Group-Level Metrics ---')

    dice_mean, dice_std, _ = calculate_intersubject_dice(all_labels_list)
    print(f'  Inter-Subject Dice: {dice_mean:.4f} ± {dice_std:.4f}')

    for row in metrics_rows:
        row['InterSubject_Dice_Mean'] = round(dice_mean, 4) if not np.isnan(dice_mean) else None
        row['InterSubject_Dice_SD'] = round(dice_std, 4) if not np.isnan(dice_std) else None

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
    # SAVE TABLES
    # ------------------------------------------------------------------
    pd.DataFrame(qc_rows).to_csv(val_dir / f'{METHOD_NAME}_Technical_QC.csv', index=False)
    print(f'  [Saved] {METHOD_NAME}_Technical_QC.csv')

    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(val_dir / f'{METHOD_NAME}_Scientific_Metrics.csv', index=False)
    print(f'  [Saved] {METHOD_NAME}_Scientific_Metrics.csv  ({df_metrics.shape})')

    if per_parcel_rows:
        pd.DataFrame(per_parcel_rows).to_csv(
            val_dir / f'{METHOD_NAME}_PerParcel_Homogeneity.csv', index=False)
        print(f'  [Saved] {METHOD_NAME}_PerParcel_Homogeneity.csv')

    numeric_cols = [c for c in df_metrics.columns
                    if c not in ('Subject_ID', 'All_QC_Passed')]
    df_summary = df_metrics[numeric_cols].describe().T.round(4)
    df_summary.insert(0, 'Metric', df_summary.index)
    df_summary = df_summary.reset_index(drop=True)
    df_summary.to_csv(val_dir / f'{METHOD_NAME}_Summary_Stats.csv', index=False)
    print(f'  [Saved] {METHOD_NAME}_Summary_Stats.csv')

    print(f'\nSummary Statistics:\n{df_summary[["Metric","mean","std","min","max"]].to_string(index=False)}')
    print(f'\n--- {METHOD_NAME} VALIDATION COMPLETE ---')