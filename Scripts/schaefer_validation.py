"""
SCHAEFER VALIDATION (Level 0)
=============================
Per-subject validation metrics for the Schaefer group atlas.
Computed here (within-subject):
    homogeneity      : REST1, REST2, and ALL
    contiguity       : fixed for a group atlas, computed once as a check
    fc_test_retest   : REST1 vs REST2 FC patterns
Inter-subject Dice is a group-level metric and is handled separately
by the aggregation step.
Output: Outputs/Method_0_Schaefer/Validation/{subject}.csv
"""

import argparse
import time
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import config
import utils
import metrics

METHOD = "Schaefer"


def load_atlas():
    """Schaefer labels aligned to data columns, split by hemisphere."""
    ref = config.get_brain_path(config.SUBJECT_IDS[0], config.RUN_IDS[0])
    vL = utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_LEFT")
    vR = utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_RIGHT")
    aL, aR = utils.load_and_filter_atlas(config.SCHAEFER_200_FILE, vL, vR)
    return aL, aR


def validate_subject(subject_id, atlas_L, atlas_R):
    """Compute all within-subject metrics for one subject."""
    atlas = np.concatenate([atlas_L, atlas_R])
    n_cortex = len(atlas)

    cleaned = config.OUTPUTS_DIR / "Cleaned"
    ts1 = np.load(cleaned / f"{subject_id}_REST1.npy")[:, :n_cortex]
    ts2 = np.load(cleaned / f"{subject_id}_REST2.npy")[:, :n_cortex]
    ts_all = np.concatenate([ts1, ts2], axis=0)

    h1, _ = metrics.homogeneity(ts1, atlas)
    h2, _ = metrics.homogeneity(ts2, atlas)
    ha, _ = metrics.homogeneity(ts_all, atlas)

    trt = metrics.fc_test_retest(ts1, ts2, atlas)

    cL = metrics.contiguity(
        atlas_L, load_npz(config.INPUTS_DIR / "adjacency_L.npz")
    )
    cR = metrics.contiguity(
        atlas_R, load_npz(config.INPUTS_DIR / "adjacency_R.npz")
    )

    n_contig = cL["n_contiguous"] + cR["n_contiguous"]
    n_parcels = cL["n_parcels"] + cR["n_parcels"]

    return {
        "subject": subject_id,
        "method": METHOD,
        "variant": config.VARIANT,
        "homogeneity_REST1": round(h1, 5),
        "homogeneity_REST2": round(h2, 5),
        "homogeneity_ALL": round(ha, 5),
        "fc_test_retest": round(trt, 5),
        "n_parcels": n_parcels,
        "pct_contiguous": round(100.0 * n_contig / n_parcels, 2),
        "mean_components_L": round(cL["mean_components"], 3),
        "mean_components_R": round(cR["mean_components"], 3),
        "max_components_L": cL["max_components"],
        "max_components_R": cR["max_components"],
    }


def main():
    parser = argparse.ArgumentParser(description="Schaefer validation.")
    parser.add_argument("--subject", required=True)
    args = parser.parse_args()

    out_dir = config.OUTPUTS_DIR / f"Method_0_{METHOD}" / "Validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    aL, aR = load_atlas()
    row = validate_subject(args.subject, aL, aR)

    pd.DataFrame([row]).to_csv(out_dir / f"{args.subject}.csv", index=False)

    print(f"Subject {args.subject}")
    for k, v in row.items():
        if k not in ("subject", "method", "variant"):
            print(f"  {k:22s} {v}")
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
