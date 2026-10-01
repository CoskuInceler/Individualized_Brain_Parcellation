"""
gMSHBM VALIDATION (Level 1)
===========================
Per-subject validation metrics for the gMSHBM parcellation.
Same metrics as the Schaefer validation, but the labels are read from
each subject's own parcellation rather than a shared atlas.
Output: Outputs/Method_1_gMSHBM/Validation/{subject}.csv
"""

import argparse
import time
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import config
import utils
import metrics

METHOD = "gMSHBM"


def validate_subject(subject_id):
    method_dir = config.OUTPUTS_DIR / f"Method_1_{METHOD}"
    labels = np.load(method_dir / "Labels" / f"{subject_id}_labels.npy")
    n_cortex = len(labels)
    # split back into hemispheres for the contiguity check
    ref = config.get_brain_path(subject_id, config.RUN_IDS[0])
    n_L = len(utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_LEFT"))
    labels_L = labels[:n_L]
    labels_R = labels[n_L:]
    cleaned = config.OUTPUTS_DIR / "Cleaned"
    ts1 = np.load(cleaned / f"{subject_id}_REST1.npy")[:, :n_cortex]
    ts2 = np.load(cleaned / f"{subject_id}_REST2.npy")[:, :n_cortex]
    ts_all = np.concatenate([ts1, ts2], axis=0)
    h1, _ = metrics.homogeneity(ts1, labels)
    h2, _ = metrics.homogeneity(ts2, labels)
    ha, _ = metrics.homogeneity(ts_all, labels)
    trt = metrics.fc_test_retest(ts1, ts2, labels)
    cL = metrics.contiguity(
        labels_L, load_npz(config.INPUTS_DIR / "adjacency_L.npz")
    )
    cR = metrics.contiguity(
        labels_R, load_npz(config.INPUTS_DIR / "adjacency_R.npz")
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
    parser = argparse.ArgumentParser(description="gMSHBM validation.")
    parser.add_argument("--subject", required=True)
    args = parser.parse_args()
    out_dir = config.OUTPUTS_DIR / f"Method_1_{METHOD}" / "Validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    row = validate_subject(args.subject)
    pd.DataFrame([row]).to_csv(out_dir / f"{args.subject}.csv", index=False)
    print(f"Variant {config.VARIANT} | Subject {args.subject}")
    for k, v in row.items():
        if k not in ("subject", "method", "variant"):
            print(f"  {k:22s} {v}")
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
