"""
SLIC VALIDATION (Levels 3 and 4)
================================
Per-subject validation metrics for SLIC-F and SLIC-C.



Both are fitted to the data, so homogeneity is reported two ways:



    within   parcellation and measurement use the same session
    across   parcellation from one session, measured on the other



Also reports the Dice overlap between the REST1 and REST2 parcellations,
which measures how stable the parcellation itself is across sessions.



The two methods differ only in whether the spatial term is included, so
one script handles both and the method is passed in.



Output: Outputs/Method_{3,4}_SLIC_{F,C}/Validation/{subject}.csv
"""

import argparse
import time
import numpy as np
import pandas as pd
from scipy.sparse import load_npz


import config
import utils
import metrics

METHODS = {
    "SLIC_F": "Method_3_SLIC_F",
    "SLIC_C": "Method_4_SLIC_C",
}


def validate_subject(subject_id, method):
    method_dir = config.OUTPUTS_DIR / METHODS[method]
    lab_dir = method_dir / "Labels"
    cleaned = config.OUTPUTS_DIR / "Cleaned"

    lab = {
        s: np.load(lab_dir / f"{subject_id}_{s}_labels.npy")
        for s in ("REST1", "REST2", "ALL")
    }

    n_cortex = len(lab["ALL"])
    ts1 = np.load(cleaned / f"{subject_id}_REST1.npy")[:, :n_cortex]
    ts2 = np.load(cleaned / f"{subject_id}_REST2.npy")[:, :n_cortex]
    ts_all = np.concatenate([ts1, ts2], axis=0)

    h_all, _ = metrics.homogeneity(ts_all, lab["ALL"])
    h_w1, _ = metrics.homogeneity(ts1, lab["REST1"])
    h_w2, _ = metrics.homogeneity(ts2, lab["REST2"])
    h_x1, _ = metrics.homogeneity(ts2, lab["REST1"])
    h_x2, _ = metrics.homogeneity(ts1, lab["REST2"])

    trt = metrics.fc_test_retest(ts1, ts2, lab["ALL"])
    dice_sessions = metrics.dice_pairwise(lab["REST1"], lab["REST2"])

    ref = config.get_brain_path(subject_id, config.RUN_IDS[0])
    n_L = len(utils.get_valid_vertices(ref, "CIFTI_STRUCTURE_CORTEX_LEFT"))
    cL = metrics.contiguity(
        lab["ALL"][:n_L], load_npz(config.INPUTS_DIR / "adjacency_L.npz")
    )
    cR = metrics.contiguity(
        lab["ALL"][n_L:], load_npz(config.INPUTS_DIR / "adjacency_R.npz")
    )

    n_contig = cL["n_contiguous"] + cR["n_contiguous"]
    n_parcels = cL["n_parcels"] + cR["n_parcels"]

    return {
        "subject": subject_id,
        "method": method,
        "variant": config.VARIANT,
        "homogeneity_ALL": round(h_all, 5),
        "homogeneity_within_REST1": round(h_w1, 5),
        "homogeneity_within_REST2": round(h_w2, 5),
        "homogeneity_across_REST1": round(h_x1, 5),
        "homogeneity_across_REST2": round(h_x2, 5),
        "homogeneity_within_mean": round((h_w1 + h_w2) / 2, 5),
        "homogeneity_across_mean": round((h_x1 + h_x2) / 2, 5),
        "fc_test_retest": round(trt, 5),
        "dice_sessions": round(dice_sessions, 5),
        "n_parcels": n_parcels,
        "pct_contiguous": round(100.0 * n_contig / n_parcels, 2),
        "mean_components_L": round(cL["mean_components"], 3),
        "mean_components_R": round(cR["mean_components"], 3),
        "max_components_L": cL["max_components"],
        "max_components_R": cR["max_components"],
    }


def main():
    parser = argparse.ArgumentParser(description="SLIC validation.")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    args = parser.parse_args()

    out_dir = config.OUTPUTS_DIR / METHODS[args.method] / "Validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    row = validate_subject(args.subject, args.method)
    pd.DataFrame([row]).to_csv(out_dir / f"{args.subject}.csv", index=False)

    print(f"Variant {config.VARIANT} | {args.method} | Subject {args.subject}")
    for k, v in row.items():
        if k not in ("subject", "method", "variant"):
            print(f"  {k:26s} {v}")
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
