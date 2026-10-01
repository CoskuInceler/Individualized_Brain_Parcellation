"""
GORDON VALIDATION (Level 5)
===========================
Per-subject validation metrics for the gradient-based parcellation.



Two label sets are validated separately:



    Labels      the number of parcels the boundary map implies, which
                varies between subjects
    Labels_200  merged down to 200, comparable with the other methods



Unlike the other five methods, this one leaves some vertices unassigned:
those sitting on strong boundaries are treated as transition zones rather
than forced into a parcel. The proportion assigned is reported, since it
is a property of the method rather than a failure.



Output: Outputs/Method_5_Gordon/Validation{_200}/{subject}.csv
"""

import argparse
import time
import numpy as np
import pandas as pd
from scipy.sparse import load_npz


import config
import utils
import metrics

METHOD = "Gordon"


def validate_subject(subject_id, label_dir_name):
    method_dir = config.OUTPUTS_DIR / f"Method_5_{METHOD}"
    lab_dir = method_dir / label_dir_name
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
        "method": METHOD if label_dir_name == "Labels" else f"{METHOD}_200",
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
        "n_parcels_REST1": int(len(np.unique(lab["REST1"][lab["REST1"] > 0]))),
        "n_parcels_REST2": int(len(np.unique(lab["REST2"][lab["REST2"] > 0]))),
        "pct_assigned": round(100.0 * (lab["ALL"] > 0).sum() / n_cortex, 2),
        "pct_contiguous": round(100.0 * n_contig / n_parcels, 2),
        "mean_components_L": round(cL["mean_components"], 3),
        "mean_components_R": round(cR["mean_components"], 3),
        "max_components_L": cL["max_components"],
        "max_components_R": cR["max_components"],
    }


def main():
    parser = argparse.ArgumentParser(description="Gordon validation.")
    parser.add_argument("--subject", required=True)
    parser.add_argument(
        "--labels", default="Labels", choices=["Labels", "Labels_200"]
    )
    args = parser.parse_args()

    suffix = "" if args.labels == "Labels" else "_200"
    out_dir = config.OUTPUTS_DIR / f"Method_5_{METHOD}" / f"Validation{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    row = validate_subject(args.subject, args.labels)
    pd.DataFrame([row]).to_csv(out_dir / f"{args.subject}.csv", index=False)

    print(f"Variant {config.VARIANT} | {args.labels} | Subject {args.subject}")
    for k, v in row.items():
        if k not in ("subject", "method", "variant"):
            print(f"  {k:26s} {v}")
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
