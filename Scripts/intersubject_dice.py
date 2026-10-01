"""
INTER-SUBJECT DICE
==================
How similar is one subject's parcellation to everybody else's.



For each subject this computes the mean Dice overlap against all other
subjects, which places the method on the individualisation gradient: a
group atlas gives 1.0 by construction, while a fully data-driven method
gives whatever the data happens to produce.



Overlap is measured through co-assignment of vertex pairs rather than
parcel identity, so parcel numbering need not correspond across subjects.
See metrics.dice_pairwise.



Run one subject per array task; the group mean follows from averaging the
per-subject values, and keeping them separate also shows which subjects
sit far from everyone else.



Usage:
    python3 intersubject_dice.py --subject 100206 --method Method_2_AGP
"""

import argparse
import time
import numpy as np
import pandas as pd


import config
import metrics

# parcellation to compare across subjects; ALL where a method has sessions
SESSION = "ALL"


def label_path(method_dir, subject_id, folder="Labels"):
    """
    Labels for one subject, whichever naming the method uses.



    Methods fitted per session store {subject}_{session}_labels.npy;
    gMSHBM, whose parcellation does not depend on the data, stores
    {subject}_labels.npy.



    `folder` selects between label sets where a method produces more than
    one, as the gradient method does with its own resolution and the
    version matched to 200 parcels.
    """
    lab_dir = method_dir / folder
    with_session = lab_dir / f"{subject_id}_{SESSION}_labels.npy"
    if with_session.exists():
        return with_session
    return lab_dir / f"{subject_id}_labels.npy"


def main():
    parser = argparse.ArgumentParser(description="Inter-subject Dice.")
    parser.add_argument("--subject", required=True)
    parser.add_argument(
        "--method", required=True, help="method folder, e.g. Method_2_AGP"
    )
    parser.add_argument(
        "--labels", default="Labels", help="label subfolder, e.g. Labels_200"
    )
    args = parser.parse_args()

    method_dir = config.OUTPUTS_DIR / args.method
    suffix = (
        "" if args.labels == "Labels" else f"_{args.labels.split('_')[-1]}"
    )
    out_dir = method_dir / f"Dice{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    mine = np.load(label_path(method_dir, args.subject, args.labels))

    others = [s for s in config.SUBJECT_IDS if s != args.subject]
    values = []
    missing = 0

    for other in others:
        path = label_path(method_dir, other, args.labels)
        if not path.exists():
            missing += 1
            continue
        values.append(metrics.dice_pairwise(mine, np.load(path)))

    values = np.asarray(values)

    row = {
        "subject": args.subject,
        "method": args.method,
        "variant": config.VARIANT,
        "dice_mean": round(float(values.mean()), 5),
        "dice_sd": round(float(values.std()), 5),
        "dice_min": round(float(values.min()), 5),
        "dice_max": round(float(values.max()), 5),
        "n_compared": len(values),
    }
    pd.DataFrame([row]).to_csv(out_dir / f"{args.subject}.csv", index=False)

    if missing:
        print(f"  {missing} subjects had no labels and were skipped")
    print(f"Variant {config.VARIANT} | {args.method} | Subject {args.subject}")
    print(
        f"  dice {row['dice_mean']:.4f} +- {row['dice_sd']:.4f} "
        f"({row['dice_min']:.4f} - {row['dice_max']:.4f}), n = {row['n_compared']}"
    )
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
