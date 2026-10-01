"""
Which of the 1,029 gMSHBM participants have data for all four runs?
Writes the result to Inputs/subjects_final.txt.
"""

import config

complete, incomplete = [], []


for s in config.SUBJECT_IDS:
    ok = all(
        config.get_brain_path(s, r).exists()
        and config.get_confound_path(s, r).exists()
        for r in config.RUN_IDS
    )
    (complete if ok else incomplete).append(s)


print(f"gMSHBM participants : {len(config.SUBJECT_IDS)}")
print(f"All four runs       : {len(complete)}")
print(f"Missing runs        : {len(incomplete)}")


if incomplete:
    print("\nParticipants with missing runs:")
    for s in incomplete:
        print("  ", s)


out = config.INPUTS_DIR / "subjects_final.txt"
out.write_text("\n".join(complete) + "\n")
print(f"\nWritten: {out}")
