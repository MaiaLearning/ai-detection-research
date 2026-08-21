"""Feature-level diagnosis of the RAID homoglyph / zero-width-space TPR and
FPR ceiling effect reported in TECHNICAL_REPORT.md Section 5.7 -- persisted
here since it was previously only verified ad hoc and not committed to
results/, unlike every other figure in this study.

Method: for one matched (source_id/adv_source_id) document pair per attack,
recompute all nine Tier-1 features before and after the attack directly with
src/features.py, on the same llama-chat RAID abstracts subset Section 5.7's
TPR/FPR table uses. This is a single illustrative pair, not an aggregate --
the point is to show the mechanism (feature saturation), not to re-estimate
the TPR/FPR numbers themselves, which already come from experiment4_raid_robustness.py.

Usage: uv run python scripts/analyze_raid_unicode_diagnostics.py
Requires: data/raid_abstracts_subset.csv
Output: results/experiment4_unicode_feature_diagnostics.csv
"""
import json
from pathlib import Path

import pandas as pd

from src import features as feat

RAID_PATH = Path("data/raid_abstracts_subset.csv")
RESULTS_DIR = Path("results")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(RAID_PATH)
    none_df = df[df["attack"] == "none"].drop_duplicates("source_id").set_index("source_id")
    homo_df = df[df["attack"] == "homoglyph"].drop_duplicates("adv_source_id").set_index("adv_source_id")
    zw_df = df[df["attack"] == "zero_width_space"].drop_duplicates("adv_source_id").set_index("adv_source_id")

    common_homo = none_df.index.intersection(homo_df.index)
    common_zw = none_df.index.intersection(zw_df.index)
    print(f"Matched pairs available: homoglyph n={len(common_homo)}, zero_width_space n={len(common_zw)}")

    sid_homo = common_homo[0]
    base_text = none_df.loc[sid_homo, "generation"]
    homo_text = homo_df.loc[sid_homo, "generation"]

    sid_zw = common_zw[0]
    base_text_zw = none_df.loc[sid_zw, "generation"]
    zw_text = zw_df.loc[sid_zw, "generation"]

    rows = []
    for name, fn in feat.TIER1_FEATURES.items():
        rows.append({
            "feature": name,
            "baseline_homoglyph_pair": fn(base_text),
            "after_homoglyph": fn(homo_text),
            "baseline_zero_width_pair": fn(base_text_zw),
            "after_zero_width_space": fn(zw_text),
        })
    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False))
    result_df.to_csv(RESULTS_DIR / "experiment4_unicode_feature_diagnostics.csv", index=False)

    manifest = {
        "homoglyph_example_source_id": str(sid_homo),
        "zero_width_space_example_source_id": str(sid_zw),
        "note": (
            "Single matched (source_id, adv_source_id) document pair per attack, "
            "from the same llama-chat RAID abstracts subset used for "
            "experiment4_raid_robustness.py's TPR/FPR table. Illustrates the "
            "feature-saturation mechanism; the TPR/FPR figures themselves come "
            "from that script's full n=107/n=493 runs, not from this pair."
        ),
    }
    with open(RESULTS_DIR / "experiment4_unicode_feature_diagnostics_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {RESULTS_DIR / 'experiment4_unicode_feature_diagnostics_manifest.json'}")


if __name__ == "__main__":
    main()
