import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

uds_df = pd.read_csv("alzheimer/udslong_imputed.csv")
mri_df = pd.read_csv("alzheimer/mriscanclean.csv")
adj_df = pd.read_csv("alzheimer/combined_adjacency_matrix.csv")

from pathlib import Path
 
TABLES = Path("tables")
FIGURES = Path("figures")
TABLES.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
 
BRAIN_COLS = [
    "LH_ENTORHINAL_GVOL", "RH_ENTORHINAL_GVOL",
    "LH_PARAHIPPOCAMPAL_GVOL", "RH_PARAHIPPOCAMPAL_GVOL",
    "LH_PRECUNEUS_GVOL", "RH_PRECUNEUS_GVOL",
    "LH_INSULA_GVOL", "RH_INSULA_GVOL",
]
KEY_VARS = (
    ["NACCAGE", "SEX", "EDUC", "INDEPEND", "MOCATOTS", "CDRGLOB", "NACCUDSD",
     "DIABETES", "HYPERTEN", "CBSTROKE"] + BRAIN_COLS
)
 
# NACCUDSD exists in BOTH files -> a merge on NACCID alone creates _x / _y
if "NACCUDSD" not in combined_df.columns and "NACCUDSD_x" in combined_df.columns:
    combined_df = (combined_df.drop(columns=["NACCUDSD_y"])
                   .rename(columns={"NACCUDSD_x": "NACCUDSD"}))
 
# Never modify combined_df: all cleaning happens on a copy
clean_df = combined_df.copy()
 
# ---------------------------------------------------------------- logging
cleaning_log = []
 
 
def log_step(variable, rule, n, action, reason):
    cleaning_log.append({"variable": variable, "rule": rule, "n_affected": n,
                         "action": action, "reason": reason})
    print(f"  [{variable}] {rule}: {n} -> {action}")
 
 
def set_nan(df, col, mask, rule, reason):
    """Set df[col] to NaN where mask is True and record it in the log."""
    n = int(mask.sum())
    df.loc[mask, col] = np.nan
    log_step(col, rule, n, "set to NaN", reason)
 
 
# ------------------------------------------------ 3 & 4. MoCA validation
print("--- 3 & 4. MoCA Score Validation & Cleaning ---")
print("Before cleaning:")
print(clean_df["MOCATOTS"].describe())
 
# (a) special code 88  (b) values outside the valid 0-30 range
set_nan(clean_df, "MOCATOTS", clean_df["MOCATOTS"] == 88,
        "value == 88", "special code (not assessed) - confirm in data guide")
set_nan(clean_df, "MOCATOTS", clean_df["MOCATOTS"] > 30,
        "value > 30 (excluding 88)",
        "outside valid MoCA range 0-30; likely imputation artifact")
 
print("\nAfter cleaning:")
print(clean_df["MOCATOTS"].describe())
 
# (c) zero scores: possible but extreme -> keep, flag, revisit in Phase 5
clean_df["MOCA_ZERO_FLAG"] = clean_df["MOCATOTS"] == 0
log_step("MOCATOTS", "value == 0", int(clean_df["MOCA_ZERO_FLAG"].sum()),
         "kept + flagged", "possible but extreme; sensitivity check in Phase 5")
 
# (d) non-integer MoCA values: MoCA is an integer score -> probably imputed
valid_moca = clean_df["MOCATOTS"].notna()
clean_df["MOCA_NONINT_FLAG"] = valid_moca & (clean_df["MOCATOTS"] % 1 != 0)
n_nonint = int(clean_df["MOCA_NONINT_FLAG"].sum())
log_step("MOCATOTS", "non-integer value", n_nonint, "kept + flagged",
         "MoCA is an integer score, so these values are probably imputed")
print(f"  Non-integer (probably imputed) MoCA: {n_nonint} of "
      f"{int(valid_moca.sum())} valid ({100 * n_nonint / valid_moca.sum():.1f}%)")
 
# ------------------------------------------- special codes in other vars
# CONFIRM each code in the project's variable guide before keeping this
SPECIAL_CODES = {
    "EDUC": [99],
    "INDEPEND": [9],
    "HYPERTEN": [9],
    "DIABETES": [9],
    "CBSTROKE": [9],
}
print("\n--- Special codes -> NaN ---")
for col, codes in SPECIAL_CODES.items():
    set_nan(clean_df, col, clean_df[col].isin(codes), f"value in {codes}",
            "special code 'unknown' - confirm in data guide")
 
# --------------------------------------------- impossible MRI volumes
# Only the cell is set to NaN; the participant stays in the sample.
print("\n--- Impossible MRI volumes -> NaN ---")
for col in BRAIN_COLS:
    set_nan(clean_df, col, clean_df[col] <= 0, "volume <= 0",
            "physically impossible (segmentation failure); participant kept")
 
# ------------------------------- binary history variables (NaN preserved)
def to_binary_history(s, threshold=0.5):
    """NaN stays NaN. Values >= threshold -> 1 (active OR remote history), else 0.
    (The old .apply(lambda x: 1 if x >= 0.5 else 0) turned NaN into 0.)"""
    return (s >= threshold).astype(float).where(s.notna())
 
 
BINARY_VARS = {"HYPERTEN": "HYPERTEN_BIN", "DIABETES": "DIABETES_BIN",
               "CBSTROKE": "STROKE_BIN"}
print("\n--- Binary history variables ---")
for col, new_col in BINARY_VARS.items():
    n_frac = int((clean_df[col].dropna() % 1 != 0).sum())
    log_step(col, "fractional value (imputed)", n_frac, "rounded by threshold",
             "binary rule: value >= 0.5 -> 1 (active or remote history)")
    clean_df[new_col] = to_binary_history(clean_df[col])
    print(clean_df[new_col].value_counts(dropna=False).sort_index().to_string())
 
# ------------------------------------------------ missingness table
analysis_vars = [c for c in KEY_VARS if c in clean_df.columns]
valid_counts = clean_df[analysis_vars].notna().sum()
total_rows = len(clean_df)
summary_df = pd.DataFrame({
    "Valid N": valid_counts,
    "Missing N": total_rows - valid_counts,
    "Missing %": (total_rows - valid_counts) / total_rows * 100,
})
print("\n--- Valid Data Count for Selected Key Variables (after cleaning) ---")
print(summary_df.round(2))
summary_df.round(2).to_csv(TABLES / "key_variables_missingness.csv")
 
# ------------------------------------------------ analytic sample + flow
n_merged = len(clean_df)
analysis_df = clean_df[clean_df["MOCATOTS"].notna()].copy()  # MoCA = main score
flow = pd.DataFrame([
    {"step": "UDS + MRI merged (unique participants)",
     "n_removed": 0, "n_remaining": n_merged},
    {"step": "MoCA missing after cleaning (code 88 or > 30)",
     "n_removed": n_merged - len(analysis_df), "n_remaining": len(analysis_df)},
])
print("\n--- Sample flow ---")
print(flow.to_string(index=False))
flow.to_csv(TABLES / "sample_flow.csv", index=False)
 
 
# ------------------ HYPERTEN / DIABETES / STROKE vs MoCA (descriptive)
def moca_by_binary(df, col, label_no, label_yes, title, fname):
    d = df[["MOCATOTS", col]].dropna().copy()
    labels = {0.0: label_no, 1.0: label_yes}
    d["group"] = d[col].map(labels)
 
    g = d.groupby("group")["MOCATOTS"]
    stats = g.agg(n="count", mean="mean", sd="std", median="median",
                  q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75),
                  min="min", max="max")
    stats["iqr"] = stats["q3"] - stats["q1"]
    stats = stats.reindex([label_no, label_yes])
    print(f"\nMoCA by {title}:\n{stats.round(2)}")
    stats.round(2).to_csv(TABLES / f"{Path(fname).stem}_stats.csv")
 
    counts = d["group"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=d, x="group", y="MOCATOTS", order=[label_no, label_yes],
                hue="group", palette="Set2", legend=False, ax=ax)
    ax.set_xticks(range(2))
    ax.set_xticklabels([f"{o}\n(n = {counts.get(o, 0)})"
                        for o in [label_no, label_yes]])
    ax.set_title(f"MoCA total score by {title}")
    ax.set_xlabel("")
    ax.set_ylabel("MoCA total score (0-30 points)")
    ax.set_ylim(-0.5, 30.5)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.savefig(FIGURES / fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return stats
 
 
print("\n" + "=" * 50)
print("--- Vascular history vs MoCA (descriptive only) ---")
print("=" * 50)
moca_by_binary(analysis_df, "HYPERTEN_BIN", "No hypertension", "Hypertension history",
               "hypertension history", "bp_vs_moca_boxplot.png")
moca_by_binary(analysis_df, "DIABETES_BIN", "No diabetes", "Diabetes history",
               "diabetes history", "diabetes_vs_moca_boxplot.png")
moca_by_binary(analysis_df, "STROKE_BIN", "No stroke", "Stroke history",
               "stroke history", "stroke_vs_moca_boxplot.png")
 
# ----------------------------------------- MRI outliers: REPORT, no removal
print("\n--- MRI Outlier Report (IQR rule, nothing removed) ---")
report_rows, flagged_parts = [], []
for col in BRAIN_COLS:
    s = analysis_df[col].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    low = analysis_df[col] < lo
    high = analysis_df[col] > hi
 
    report_rows.append({"variable": col, "n_valid": len(s), "min": s.min(),
                        "max": s.max(), "Q1": q1, "Q3": q3, "IQR": iqr,
                        "lower_bound": lo, "upper_bound": hi,
                        "n_low": int(low.sum()), "n_high": int(high.sum()),
                        "pct_flagged": 100 * int((low | high).sum()) / len(s)})
    analysis_df[f"OUT_{col}"] = low | high  # flag column; rows are kept
 
    flagged = analysis_df.loc[low | high, ["NACCID", col]].rename(columns={col: "value"})
    flagged["variable"] = col
    flagged["direction"] = np.where(flagged["value"] < lo, "low", "high")
    flagged_parts.append(flagged)
 
outlier_report = pd.DataFrame(report_rows)
outlier_ids = pd.concat(flagged_parts, ignore_index=True)
print(outlier_report.round(1).to_string(index=False))
outlier_report.round(2).to_csv(TABLES / "mri_outlier_report.csv", index=False)
outlier_ids.to_csv(TABLES / "mri_outlier_ids.csv", index=False)
 
analysis_df["MRI_OUTLIER_COUNT"] = analysis_df[[f"OUT_{c}" for c in BRAIN_COLS]].sum(axis=1)
n_any = int((analysis_df["MRI_OUTLIER_COUNT"] > 0).sum())
print(f"\nParticipants flagged in >= 1 of {len(BRAIN_COLS)} regions: {n_any} "
      f"of {len(analysis_df)} ({100 * n_any / len(analysis_df):.1f}%) - none removed")
 
# ------------------------------------------------------- save everything
pd.DataFrame(cleaning_log).to_csv(TABLES / "cleaning_log.csv", index=False)
analysis_df.to_csv("cleaned_phase1_data.csv", index=False)  # add to .gitignore
print(f"\nFinal analytic sample: {len(analysis_df)} participants "
      f"-> cleaned_phase1_data.csv")

