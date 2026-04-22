import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy.stats import skew, norm, ttest_rel, t
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


plt.style.use("default")
plt.rcParams["axes.grid"] = False
plt.rcParams["grid.alpha"] = 0.0
plt.rcParams["axes.facecolor"] = "white"


NT_FILE = r"C:\Users\iamas\Downloads\Data Analytics PRT564\NT Project\ntipp-2022-powerbi-xls.xlsx"
ABS_FILE = r"C:\Users\iamas\Downloads\Data Analytics PRT564\NT Project\ABS Population Data.xlsx"
OUTPUT_DIR = r"C:\Users\iamas\Downloads\PRT564"

os.makedirs(OUTPUT_DIR, exist_ok=True)

COST_COL = "Est Cost $M"
REGION_COL = "Region"
SECTOR_COL = "Industry Sector"

CATEGORICAL_COLS = [
    "Region",
    "Sub Sector",
    "Enabling Infrastructure",
    "Industry Sector",
    "Category"
]

NUMERIC_CANDIDATE_COLS = [
    "2022-23",
    "2023-24",
    "0-5",
    "5-10",
    "0-10",
    "0-15",
    "10-15",
    "15+"
]

REGION_RENAME_MAP = {
    "Central Australia": "Alice Springs",
    "Big Rivers": "Katherine"
}

REGION_ORDER = [
    "Greater Darwin",
    "Alice Springs",
    "Katherine",
    "Territory Wide",
    "East Arnhem",
    "Top End Rural",
    "Barkly"
]

ABS_SA3_TO_PRESENTED_REGION = {
    "Darwin City": "Greater Darwin",
    "Darwin Suburbs": "Greater Darwin",
    "Litchfield": "Greater Darwin",
    "Palmerston": "Greater Darwin",
    "Alice Springs": "Alice Springs",
    "Barkly": "Barkly",
    "East Arnhem": "East Arnhem",
    "Katherine": "Katherine",
    "Daly - Tiwi - West Arnhem": "Top End Rural"
}

TITLE_COLOR = "#0D2B45"
TEXT_COLOR = "#334155"
GRID_COLOR = "#D9E2EC"
COL_RAW = "#1565C0"
COL_LOG = "#1A6E3C"
COL_GAUSS = "#4B5563"
COL_BG = "#F8FAFC"
COL_AXIS = "#94A3B8"
COL_NOTE_FILL = "#FFF7E6"
COL_NOTE_EDGE = "#D97706"
COL_INFO_FILL = "#EEF6FF"
COL_INFO_EDGE = "#3B82F6"

REGION_COLORS = [
    "#0F8B9D",
    "#2F7DE1",
    "#2DB55D",
    "#E7A11A",
    "#C53334",
    "#6F3A8A",
    "#2E6B2E"
]

SECTOR_COLORS = [
    "#0F8B9D",
    "#2F7DE1",
    "#2DB55D",
    "#E7A11A",
    "#C53334",
    "#6F3A8A",
    "#2E6B2E",
    "#8B5CF6",
    "#EC4899",
    "#14B8A6"
]


def out_path(filename: str) -> str:
    return os.path.join(OUTPUT_DIR, filename)


def estimate_mode_from_hist(data, bins=14):
    counts, bin_edges = np.histogram(data, bins=bins)
    max_bin_index = np.argmax(counts)
    return (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2


def add_gaussian_curve(ax, data, bins):
    mu = np.mean(data)
    sigma = np.std(data, ddof=1)
    if sigma <= 0:
        return
    x = np.linspace(min(data), max(data), 500)
    bin_width = bins[1] - bins[0]
    y = norm.pdf(x, mu, sigma) * len(data) * bin_width
    ax.plot(x, y, color=COL_GAUSS, linewidth=2.4, linestyle="-")


def draw_histogram_panel(ax, data, title, xlabel, bar_color, skew_val):
    mean_val = np.mean(data)
    median_val = np.median(data)
    mode_val = estimate_mode_from_hist(data, bins=14)

    ax.set_facecolor(COL_BG)
    _, bins, _ = ax.hist(
        data,
        bins=14,
        color=bar_color,
        edgecolor="black",
        linewidth=0.9,
        alpha=0.88
    )

    ax.grid(False)
    ax.xaxis.grid(False, which="both")
    ax.yaxis.grid(False, which="both")
    ax.minorticks_off()

    add_gaussian_curve(ax, data, bins)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COL_AXIS)
    ax.spines["bottom"].set_color(COL_AXIS)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=14, color=TEXT_COLOR)
    ax.set_xlabel(xlabel, fontsize=11.5, color=TEXT_COLOR, labelpad=8)
    ax.set_ylabel("Frequency", fontsize=11.5, color=TEXT_COLOR, labelpad=8)
    ax.tick_params(axis="both", labelsize=10, colors="#475569")

    stats_text = (
        f"Mean   = {mean_val:.2f}\n"
        f"Median = {median_val:.2f}\n"
        f"Mode   = {mode_val:.2f}\n"
        f"Skewness = {skew_val:+.3f}"
    )

    ax.text(
        0.98, 0.97,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.1,
        color=TEXT_COLOR,
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.40",
            facecolor=COL_NOTE_FILL,
            edgecolor=COL_NOTE_EDGE,
            linewidth=1.15,
            alpha=0.97
        )
    )


def adjusted_r2_score(r2: float, n: int, p: int) -> float:
    if n <= p + 1:
        return float("nan")
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))


def load_abs_population(abs_file: str) -> pd.DataFrame:
    abs_df = pd.read_excel(abs_file, sheet_name="Table 3", header=6)
    abs_df = abs_df[abs_df["S/T name"] == "Northern Territory"].copy()

    persons_col = abs_df.columns[-1]
    abs_df = abs_df[["SA3 name", "SA2 name", persons_col]].copy()
    abs_df = abs_df.rename(columns={persons_col: "Persons"})
    abs_df = abs_df[abs_df["Persons"].notna()].copy()

    abs_df["Presented Region"] = abs_df["SA3 name"].map(ABS_SA3_TO_PRESENTED_REGION)

    abs_region_pop = (
        abs_df.dropna(subset=["Presented Region"])
        .groupby("Presented Region", as_index=False)["Persons"]
        .sum()
    )

    abs_region_pop["Presented Region"] = pd.Categorical(
        abs_region_pop["Presented Region"],
        categories=REGION_ORDER,
        ordered=True
    )
    return abs_region_pop.sort_values("Presented Region")


def group_sector_name(sector_value: str) -> str:
    if pd.isna(sector_value):
        return "Other"

    s = str(sector_value).strip().lower()

    if any(k in s for k in ["road", "transport", "aviation", "airport", "port", "marine"]):
        return "Transport"
    if any(k in s for k in ["education", "training", "school"]):
        return "Education"
    if "health" in s:
        return "Health"
    if "housing" in s or "residential" in s:
        return "Housing"
    if any(k in s for k in ["tourism", "visitor", "hospitality"]):
        return "Tourism"
    if any(k in s for k in ["water", "power", "energy", "electric", "waste", "utility", "telecom", "communications"]):
        return "Utilities"
    if any(k in s for k in ["community", "culture", "arts", "library", "youth"]):
        return "Community"
    if any(k in s for k in ["justice", "police", "fire", "emergency", "correction", "safety"]):
        return "Public Safety"
    if any(k in s for k in ["sport", "recreation", "park", "leisure"]):
        return "Recreation"

    return "Other"


def collapse_small_groups(count_series: pd.Series, keep_top_n: int = 6) -> pd.Series:
    if len(count_series) <= keep_top_n:
        return count_series.sort_values(ascending=False)

    top = count_series.nlargest(keep_top_n).copy()
    other_sum = count_series.drop(top.index).sum()

    if other_sum > 0:
        top.loc["Other"] = other_sum

    return top.sort_values(ascending=False)


print("=" * 88)
print("PRT564 FULL ANALYSIS — PREPROCESSING, EDA, ABS INTEGRATION, AND REGRESSION")
print("=" * 88)
print(f"Output directory: {OUTPUT_DIR}")

nt_df = pd.read_excel(NT_FILE, sheet_name="PBIData")
print(f"\nOriginal NT dataset shape: {nt_df.shape}")

nt_df[COST_COL] = pd.to_numeric(nt_df[COST_COL], errors="coerce")
cost_median = nt_df[COST_COL].median()
nt_df[COST_COL] = nt_df[COST_COL].fillna(cost_median)

print(f"\nMedian imputation applied to '{COST_COL}'")
print(f"Imputed median value: {cost_median:.4f}")

before_filter = len(nt_df)
nt_df = nt_df[nt_df[COST_COL] > 0].copy()
after_filter = len(nt_df)
print(f"Removed {before_filter - after_filter} rows with non-positive cost values.")

nt_df["log_cost"] = np.log10(nt_df[COST_COL])
print("Created target variable: log_cost")

for col in NUMERIC_CANDIDATE_COLS:
    if col in nt_df.columns:
        nt_df[col] = pd.to_numeric(nt_df[col], errors="coerce")
        if nt_df[col].notna().any():
            nt_df[col] = nt_df[col].fillna(nt_df[col].median())

nt_df["Region_Presented"] = nt_df[REGION_COL].astype(str).str.strip().replace(REGION_RENAME_MAP)

available_categorical_cols = [c for c in CATEGORICAL_COLS if c in nt_df.columns]
df_encoded = pd.get_dummies(nt_df, columns=available_categorical_cols, drop_first=True, dtype=int)
df_encoded.to_csv(out_path("nt_preprocessed_encoded.csv"), index=False)
print(f"Saved -> {out_path('nt_preprocessed_encoded.csv')}")

raw_skew = skew(nt_df[COST_COL])
log_skew = skew(nt_df["log_cost"])

fig, axes = plt.subplots(1, 2, figsize=(18, 7.4), facecolor="white")
fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.17, wspace=0.12)

draw_histogram_panel(
    ax=axes[0],
    data=nt_df[COST_COL],
    title="Before Transformation — Raw Project Cost",
    xlabel="Estimated Cost ($M)",
    bar_color=COL_RAW,
    skew_val=raw_skew
)

draw_histogram_panel(
    ax=axes[1],
    data=nt_df["log_cost"],
    title="After Transformation — log₁₀(Estimated Cost)",
    xlabel="log₁₀(Estimated Cost $M)",
    bar_color=COL_LOG,
    skew_val=log_skew
)

fig.suptitle(
    "Log Transformation of NT Infrastructure Project Cost",
    fontsize=20,
    fontweight="bold",
    color=TITLE_COLOR,
    y=0.95
)

fig.text(
    0.5, 0.08,
    f"Skewness reduced from {raw_skew:.2f} to {log_skew:.2f}   |   "
    "Distribution appears more Gaussian-like after log transformation",
    ha="center",
    va="center",
    fontsize=10.8,
    color=TEXT_COLOR,
    fontfamily="monospace",
    bbox=dict(
        boxstyle="round,pad=0.42",
        facecolor=COL_INFO_FILL,
        edgecolor=COL_INFO_EDGE,
        linewidth=1.25,
        alpha=0.96
    )
)

plt.savefig(out_path("cost_distribution_comparison.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved -> {out_path('cost_distribution_comparison.png')}")

region_counts = nt_df["Region_Presented"].value_counts().reindex(REGION_ORDER)
region_summary = pd.DataFrame({
    "Presented Region": region_counts.index,
    "Project Count": region_counts.values
})
region_summary["Project Share (%)"] = (region_summary["Project Count"] / region_summary["Project Count"].sum() * 100).round(1)
region_summary.to_csv(out_path("project_count_by_presented_nt_region.csv"), index=False)
print(f"Saved -> {out_path('project_count_by_presented_nt_region.csv')}")

plt.figure(figsize=(11, 6))
bars = plt.bar(
    region_counts.index,
    region_counts.values,
    color=REGION_COLORS,
    edgecolor="none",
    width=0.42
)

for bar, value in zip(bars, region_counts.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 5,
        str(int(value)),
        ha="center",
        va="bottom",
        fontsize=12,
        color="#16324F"
    )

plt.title("Project Count by NT Region", fontsize=16, weight="normal", color="#16324F", pad=14)
plt.xlabel("Region", fontsize=12)
plt.ylabel("Projects", fontsize=12)
plt.xticks(rotation=42, ha="right", fontsize=11)
plt.yticks(fontsize=11)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#808080")
ax.spines["bottom"].set_color("#808080")
ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(out_path("project_count_by_nt_region.png"), dpi=300)
plt.close()
print(f"Saved -> {out_path('project_count_by_nt_region.png')}")

abs_region_pop = load_abs_population(ABS_FILE)
abs_region_pop.to_csv(out_path("abs_population_by_presented_region.csv"), index=False)
print(f"Saved -> {out_path('abs_population_by_presented_region.csv')}")

integration_df = region_summary.merge(
    abs_region_pop.rename(columns={"Persons": "ABS Population"}),
    on="Presented Region",
    how="left"
)

integration_df["Population Share (%)"] = (
    integration_df["ABS Population"] / integration_df["ABS Population"].sum() * 100
).round(1)

integration_df.to_csv(out_path("region_project_population_integration.csv"), index=False)
print(f"Saved -> {out_path('region_project_population_integration.csv')}")

x = np.arange(len(integration_df))
width = 0.36

fig, ax = plt.subplots(figsize=(12, 6.5), facecolor="white")
ax.bar(x - width/2, integration_df["Project Share (%)"], width, label="Project Share (%)", color="#0F8B9D")
ax.bar(x + width/2, integration_df["Population Share (%)"], width, label="ABS Population Share (%)", color="#E7A11A")

ax.set_title("Heterogeneous Integration — Project Share vs ABS Population Share", fontsize=16, color=TITLE_COLOR, pad=14)
ax.set_ylabel("Share (%)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(integration_df["Presented Region"], rotation=40, ha="right")
ax.legend(frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#808080")
ax.spines["bottom"].set_color("#808080")
ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(out_path("heterogeneous_region_comparison.png"), dpi=300)
plt.close()
print(f"Saved -> {out_path('heterogeneous_region_comparison.png')}")

sector_df = nt_df[nt_df[SECTOR_COL].notna()].copy()
sector_df[SECTOR_COL] = sector_df[SECTOR_COL].astype(str).str.strip()
sector_df["Sector Group"] = sector_df[SECTOR_COL].apply(group_sector_name)

group_counts = sector_df["Sector Group"].value_counts()
group_counts = collapse_small_groups(group_counts, keep_top_n=6)

sector_df["Sector Group Final"] = sector_df["Sector Group"].apply(
    lambda x: x if x in group_counts.index else "Other"
)

sector_counts = sector_df["Sector Group Final"].value_counts().reindex(group_counts.index)
sector_share = (sector_counts / sector_counts.sum() * 100).round(1)

sector_count_table = pd.DataFrame({
    "Sector Group": sector_counts.index,
    "Project Count": sector_counts.values,
    "Project Share (%)": sector_share.values
})
sector_count_table.to_csv(out_path("sector_project_counts_grouped.csv"), index=False)
print(f"Saved -> {out_path('sector_project_counts_grouped.csv')}")

sector_avg_cost = (
    sector_df.groupby("Sector Group Final")[COST_COL]
    .mean()
    .reindex(group_counts.index)
    .round(2)
)

sector_cost_table = pd.DataFrame({
    "Sector Group": sector_avg_cost.index,
    "Average Cost ($M)": sector_avg_cost.values
})
sector_cost_table.to_csv(out_path("sector_average_cost_grouped.csv"), index=False)
print(f"Saved -> {out_path('sector_average_cost_grouped.csv')}")

fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="white")
fig.subplots_adjust(left=0.05, right=0.97, top=0.84, bottom=0.18, wspace=0.20)

pie_colors = SECTOR_COLORS[:len(sector_counts)]

axes[0].pie(
    sector_counts.values,
    labels=sector_counts.index,
    autopct="%1.1f%%",
    startangle=110,
    colors=pie_colors,
    textprops={"fontsize": 10}
)
axes[0].set_title("Project Share by Sector Group", fontsize=16, color="#16324F", pad=12)

bars = axes[1].bar(
    sector_avg_cost.index,
    sector_avg_cost.values,
    color=pie_colors,
    edgecolor="none",
    width=0.58
)

for bar, value in zip(bars, sector_avg_cost.values):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        value + max(sector_avg_cost.values) * 0.02,
        f"{value:.1f}",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#16324F"
    )

axes[1].set_title("Average Cost by Sector Group ($M)", fontsize=16, color="#16324F", pad=12)
axes[1].set_ylabel("Average Cost ($M)", fontsize=12)
axes[1].tick_params(axis="x", rotation=35, labelsize=10)
axes[1].tick_params(axis="y", labelsize=10)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
axes[1].spines["left"].set_color("#808080")
axes[1].spines["bottom"].set_color("#808080")
axes[1].grid(axis="y", color=GRID_COLOR, linewidth=0.8)
axes[1].set_axisbelow(True)

top_sector = sector_share.idxmax()
top_sector_share = sector_share.max()
highest_cost_sector = sector_avg_cost.idxmax()
highest_cost_value = sector_avg_cost.max()

fig.suptitle(
    "EDA — Sector Analysis & Cost Patterns",
    fontsize=20,
    fontweight="bold",
    color=TITLE_COLOR,
    y=0.95
)

fig.text(
    0.5, 0.07,
    f"{top_sector} has the largest project share ({top_sector_share}%), "
    f"while {highest_cost_sector} has the highest average cost (${highest_cost_value:.1f}M). "
    "Agriculture and Mining are intentionally retained within Other for a cleaner sector summary.",
    ha="center",
    va="center",
    fontsize=10.2,
    color=TEXT_COLOR,
    bbox=dict(
        boxstyle="round,pad=0.42",
        facecolor=COL_INFO_FILL,
        edgecolor=COL_INFO_EDGE,
        linewidth=1.2,
        alpha=0.96
    )
)

plt.savefig(out_path("sector_analysis_cost_patterns.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved -> {out_path('sector_analysis_cost_patterns.png')}")

model_df = nt_df.merge(
    abs_region_pop.rename(columns={"Presented Region": "Region_Presented", "Persons": "ABS Population"}),
    on="Region_Presented",
    how="left"
)

if model_df["ABS Population"].isna().any():
    model_df["ABS Population"] = model_df["ABS Population"].fillna(model_df["ABS Population"].median())

available_categorical_cols = [c for c in CATEGORICAL_COLS if c in model_df.columns]

model_encoded = pd.get_dummies(
    model_df,
    columns=available_categorical_cols,
    drop_first=True,
    dtype=int
)

candidate_drop_cols = [
    COST_COL,
    "log_cost",
    "Project",
    "RefID",
    "Location",
    "Challenge / Opportunity",
    "Region_Presented"
]

y = model_encoded["log_cost"].copy()

drop_from_x = [c for c in candidate_drop_cols if c in model_encoded.columns]
X = model_encoded.drop(columns=drop_from_x, errors="ignore").copy()
X = X.select_dtypes(include=[np.number]).copy()

all_nan_cols = X.columns[X.isna().all()].tolist()
if all_nan_cols:
    print("\nDropping all-NaN columns:")
    for col in all_nan_cols:
        print(f"- {col}")
    X = X.drop(columns=all_nan_cols)

for col in X.columns:
    if X[col].isna().any():
        X[col] = X[col].fillna(X[col].median())

remaining_nan = X.isna().sum().sum()
print(f"\nRemaining NaN values in X: {remaining_nan}")
if remaining_nan > 0:
    nan_summary = X.isna().sum()
    nan_summary = nan_summary[nan_summary > 0]
    print(nan_summary)
    raise ValueError("Predictor matrix X still contains NaN values.")

print(f"\nRegression design matrix shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
response_range = y_test.max() - y_test.min()
nrmse = rmse / response_range if response_range != 0 else np.nan
r2 = r2_score(y_test, y_pred)
adj_r2 = adjusted_r2_score(r2, n=len(y_test), p=X_test.shape[1])

pred_df = pd.DataFrame({
    "Actual_log_cost": y_test.values,
    "Predicted_log_cost": y_pred
})
pred_df.to_csv(out_path("regression_predictions.csv"), index=False)
print(f"Saved -> {out_path('regression_predictions.csv')}")

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": reg.coef_
}).sort_values(by="Coefficient", key=np.abs, ascending=False)

coef_df.to_csv(out_path("regression_coefficients.csv"), index=False)
print(f"Saved -> {out_path('regression_coefficients.csv')}")

metrics_df = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "NRMSE", "R2", "Adjusted R2", "Train Rows", "Test Rows", "Predictor Count"],
    "Value": [mae, rmse, nrmse, r2, adj_r2, len(X_train), len(X_test), X.shape[1]]
})
metrics_df.to_csv(out_path("regression_metrics.csv"), index=False)
print(f"Saved -> {out_path('regression_metrics.csv')}")

print("\nRegression Performance")
print(f"MAE        : {mae:.4f}")
print(f"RMSE       : {rmse:.4f}")
print(f"NRMSE      : {nrmse:.4f}")
print(f"R2         : {r2:.4f}")
print(f"Adjusted R2: {adj_r2:.4f}")

plt.figure(figsize=(8.5, 6.5))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolor="black", linewidth=0.4)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linewidth=2)

plt.title("Regression Model — Actual vs Predicted log_cost", fontsize=15, color=TITLE_COLOR, pad=12)
plt.xlabel("Actual log_cost", fontsize=12)
plt.ylabel("Predicted log_cost", fontsize=12)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="both", color=GRID_COLOR, linewidth=0.8)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(out_path("regression_actual_vs_predicted.png"), dpi=300)
plt.close()
print(f"Saved -> {out_path('regression_actual_vs_predicted.png')}")

top10 = coef_df.head(10).iloc[::-1]

plt.figure(figsize=(10, 6.5))
plt.barh(top10["Feature"], top10["Coefficient"])
plt.title("Top 10 Regression Coefficients by Magnitude", fontsize=15, color=TITLE_COLOR, pad=12)
plt.xlabel("Coefficient", fontsize=12)
plt.ylabel("Feature", fontsize=12)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(out_path("top10_regression_coefficients.png"), dpi=300)
plt.close()
print(f"Saved -> {out_path('top10_regression_coefficients.png')}")

print("\nPrimary analysis outputs generated successfully.")

# ============================================================
# EXTRA VALIDATION — CROSS-VALIDATION, VARIANCE, OUTLIERS, T-TEST
# ============================================================

print("\n" + "=" * 88)
print("EXTRA VALIDATION — CROSS-VALIDATION, VARIANCE, OUTLIERS, AND PAIRED T-TEST")
print("=" * 88)

# Base model without ABS
base_model_df = nt_df.copy()

available_categorical_cols_base = [c for c in CATEGORICAL_COLS if c in base_model_df.columns]

base_encoded = pd.get_dummies(
    base_model_df,
    columns=available_categorical_cols_base,
    drop_first=True,
    dtype=int
)

base_drop_cols = [
    COST_COL,
    "log_cost",
    "Project",
    "RefID",
    "Location",
    "Challenge / Opportunity",
    "Region_Presented"
]

y_base = base_encoded["log_cost"].copy()
drop_from_x_base = [c for c in base_drop_cols if c in base_encoded.columns]
X_base = base_encoded.drop(columns=drop_from_x_base, errors="ignore").copy()
X_base = X_base.select_dtypes(include=[np.number]).copy()

all_nan_cols_base = X_base.columns[X_base.isna().all()].tolist()
if all_nan_cols_base:
    X_base = X_base.drop(columns=all_nan_cols_base)

for col in X_base.columns:
    if X_base[col].isna().any():
        X_base[col] = X_base[col].fillna(X_base[col].median())

if len(X_base) != len(y):
    raise ValueError("Base model row count does not match target row count.")

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_base_r2 = cross_val_score(
    LinearRegression(),
    X_base,
    y_base,
    cv=kf,
    scoring="r2"
)

cv_full_r2 = cross_val_score(
    LinearRegression(),
    X,
    y,
    cv=kf,
    scoring="r2"
)

cv_compare_df = pd.DataFrame({
    "Fold": [f"Fold {i}" for i in range(1, 6)],
    "Base_Model_R2": cv_base_r2,
    "Full_Model_R2": cv_full_r2,
    "Improvement": cv_full_r2 - cv_base_r2
})
cv_compare_df.to_csv(out_path("cv_base_vs_full_r2.csv"), index=False)
print(f"Saved -> {out_path('cv_base_vs_full_r2.csv')}")

mean_cv_base = cv_base_r2.mean()
mean_cv_full = cv_full_r2.mean()
std_cv_full = cv_full_r2.std(ddof=1)
range_cv_full = cv_full_r2.max() - cv_full_r2.min()

# Paired t-test
t_stat, p_value = ttest_rel(cv_full_r2, cv_base_r2)

ttest_df = pd.DataFrame({
    "Metric": [
        "Mean CV R2 (Base)",
        "Mean CV R2 (Full)",
        "Std Dev CV R2 (Full)",
        "Range CV R2 (Full)",
        "Paired t-statistic",
        "Paired p-value"
    ],
    "Value": [
        mean_cv_base,
        mean_cv_full,
        std_cv_full,
        range_cv_full,
        t_stat,
        p_value
    ]
})
ttest_df.to_csv(out_path("paired_ttest_cv_summary.csv"), index=False)
print(f"Saved -> {out_path('paired_ttest_cv_summary.csv')}")

print("\nCross-Validation Summary")
print(f"Mean CV R2 (Base) : {mean_cv_base:.3f}")
print(f"Mean CV R2 (Full) : {mean_cv_full:.3f}")
print(f"Std Dev CV R2     : {std_cv_full:.3f}")
print(f"Range CV R2       : {range_cv_full:.3f}")
print(f"Paired t-stat     : {t_stat:.3f}")
print(f"Paired p-value    : {p_value:.3f}")

# 5-fold CV chart
x_pos = np.arange(5)
bar_width = 0.28

plt.figure(figsize=(10.5, 6.5))
bars1 = plt.bar(
    x_pos - bar_width / 2,
    cv_base_r2,
    width=bar_width,
    label="Base Model (no ABS)",
    color="#9FB7C9",
    edgecolor="none"
)

bars2 = plt.bar(
    x_pos + bar_width / 2,
    cv_full_r2,
    width=bar_width,
    label="Full Model (+ ABS)",
    color="#1F8A9B",
    edgecolor="none"
)

for bar, value in zip(bars1, cv_base_r2):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.002,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#16324F"
    )

for bar, value in zip(bars2, cv_full_r2):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.002,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#16324F"
    )

plt.title("5-Fold Cross-Validation — Generalisation Check", fontsize=16, color=TITLE_COLOR, pad=12)
plt.ylabel("R²", fontsize=12)
plt.xticks(x_pos, [f"Fold {i}" for i in range(1, 6)], fontsize=11)
plt.ylim(min(cv_base_r2.min(), cv_full_r2.min()) - 0.02, max(cv_base_r2.max(), cv_full_r2.max()) + 0.03)
plt.legend(frameon=False, fontsize=10)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#808080")
ax.spines["bottom"].set_color("#808080")
ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(out_path("cv_generalisation_check.png"), dpi=300)
plt.close()
print(f"Saved -> {out_path('cv_generalisation_check.png')}")

# Prediction error distribution
residuals = y_test - y_pred
absolute_errors = np.abs(residuals)

error_df = pd.DataFrame({
    "Actual_log_cost": y_test.values,
    "Predicted_log_cost": y_pred,
    "Residual": residuals,
    "Absolute_Error": absolute_errors
})
error_df.to_csv(out_path("prediction_error_distribution_data.csv"), index=False)
print(f"Saved -> {out_path('prediction_error_distribution_data.csv')}")

large_error_threshold = absolute_errors.mean() + 2 * absolute_errors.std(ddof=1)

plt.figure(figsize=(10, 6.5))
counts, bins, _ = plt.hist(
    absolute_errors,
    bins=20,
    color="#2A99A9",
    edgecolor="black",
    linewidth=0.6,
    alpha=0.9
)

mu_err = absolute_errors.mean()
sd_err = absolute_errors.std(ddof=1)
if sd_err > 0:
    x_curve = np.linspace(absolute_errors.min(), absolute_errors.max(), 400)
    y_curve = norm.pdf(x_curve, mu_err, sd_err)
    bin_width = bins[1] - bins[0]
    y_curve = y_curve * len(absolute_errors) * bin_width
    plt.plot(x_curve, y_curve, color="#1B3B7A", linewidth=1.8)

plt.axvline(
    large_error_threshold,
    color="red",
    linestyle="--",
    linewidth=1.3,
    label=f"Large error threshold = {large_error_threshold:.2f}"
)

plt.title("Prediction Error Distribution (Absolute Error) — Variance Check", fontsize=16, color=TITLE_COLOR, pad=12)
plt.xlabel("Absolute Error (|Actual − Predicted|) in log units", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend(frameon=False, fontsize=10)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#808080")
ax.spines["bottom"].set_color("#808080")
ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(out_path("prediction_error_variance_check.png"), dpi=300)
plt.close()
print(f"Saved -> {out_path('prediction_error_variance_check.png')}")

# Cook's distance
X_sm = sm.add_constant(X)
ols_model = sm.OLS(y, X_sm).fit()
influence = ols_model.get_influence()
cooks_d = influence.cooks_distance[0]

cooks_threshold = 4 / len(X)
cooks_sorted = np.sort(cooks_d)
obs_sorted = np.arange(1, len(cooks_sorted) + 1)

cooks_df = pd.DataFrame({
    "Observation_Sorted": obs_sorted,
    "Cooks_Distance_Sorted": cooks_sorted
})
cooks_df.to_csv(out_path("cooks_distance_sorted.csv"), index=False)
print(f"Saved -> {out_path('cooks_distance_sorted.csv')}")

plt.figure(figsize=(10, 6.5))
plt.scatter(
    obs_sorted,
    cooks_sorted,
    s=18,
    color="#2F7DE1",
    alpha=0.85,
    edgecolor="none"
)

plt.axhline(
    cooks_threshold,
    color="red",
    linestyle="--",
    linewidth=1.3,
    label=f"Influential threshold = {cooks_threshold:.4f}"
)

plt.yscale("log")
plt.title("Outlier Impact Check — Cook's Distance", fontsize=16, color=TITLE_COLOR, pad=12)
plt.xlabel("Observations (Sorted)", fontsize=12)
plt.ylabel("Cook's Distance (log scale)", fontsize=12)
plt.legend(frameon=False, fontsize=10)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#808080")
ax.spines["bottom"].set_color("#808080")
ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(out_path("cooks_distance_outlier_check.png"), dpi=300)
plt.close()
print(f"Saved -> {out_path('cooks_distance_outlier_check.png')}")

# t-distribution plot
df_t = len(cv_full_r2) - 1
x_t = np.linspace(-6, 6, 1000)
y_t_pdf = t.pdf(x_t, df=df_t)

alpha = 0.05
t_crit = t.ppf(1 - alpha / 2, df=df_t)

plt.figure(figsize=(9.5, 6))
plt.plot(x_t, y_t_pdf, color="#1B3B7A", linewidth=2)

plt.fill_between(x_t, y_t_pdf, where=(x_t <= -t_crit), color="#F4B000", alpha=0.35)
plt.fill_between(x_t, y_t_pdf, where=(x_t >= t_crit), color="#F4B000", alpha=0.35)

plt.axvline(t_stat, color="red", linestyle="--", linewidth=1.8, label=f"Observed t = {t_stat:.3f}")
plt.axvline(-t_crit, color="#2E8B57", linestyle=":", linewidth=1.5, label=f"± Critical t = {t_crit:.3f}")
plt.axvline(t_crit, color="#2E8B57", linestyle=":", linewidth=1.5)

plt.title("Paired t-Test — t Distribution", fontsize=16, color=TITLE_COLOR, pad=12)
plt.xlabel("t value", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend(frameon=False, fontsize=10)

ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#808080")
ax.spines["bottom"].set_color("#808080")
ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(out_path("paired_ttest_t_distribution.png"), dpi=300)
plt.close()
print(f"Saved -> {out_path('paired_ttest_t_distribution.png')}")

# Validation summary CSV
validation_summary_df = pd.DataFrame({
    "Metric": [
        "Mean CV R2 (Base Model)",
        "Mean CV R2 (Full Model)",
        "Std Dev CV R2 (Full Model)",
        "Range CV R2 (Full Model)",
        "Hold-out R2",
        "Paired t-statistic",
        "Paired p-value",
        "Large Error Threshold",
        "Max Cook's Distance",
        "Cook's Distance Threshold"
    ],
    "Value": [
        mean_cv_base,
        mean_cv_full,
        std_cv_full,
        range_cv_full,
        r2,
        t_stat,
        p_value,
        large_error_threshold,
        cooks_d.max(),
        cooks_threshold
    ]
})
validation_summary_df.to_csv(out_path("model_stability_variance_outlier_summary.csv"), index=False)
print(f"Saved -> {out_path('model_stability_variance_outlier_summary.csv')}")

print("\nStability / Variance / Outlier / T-test Summary")
print(f"Mean CV R2 (Base)       : {mean_cv_base:.3f}")
print(f"Mean CV R2 (Full)       : {mean_cv_full:.3f}")
print(f"Std Dev CV R2 (Full)    : {std_cv_full:.3f}")
print(f"Range CV R2 (Full)      : {range_cv_full:.3f}")
print(f"Hold-out R2             : {r2:.3f}")
print(f"Paired t-statistic      : {t_stat:.3f}")
print(f"Paired p-value          : {p_value:.3f}")
print(f"Large error threshold   : {large_error_threshold:.3f}")
print(f"Max Cook's Distance     : {cooks_d.max():.5f}")
print(f"Cook's Threshold        : {cooks_threshold:.5f}")

print("\nAll analysis and validation outputs generated successfully.")