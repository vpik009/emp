import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


csv_path = "metrics.csv"  # path to the CSV file with the model metrics
metrics = ["val_brier-minFDE6", "val_MR", "val_minFDE6", "val_minADE6"]
metric_labels = ["Brier-FDE6", "MR", "FDE6", "ADE6"]
method_colors = {"RKD": "green", "Hint": "red", "TDD": "blue"}
model_layout = ["Tiny thin", "Tiny wide", "Small thin", "Small wide"]

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
df["Baseline"] = df["Baseline"].astype(bool)

rows = []

for model in df["Model"].unique():
    if model == "EMPD":
        continue  # dont plot the EMPD model   

    tdd_baseline = (
        df[(df["Model"] == model) & (df["Method"] == "TDD") & (df["Baseline"])].groupby("Model").tail(1)
    )
    baseline_vals = tdd_baseline.iloc[0][metrics]

    # compare all nonbaseline methods for the same model
    for method in df["Method"].unique():
        if pd.isna(method):
            continue  # none (nan) is reserved for EMPD - we skip this model in this plot
        nonbaseline = df[
            (df["Model"] == model)
            & (df["Method"] == method)
            & (~df["Baseline"])
        ]
        nonbaseline_vals = nonbaseline.iloc[0][metrics]
        pct_change = ((nonbaseline_vals - baseline_vals) / baseline_vals) * 100 * -1
        rows.append({"Model": model, "Method": method, **pct_change.to_dict()})

df_pct = pd.DataFrame(rows)

# use a 2x2 grid for the grouped bar chart
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
axes = axes.flatten()

for idx, model in enumerate(model_layout):
    ax = axes[idx]
    model_df = df_pct[df_pct["Model"] == model]
    x = np.arange(len(metrics))
    bar_width = 0.25

    for i, method in enumerate(method_colors.keys()):
        method_row = model_df[model_df["Method"] == method]
        if method_row.empty:
            values = [0] * len(metrics)
        else:
            values = method_row[metrics].values.flatten()
        ax.bar(x + i * bar_width, values, width=bar_width,
               color=method_colors[method], label=method if idx == 0 else None)

    ax.set_title(model)
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(metric_labels, rotation=45)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylabel("% improvement from TDD Baseline")


# plot legends on top
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()
