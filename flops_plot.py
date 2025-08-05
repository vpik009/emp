import pandas as pd
import matplotlib.pyplot as plt


csv_path = "metrics.csv"  # path to the CSV file with the model metrics
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()


x_col = "GFlops"
y_col = "val_brier-minFDE6"
method_colors = {"RKD": "green", "Hint": "red", "TDD": "blue"}
model_markers = {
    "Small wide": "o",   # circle
    "Small thin": "s",   # square
    "Tiny wide": "D",    # diamond
    "Tiny thin": "^"     # triangle
}


df["Baseline"] = df["Baseline"].astype(bool)
df_nonbase = df[~df["Baseline"]]
df_base = df[df["Baseline"]]


fig, ax = plt.subplots(figsize=(10, 6))

# plot nonbaseline points
for _, row in df_nonbase.iterrows():
    method = row["Method"]
    model = row["Model"]
    color = method_colors.get(method, "gray")
    marker = model_markers.get(model, "o")
    ax.scatter(row[x_col], row[y_col], s=100, color=color, marker=marker, edgecolor="black")

# plot the baseline points (use TDD baseline)
# df_base_last = df_base.groupby("Model").tail(1)

for _, row in df_base.iterrows():  # iter through models and get TDD baseline
    model = row["Model"]
    ax.scatter(row[x_col], row[y_col], s=100, color=method_colors[row["Method"]], marker=".", edgecolor="black", alpha=0.4)
 
ax.set_xlabel("Inference FLOPs (G)")
ax.set_ylabel(y_col)
ax.grid(True, linestyle='--', alpha=0.7)

# legends
method_handles = [
    plt.Line2D([], [], marker='o', color='w', markerfacecolor=color,
               markeredgecolor="black", markersize=10, label=method)
    for method, color in method_colors.items()
]

baseline_handle = plt.Line2D([], [], marker='.', color='w', markerfacecolor="grey",
                             markeredgecolor="black", markersize=10, label="Baseline", alpha=0.4)

model_handles = [
    plt.Line2D([], [], marker=marker, color='w', markerfacecolor="gray",
               markeredgecolor="black", markersize=10, label=model)
    for model, marker in model_markers.items()
]

legend1 = ax.legend(handles=method_handles + [baseline_handle], title="Method",
                    loc="center left", bbox_to_anchor=(1.05, 0.7))
ax.add_artist(legend1)
ax.legend(handles=model_handles, title="Model",
          loc="center left", bbox_to_anchor=(1.05, 0.3))

fig.subplots_adjust(right=0.8)  # make sure the legend fits into the plot area
plt.show()
