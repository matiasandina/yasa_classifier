import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from yasa import hypno_int_to_str
import pandas as pd
from rich.console import Console
from matplotlib.ticker import FuncFormatter

# Function to format the tick labels in scientific notation
def scientific_notation(x, pos):
    return f'{x/1e5:.0f}x$10^5$'  # Adjust this formatting based on your specific range and preference


console = Console()
classifier_dir = Path("output/classifiers")
classifier_paths = sorted(list(classifier_dir.glob('**/*.joblib')))
n_classifiers = len(classifier_paths)

# Setup for grids
rows, cols = 2, 5  # Adjust as needed for your number of classifiers

# Figure for Confusion Matrices
fig_cm, axes_cm = plt.subplots(rows, cols, figsize=(20, 10))  # Customize the size as needed

# Figure for Accuracy & Cohen Kappa
fig_acc, axes_acc = plt.subplots(rows, cols, figsize=(20, 10))  # Customize the size as needed

# Let's assume the new subplot for feature importances is axes_fi
fig_fi, axes_fi = plt.subplots(rows, cols, figsize=(30, 10))

for i, classifier_path in enumerate(classifier_paths):
    row = i // cols
    col = i % cols
    clf_name = classifier_path.parts[-2]
    console.print(f"[blue] Evaluating {clf_name} results")
    evaluation_path = f'output/results/{clf_name}/evaluation_metrics.npz'
    evaluation_metrics = np.load(evaluation_path)
    console.print(f'[green] {clf_name} metrics loaded')
    feature_importance_path = classifier_dir / clf_name / "feature_importances.csv"
    console.print(f'[blue] Loading {clf_name} Feature Importance')
    # Load feature importances and select top 10
    feature_importances = pd.read_csv(feature_importance_path)
    top_features = feature_importances.nlargest(10, 'Importance')

    # Plotting feature importances
    ax_fi = axes_fi[row, col]
    sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax_fi, orient='h')
    ax_fi.set_title(f'Top 10 Features: {clf_name}')
    ax_fi.set_xlabel('Importance')
    ax_fi.set_ylabel('Features')
    # Apply the formatter to the x-axis
    ax_fi.xaxis.set_major_formatter(FuncFormatter(scientific_notation))

    # Confusion matrix
    cm = np.mean(evaluation_metrics['confusion_matrix'], axis=0)
    cm_labels = hypno_int_to_str(np.array([0, 2, 4]))

    # Plotting confusion matrix
    ax_cm = axes_cm[row, col]
    sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel('Predicted labels')
    ax_cm.set_ylabel('True labels')
    ax_cm.set_title(f'CM: {clf_name}')
    ax_cm.xaxis.set_ticklabels(cm_labels)
    ax_cm.yaxis.set_ticklabels(cm_labels)

    # Accuracy & Cohen Kappa plotting
    accuracy_df = pd.DataFrame({
        "accuracy": evaluation_metrics['accuracy'],
        'cohen_kappa': evaluation_metrics['cohen_kappa']
    })

    ax_acc = axes_acc[row, col]
    sns.swarmplot(x='variable', y='value', data=accuracy_df.melt(), ax=ax_acc)
    ax_acc.set_title(f"Metrics: {clf_name}")
    ax_acc.set_xlabel("")
    ax_acc.set_ylabel("Value")

fig_cm.tight_layout()
plt.show(block=False)
plt.pause(0.001)  # Small pause to update figuresinput("PRESS ENTER FOR NEXT PLOT")
fig_acc.tight_layout()
plt.show(block=False)
plt.pause(0.001)  # Small pause to update figures
fig_fi.tight_layout()
plt.show(block=False)
plt.pause(0.001)  # Small pause to update figures
input("PRESS [ENTER] to Exit")

"output/classifiers/eeg+emg_full/feature_importances.csv"