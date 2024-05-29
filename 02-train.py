import os
from pathlib import Path
from src.utils import *
from rich.console import Console
from rich import pretty
from rich.table import Table
import pyarrow as pa
import pyarrow.parquet as pq
import joblib
from lightgbm import LGBMClassifier
import datetime
import polars as pl

def str_in_list(string, query_list):
    '''
    function checks if `string` is contained in any `query_list` elements 
    '''
    return any(element in string for element in query_list)

# Formating and loggers 
############################################################
pretty.install()
console = Console()
# Console stuff
console.rule("Running lightgbm Training")

# configuration 
############################################################
config = read_yaml("config.yaml")
console.print("Using configuration from `config.yaml` file:")
console.print(config)

# Training Data 
############################################################
data_folder = "data"
root_folder = os.path.join(data_folder,"24-hour_recordings")
all_feature_parquets = list(Path(root_folder).glob('**/*/*features.parquet'))
console.log(f"Found {len(all_feature_parquets)} feature files in [blue bold]{root_folder}[/blue bold]")
# Read and concatenate Parquet files
console.log("Reading features")
tables = [pq.read_table(path) for path in all_feature_parquets]
console.log("Concatenating features")
df = pa.concat_tables(tables)

# Handling missing data
console.log("Checking for NAs")
total_nas=pl.from_arrow(df).with_columns(pl.all().is_null()).to_numpy().sum()
console.log(f"Found {total_nas} NAs in the dataset")

# Extracting 'stage' column for labels and dropping it from features
console.log("Extracting labels and dropping `label` column from features")
y = df.column("label")
df = df.drop(["label"])

console.log("Counting data inbalance")
stage_counts = y.value_counts()
total_predictions = sum(stage_counts.field(1).to_pylist())
# Mapping sleep stages
stage_mapping = {4: "REM", 2: "NREM", 0: "Awake"}

# Create Rich table to display counts
table = Table(show_header=True, header_style="bold blue")
table.add_column("Sleep Stage", style="dim", width=12)
table.add_column("Count", justify="right")
table.add_column("Frac", justify="right")

# Iterate over StructArray correctly
for record in stage_counts.to_pylist():
    stage_name = stage_mapping.get(record['values'], "Unknown")
    count = record['counts']
    table.add_row(stage_name, str(count), str(round(count/total_predictions, 2)))

# Display the table
console.print(table)

# Training hyperparams
##########################################################
# Define hyper-parameters
params = dict(
    boosting_type='gbdt',
    n_estimators=400,
    max_depth=5,
    num_leaves=90,
    colsample_bytree=0.5,
    importance_type='gain',
)

from sklearn.utils.class_weight import compute_class_weight
# compute class weights
# these will be in the order 
# 0: "Wake", 2: "NREM", 4:"REM"
# This takes a while so it was saved for later purposes
if config['recompute_imbalance']:
  numeric_weights = compute_class_weight(
    class_weight ='balanced',
    classes=np.unique(y), 
    y=y)
else:
  numeric_weights = [0.6464329605333303,
                     0.7942784646041191,
                     5.153446964613251
                    ]
  console.print("Using previously calculated class imbalance", style="yellow")

class_weight_dict = dict(zip([0, 2, 4], numeric_weights))
console.print("Check numeric weights", style='yellow')
console.print(class_weight_dict)
params["class_weight"] = class_weight_dict


# Column selections
cols_all = df.column_names
cols_eeg = [col for col in cols_all if col.startswith('eeg_')]
cols_emg = [col for col in cols_all if col.startswith('emg_')]
console.rule("Feature Set")
console.log(f"Complete set of features contains {len(cols_all)} features.")
console.log(f"EEG features: {len(cols_eeg)}. EMG features: {len(cols_emg)}")

# Define predictors, 
# excluding certain features for importance testing
exclude_features = {
    'full': [], # all predictors included
    'no_kurt': ['kurt'],
    'no_perm': ['perm'],
    'no_std': ['std'],
    'no_log_rms': ['log_rms']
}

console.rule("Feature Exclusion")
console.log("Combinations of these feature selections will be trained")
console.print(exclude_features)

X_all = {}
for key, exclusions in exclude_features.items():
    console.print(f"Excluding columns contained in {exclusions}")
    included_eeg = [col for col in cols_eeg if not str_in_list(col, exclusions)]
    included_emg = [col for col in cols_emg if not str_in_list(col, exclusions)]
    X_all[f'eeg_{key}'] = df.select(included_eeg)
    X_all[f'eeg+emg_{key}'] = df.select(included_eeg + included_emg)

console.rule("Model Training")
console.log(f"A total of {len(X_all.keys())} models will be trained, see below")
console.print(X_all.keys())


outdir = Path("output/classifiers")
outdir.mkdir(parents=True, exist_ok=True)


# Parallel processing when building the trees.
params['n_jobs'] = 8

# Training loop using rich Progress
from rich.progress import Progress
# Training loop with rich Progress
with Progress() as progress:
    task1 = progress.add_task("[cyan]Preparing to train models...", total=len(X_all))
    for name, X in X_all.items():
        # Update the task description to include the current model
        progress.update(task1, description=f"[magenta]Training {name}...")
        model_dir = outdir / name
        model_dir.mkdir(exist_ok=True)

        X_pd = X.to_pandas()
        clf = LGBMClassifier(**params)
        clf.fit(X_pd, y)
        model_path = model_dir / "model.joblib"
        joblib.dump(clf, model_path, compress=True)

        feature_importances = pd.Series(clf.feature_importances_, index=X_pd.columns, name='Importance')
        feature_importances = feature_importances.sort_values(ascending=False)

        # Convert Series to DataFrame for better control over column names
        importances_df = feature_importances.reset_index()
        importances_df.columns = ['Feature', 'Importance']

        # Save to CSV
        importances_df.to_csv(model_dir / "feature_importances.csv", index=False)

        metadata = {
            "training_dt": datetime.datetime.now().isoformat("T"),
            "model_parameters": params,
            "training_accuracy": clf.score(X_pd, y)
        }
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
        progress.update(task1, advance=1)

console.log("All models trained and saved successfully.")