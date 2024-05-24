import numpy as np
import pandas as pd
from src.utils import *
import os
from pathlib import Path
from rich import pretty
from rich.console import Console
import time


pretty.install()
console = Console()
# Console stuff
console.rule("Running lightgbm Training")

# configuration 
############################################################
config = read_yaml("config.yaml")
console.print("Using configuration from `config.yaml` file:")
console.print(config)

# Testing Data 
############################################################
data_folder = "data"
root_folder = os.path.join(data_folder,"4-hour_recordings")
all_paths = list(Path(data_folder).glob('4-hour_recordings/*/**'))
all_dirs = sorted([folder for folder in all_paths if folder.is_dir() and "Day" in str(folder)])
console.print(f"Total recordings {len(all_dirs)}")

# Finding classifiers
#############################################################
classifier_dir = Path("output/classifiers")
classifier_paths = list(classifier_dir.glob('**/*.joblib')) 
console.print(f"Classifiers to evaluate: {len(classifier_paths)}")
console.print(classifier_paths)

results_dir = Path("output/results")
results_dir.mkdir(parents=True, exist_ok=True)

# Evaluate 
############################################################
from rich.progress import Progress
# Training loop with rich Progress
with Progress() as progress:
    task1 = progress.add_task("[cyan]Preparing to evaluate classifiers...", total=len(classifier_paths))

    for classifier_path in classifier_paths:
        progress.update(task1, description=f"[cyan]Evaluating {classifier_path.parts[-2]} ")
        classifier_results_dir = results_dir / classifier_path.stem  # Create a subdirectory for each classifier
        classifier_results_dir.mkdir(exist_ok=True)
        
        # Dictionaries to hold results
        results = {
            'accuracy': [],
            'confusion_matrix': [],
            'cohen_kappa': [],
            'prediction_time': []
        }

        task2 = progress.add_task(f"[magenta]Evaluating mice", total=len(all_dirs))
        
        for mouse_folder in all_dirs:
            progress.update(task2, description=f"[magenta]Evaluating mouse {mouse_folder}")
            start_time = time.time()
            sls, true_labels = predict_mouse(mouse_folder, config, classifier_path)
            end_time = time.time()
            
            accuracy, cm, cohen = evaluate(sls._predicted, true_labels)
            
            # Append results
            results['accuracy'].append(accuracy)
            results['confusion_matrix'].append(cm)
            results['cohen_kappa'].append(cohen)
            results['prediction_time'].append((end_time - start_time) / len(true_labels))

            progress.advance(task2)
            del sls, true_labels  # Clean up

        # Save all results in a single .npz file
        np.savez_compressed(classifier_results_dir / "evaluation_metrics.npz", **results)

        console.print(f"Results saved for [blue]{classifier_path.name}[/blue] in [blue]{classifier_results_dir}[/blue]", style='green')
        progress.advance(task1)

console.log("All models evaluated and results saved successfully.", style="bold on green")