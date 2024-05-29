import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from yasa import hypno_int_to_str
import pandas as pd
classifier_dir = Path("output/classifiers")
classifier_paths = list(classifier_dir.glob('**/*.joblib')) 
for classifier_path in classifier_paths:
    clf_name = classifier_path.parts[-2]
    evaluation_path = f'output/results/{clf_name}/evaluation_metrics.npz'
    evaluation_metrics = np.load(evaluation_path)
    
    # Confusion matrix
    cm = np.mean(evaluation_metrics['confusion_matrix'], axis = 0)
    cm_labels = hypno_int_to_str(np.array([0, 2, 4]))
    ax=plt.subplot()
    sns.heatmap(cm/np.sum(cm), annot=True, 
                fmt='.2%', cmap='Blues')
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels') 
    ax.set_title(f'Confusion Matrix clf: {clf_name}') 
    ax.xaxis.set_ticklabels(cm_labels) 
    ax.yaxis.set_ticklabels(cm_labels)
    plt.show()


    # accuracy & cohen
    accuracy_df = pd.DataFrame({
        "accuracy": evaluation_metrics['accuracy'],
        'cohen_kapa': evaluation_metrics['cohen_kappa']
        })
    sns.swarmplot(x = 'variable', y  = 'value', data=accuracy_df.melt())
    plt.title(f"Accuracy Evaluation for {clf_name}")
    plt.xlabel("")
    plt.show()