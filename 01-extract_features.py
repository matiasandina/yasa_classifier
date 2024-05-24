import os
from pathlib import Path
from src.utils import *
import logging
from rich.logging import RichHandler
from rich.console import Console
from rich import pretty

# Formating and loggers 
############################################################
pretty.install()
console = Console()
# Console stuff
FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger("rich")
console.rule("Running Feature Extraction")

# configuration 
############################################################
config = read_yaml("config.yaml")
console.print("Using configuration from `config.yaml` file:")
console.print(config)

# Training Data Validation
############################################################
data_folder = "data"
root_folder = os.path.join(data_folder,"24-hour_recordings")
# Get the training data
p = Path(root_folder)
all_paths = list(Path(root_folder).glob('**/*'))
# This is for the 24 hour
all_dirs = sorted([folder for folder in all_paths if folder.is_dir() and "cycle" in str(folder)])
log.info(f"found {len(all_dirs)} directories in [blue bold]{root_folder}[/blue bold]")
if config["validate_lengths"]:
    log.info(f"Validating file lengths")
    report_validation_results(validate_file_lengths(all_dirs, 512, 2.5))
else:
    log.warning(f"File length validation not performed. Errors will appear if file lenghts do not match")

# Feature extraction
#################################################################


# If we wanted to run custom bands we can use here
custom_bands =  [
          (0.4, 1, 'sdelta'), 
          (1, 4, 'fdelta'), 
          (4, 8, 'theta'),
          (8, 12, 'alpha'), 
          (12, 16, 'sigma'), 
          (16, 30, 'beta')
      ]

for directory in all_dirs:
    process_data(directory, config)

log.info(f"[bold on green]Finished Feature Extraction[/bold on green]")
console.rule()
