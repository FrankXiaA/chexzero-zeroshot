import zero_shot
import preprocess_padchest
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
import sys
sys.path.append('../')
import eval
from eval import evaluate, bootstrap
from zero_shot import make, make_true_labels, run_softmax_eval


"""

# Directory containing your .png files
directory = 'C:/Users/26984/PycharmProjects/pythonProject/Github/CheXzero-main/CheXzero-main/data/padchest_sample2'

# List to hold file paths
png_files = []

# Loop through each file in the directory
for filename in os.listdir(directory):
    # Check if the file is a .png
    if filename.endswith('.png'):
        # Construct the full file path using os.path.join
        file_path = os.path.join(directory, filename)
        # Normalize the path to use consistent backslashes
        normalized_path = os.path.normpath(file_path)
        # Append the normalized file path to the png_files list
        png_files.append(normalized_path)
preprocess_padchest.img_to_h5(
png_files,
'C:/Users/26984/PycharmProjects/pythonProject/Github/CheXzero-main/CheXzero-main/data/h5'
)

"""


# ----- DIRECTORIES ------ #
cxr_filepath: str = 'C:/Users/26984/PycharmProjects/pythonProject/Github/CheXzero-main/CheXzero-main/data/chexpert_test.h5' # filepath of chest x-ray images (.h5)
cxr_true_labels_path: Optional[str] = 'C:/Users/26984/PycharmProjects/pythonProject/Github/CheXzero-main/CheXzero-main/data/groundtruth.csv' # (optional for evaluation) if labels are provided, provide path
model_dir: str = 'C:/Users/26984/PycharmProjects/pythonProject/Github/CheXzero-main/CheXzero-main/checkpoints' # where pretrained models are saved (.pt)
predictions_dir: Path = Path('C:/Users/26984/PycharmProjects/pythonProject/Github/CheXzero-main/CheXzero-main/predictions') # where to save predictions
cache_dir: str = predictions_dir / "cached" # where to cache ensembled predictions

context_length: int = 77

# ------- LABELS ------  #
# Define labels to query each image | will return a prediction for each label
cxr_labels: List[str] = ['Atelectasis','Cardiomegaly',
                                      'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                                      'Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia',
                                      'Pneumothorax', 'Support Devices']

# ---- TEMPLATES ----- #
# Define set of templates | see Figure 1 for more details
cxr_pair_template: Tuple[str,str] = ("{}", "no {}")

# ----- MODEL PATHS ------ #
# If using ensemble, collect all model paths
model_paths = ['C:/Users/26984/PycharmProjects/pythonProject/Github/CheXzero-main/CheXzero-main/checkpoints/best_64_0.0001_original_35000_0.864.pt']



## Run the model on the data set using ensembled models
def ensemble_models(
        model_paths: List[str],
        cxr_filepath: str,
        cxr_labels: List[str],
        cxr_pair_template: Tuple[str],
        cache_dir: str = None,
        save_name: str = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Given a list of `model_paths`, ensemble model and return
    predictions. Caches predictions at `cache_dir` if location provided.

    Returns a list of each model's predictions and the averaged
    set of predictions.
    """

    predictions = []
    model_paths = sorted(model_paths)  # ensure consistency of
    for path in model_paths:  # for each model
        model_name = Path(path).stem

        # load in model and `torch.DataLoader`
        model, loader = make(
            model_path=path,
            cxr_filepath=cxr_filepath,
        )

        # path to the cached prediction
        if cache_dir is not None:
            if save_name is not None:
                cache_path = Path(cache_dir) / f"{save_name}_{model_name}.npy"
            else:
                cache_path = Path(cache_dir) / f"{model_name}.npy"

        # if prediction already cached, don't recompute prediction
        if cache_dir is not None and os.path.exists(cache_path):
            print("Loading cached prediction for {}".format(model_name))
            y_pred = np.load(cache_path)
        else:  # cached prediction not found, compute preds
            print("Inferring model {}".format(path))
            y_pred = run_softmax_eval(model, loader, cxr_labels, cxr_pair_template)
            if cache_dir is not None:
                Path(cache_dir).mkdir(exist_ok=True, parents=True)
                np.save(file=cache_path, arr=y_pred)
        predictions.append(y_pred)

    # compute average predictions
    y_pred_avg = np.mean(predictions, axis=0)

    return predictions, y_pred_avg

predictions, y_pred_avg = ensemble_models(
    model_paths=model_paths,
    cxr_filepath=cxr_filepath,
    cxr_labels=cxr_labels,
    cxr_pair_template=cxr_pair_template,
    cache_dir=cache_dir,
)

# save averaged preds
pred_name = "chexpert_preds.npy" # add name of preds
predictions_dir = predictions_dir / pred_name
np.save(file=predictions_dir, arr=y_pred_avg)

test_pred = y_pred_avg
print(len(test_pred))
test_true = make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)
print(len(test_true))
# evaluate model
cxr_results: pd.DataFrame = eval.evaluate(test_pred, test_true, cxr_labels) # eval on full test datset


a = np.load(predictions_dir)
print(a[0])
