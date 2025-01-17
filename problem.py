import os
import sys
import pickle
import numpy as np
from sklearn.model_selection import GroupKFold
import re
from collections import Counter

from rampwf.workflows import ObjectDetector


class utils:
    """Utility functions helpful in the challenge."""

    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    from ramp_custom_lwfa.scoring import (
        ClassAveragePrecision,
        MeanAveragePrecision,
    )
    from ramp_custom_lwfa.predictions import make_custom_predictions
    from ramp_custom_lwfa.geometry import apply_NMS_to_predictions


problem_title = "Object Detection in Laser Wakefield Acceleration Simulation"

SCORING_IOU = 0.5

_event_label_names = [
    "beam_from_ionisation",
    "laser_driven_wakefield",
    "beam_driven_wakefield",
    "beam_from_background",
]

# Correspondence between categories and int8 categories
# Mapping int to categories
int_to_cat = {
    0: "beam_from_ionisation",
    1: "laser_driven_wakefield",
    2: "beam_driven_wakefield",
    3: "beam_from_background",
}
# Mapping categories to int
cat_to_int = {v: k for k, v in int_to_cat.items()}

_event_label_int = list(int_to_cat)


Predictions = utils.make_custom_predictions(iou_threshold=SCORING_IOU)
workflow = ObjectDetector()
score_types = [
    utils.MeanAveragePrecision(
        class_names=["beam_from_ionisation", "laser_driven_wakefield",
                     "beam_driven_wakefield", "beam_from_background"],
        weights=[1, 1, 1, 1],
        iou_threshold=SCORING_IOU,
    ),
    utils.ClassAveragePrecision(
        "beam_from_ionisation", iou_threshold=SCORING_IOU),
    utils.ClassAveragePrecision(
        "laser_driven_wakefield", iou_threshold=SCORING_IOU),
    utils.ClassAveragePrecision(
        "beam_driven_wakefield", iou_threshold=SCORING_IOU),
    utils.ClassAveragePrecision(
        "beam_from_background", iou_threshold=SCORING_IOU),
]


def _get_data(path=".", split="train"):
    """
    Get the data for the given split (train or test).

    This function reads the diagnostic data and 
    corresponding labels from the specified data path.
    It processes both diagnostic channels (diag0 and diag1) and their
    respective label files.

    Parameters:
    -----------
    path : str, optional
        The base path to the data directory. Default is current directory (".").
    split : str, optional
        The data split to retrieve, either "train" or "test". Default is "train".

    Returns:
    --------
    X : list
        A list of combined diagnostic data arrays, where each array
        has shape (2, height, width).
        The first channel corresponds to diag0 and the second to diag1.
    y : list
        A list of label dictionaries.
        Each dictionary contains bounding box coordinates
        and corresponding class labels for the objects in the image.
    """
    data_path = os.path.join(path, "data", split)
    X = []
    y = []

    # Initialize paths lists
    diag0_files_paths = []
    diag1_files_paths = []
    diag0_labels_paths = []
    diag1_labels_paths = []
    # Get all files in the directory
    all_files = os.listdir(data_path)
    # Sort files to ensure consistent ordering
    all_files.sort()
    for file in all_files:
        if file.endswith("_00.pkl"):
            diag0_path = os.path.join(data_path, file)
            diag1_path = os.path.join(data_path,
                                      file.replace("_00.pkl", "_01.pkl"))
            diag0_lbl_path = os.path.join(data_path,
                                          file.replace("_00.pkl", "_00.txt"))
            diag1_lbl_path = os.path.join(data_path,
                                          file.replace("_00.pkl", "_01.txt"))

            diag0_files_paths.append(diag0_path)
            diag1_files_paths.append(diag1_path)
            diag0_labels_paths.append(diag0_lbl_path)
            diag1_labels_paths.append(diag1_lbl_path)

    assert len(diag0_files_paths) == len(diag1_files_paths) == \
        len(diag0_labels_paths) == len(diag1_labels_paths), \
        "Mismatch in number of files and labels"

    for diag0, diag1, label_diag0, label_diag1 in zip(diag0_files_paths,
                                                      diag1_files_paths,
                                                      diag0_labels_paths,
                                                      diag1_labels_paths):
        # Load the diagnostic files
        with open(diag0, 'rb') as f:
            diag_0 = pickle.load(f)
        with open(diag1, 'rb') as f:
            diag_1 = pickle.load(f)

        diag_0_data = diag_0['data']
        diag_1_data = diag_1['data']

        # Stack the arrays along a new axis to create a 2-channel array
        combined_diag = np.stack([diag_0_data, diag_1_data], axis=0)

        # Create a structured array with data and identifier
        x_with_id = {
            'data': combined_diag,
            'id': (diag0, diag1)  # Using the file paths as identifier
        }

        # Add the data to X
        X.append(x_with_id)

        # Process labels for both diag0 and diag1
        example_annotations = []
        for label_file in [label_diag0, label_diag1]:
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.split()
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(
                            float, parts[1:5])
                        example_annotations.append({
                            'bbox': [x_center, y_center, width, height],
                            'class': class_id
                        })

        y.append(example_annotations)

    X = np.array(X)
    y = np.array(y, dtype=object)

    if os.environ.get("RAMP_TEST_MODE", False):
        # Launched with --quick-test option; only a small subset of the data
        # Extract simulation numbers from file paths
        groups = []
        for x in X:
            sim_num = re.match(r'.*/S(\d{3})_', x['id'][0]).group(1)
            groups.append(sim_num)

        # Count examples per group
        group_counts = Counter(groups)

        # Get 5 groups with most examples
        n_groups = 5
        groups_sorted_by_count = sorted(group_counts.items(),
                                        key=lambda x: x[1],
                                        reverse=True)
        top_groups = [group for group,
                      count in groups_sorted_by_count[:n_groups]]

        # Get the last 4 examples from each of the 5 groups
        quick_test_indices = []
        examples_per_group = 4  # 4 examples × 5 groups = 20 total
        for group in top_groups:
            group_indices = [i for i, g in enumerate(groups) if g == group]
            quick_test_indices.extend(group_indices[-examples_per_group:])

        # Select subset of data
        X = X[quick_test_indices]
        y = y[quick_test_indices]

    return X, y


def get_train_data(path="."):
    return _get_data(path, "train")


def get_test_data(path="."):
    return _get_data(path, "test")


def get_cv(X, y, random_state=42):
    """Get cross-validation splits using GroupKFold.

    Parameters
    ----------
    X : array-like
        Each element has an 'id' key containing file paths like
        './data/train/S000_002400_00.pkl'
    y : array-like
        Labels
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    cv_splits : generator
        Yields (train_idx, test_idx) for each fold
    """
    # Set random seed
    np.random.seed(random_state)

    # Extract simulation numbers from file paths
    groups = []
    for x in X:
        sim_num = re.match(rf'.*{os.sep}S(\d{{3}})_', x['id'][0]).group(1)
        groups.append(sim_num)

    # Convert to numpy array for consistent handling
    groups = np.array(groups)

    cv = GroupKFold(n_splits=5)

    return cv.split(X, y, groups=groups)
