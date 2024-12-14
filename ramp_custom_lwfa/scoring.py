import numpy as np

from rampwf.score_types.base import BaseScoreType
from .geometry import compute_iou


class ClassAveragePrecision(BaseScoreType):
    """Compute average precision of predictions for one class.

    Example
    -------
    >>> X_train, y_train = problem.get_train_data()
    >>> y_pred = model.predict(X_train)
    >>> metric = ClassAveragePrecision(class_name="Secondary",
                                    iou_threshold=problem.SCORING_IOU)
    >>> metric(y_train, y_pred)
    0.823

    """

    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0
    worst = 0.0

    def __init__(self, class_name, iou_threshold):
        cat_to_int = {'beam_from_ionisation': 0,
                      'laser_driven_wakefield': 1,
                      'beam_driven_wakefield': 2,
                      'beam_from_background': 3
                      }
        self.name = f"AP {class_name}"
        self.precision = 3

        self.class_id = cat_to_int[class_name]
        self.iou_threshold = iou_threshold

    def __call__(self, y_true, y_pred):

        precision, recall, _ = precision_recall_for_class(
            y_true, y_pred, self.class_id, self.iou_threshold
        )
        return average_precision(precision, recall)


class MeanAveragePrecision(BaseScoreType):
    """Compute mean of (average precision of predictions for one class).

    Example
    -------
    >>> X_train, y_train = problem.get_train_data()
    >>> y_pred = model.predict(X_train)
    >>> metric = ClassAveragePrecision(iou_threshold=problem.SCORING_IOU)
    >>> metric(y_train, y_pred)
    0.823

    """

    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0
    worst = 0.0

    def __init__(self, class_names, weights, iou_threshold):
        self.name = "mean AP"
        self.precision = 3

        # Convert class names to integers
        cat_to_int = {'beam_from_ionisation': 0,
                      'laser_driven_wakefield': 1,
                      'beam_driven_wakefield': 2,
                      'beam_from_background': 3
                      }
        self.class_ids = [cat_to_int[name] for name in class_names]
        if weights is None:
            weights = [1 for _ in class_names]
        self.weights = weights
        self.iou_threshold = iou_threshold

    def __call__(self, y_true, y_pred):

        mean_AP = 0
        for class_id, weight in zip(self.class_ids, self.weights):
            precision, recall, _ = precision_recall_for_class(
                y_true, y_pred, class_id, self.iou_threshold
            )
            mean_AP += weight * average_precision(precision, recall)
        mean_AP /= sum(self.weights)
        return mean_AP


#def average_precision(precision, recall):
#    """Compute average precision from precision and recall values."""    
#    # Compute AP as area under PR curve using trapezoidal rule
#    ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
#    return ap


def average_precision(precision, recall):
    """Compute average precision from precision and recall values.
    
    Implementation follows VOC metric, which:
    1. Uses all recall points
    2. Interpolates precision by taking maximum over all higher recall levels
    """
    # Make precision monotonically decreasing
    precision = np.concatenate([[0], precision, [0]])
    recall = np.concatenate([[0], recall, [1]])
    
    # Compute maximum precision for recall levels >= current recall
    for i in range(len(precision)-2, -1, -1):
        precision[i] = max(precision[i], precision[i+1])
    
    # Find points where recall changes
    i = np.where(recall[1:] != recall[:-1])[0]
    
    # Sum area under PR curve
    ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])
    
    return ap


def precision_recall_for_class(y_true, y_pred, class_id, iou_threshold):
    """Compute precision and recall for a specific class."""

    # Extract ground truth if it's in a predictions object
    if hasattr(y_true, 'y_pred'):
        y_true = y_true.y_pred

    # Convert to numpy arrays if needed
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true, dtype=object)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred, dtype=object)

    # Initialize counters
    n_true = 0
    true_positives = []
    scores = []

    for i, (gt, pred) in enumerate(zip(y_true, y_pred)):
        # Ensure we have lists to work with
        if not isinstance(gt, list):
            gt = []
        if not isinstance(pred, list):
            pred = []

        # Filter boxes by class
        gt_class = []
        for box in gt:
            if isinstance(box, dict) and box.get('class') == class_id:
                gt_class.append(box)

        pred_class = []
        for box in pred:
            if isinstance(box, dict):
                # Handle both string and integer class values
                pred_class_val = box.get('class')
                if isinstance(pred_class_val, str):
                    pred_class_val = int(pred_class_val)
                if pred_class_val == class_id:
                    pred_class.append(box)

        # Update total number of ground truth boxes
        n_true += len(gt_class)

        # Sort predictions by confidence score
        pred_class = sorted(
            pred_class, key=lambda x: x.get('proba', 0), reverse=True
        )

        # For each prediction, check if it matches a ground truth box
        for pred_box in pred_class:
            scores.append(pred_box.get('proba', 0))

            # Find best matching ground truth box
            best_iou = 0
            best_gt_idx = None

            for i, gt_box in enumerate(gt_class):
                iou = compute_iou(
                    np.array([pred_box['bbox']]), 
                    np.array([gt_box['bbox']])
                )[0, 0]

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            # If we found a match above the threshold
            if best_iou >= iou_threshold:
                true_positives.append(1)
                # Remove the matched ground truth box
                if best_gt_idx is not None:
                    gt_class.pop(best_gt_idx)
            else:
                true_positives.append(0)

    # Handle case where no predictions were made
    if not scores:
        return np.array([1.]), np.array([0.]), np.array([])

    # Convert to numpy arrays
    true_positives = np.array(true_positives)
    scores = np.array(scores)

    # Sort by score
    sort_idx = np.argsort(scores)[::-1]
    true_positives = true_positives[sort_idx]
    scores = scores[sort_idx]

    # Compute cumulative sum of true positives
    tp_cumsum = np.cumsum(true_positives)

    # Compute precision and recall
    precision = tp_cumsum / np.arange(1, len(tp_cumsum) + 1)
    recall = tp_cumsum / n_true if n_true > 0 else np.zeros_like(tp_cumsum)

    # Add the (1,0) point to precision-recall curve
    precision = np.concatenate([[1.], precision])
    recall = np.concatenate([[0.], recall])

    return precision, recall, scores
