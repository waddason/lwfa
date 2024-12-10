import numpy as np


def find_matching_bbox(prediction, list_of_true_values, iou_threshold):
    """Find the index of the bounding box that is closest to the prediction.

    Parameters
    ----------
    prediction : dict
        with keys "bbox" and "class". The bbox is in format
        [x_center, y_center, width, height].
        This is the reference bounding box against which we are comparing.
    list_of_true_values : list of dict
        same keys as prediction. Each dict contains bbox coordinates in format
        [x_center, y_center, width, height] and class label.

    Returns
    -------
    index : int
        index in the ``list_of_true_values`` array of the bounding box that
        is the closest to the ``prediction``.
    success : bool
        True if the IoU between the prediction and highest scoring true_value
        is higher than the ``iou_threshold``

    """
    predicted_bbox = np.array(prediction["bbox"]).reshape(1, 4)
    all_true_bbox = np.array(
        [value["bbox"] for value in list_of_true_values]
    ).reshape(len(list_of_true_values), 4)

    ious = compute_iou(predicted_bbox, all_true_bbox)[0, :]
    is_different_class = np.array(
        [value["class"] != prediction["class"]
         for value in list_of_true_values]
    )
    ious[is_different_class] = 0

    index, maximum = np.argmax(ious), np.max(ious)
    return index, maximum > iou_threshold


def xywh_to_x1y1x2y2(box):
    """Convert [x_center, y_center, width, height] to [x1, y1, x2, y2]"""
    x_center, y_center, width, height = box
    x1 = x_center - width/2
    y1 = y_center - height/2
    x2 = x_center + width/2
    y2 = y_center + height/2
    return np.array([x1, y1, x2, y2])


def compute_iou(boxes1, boxes2):
    """Computes pairwise IOU matrix for given two sets of boxes

    IOU : Intersection over Union.
    0 < IOU < 1

    Doc: https://github.com/rafaelpadilla/Object-Detection-Metrics

    Parameters
    ----------
    boxes1 : np.array of shape `(N, 4)`
        these represent bounding boxes
        where each box is of the format `[x1, y1, x2, y2]`.
    boxes2 : np.array shape `(M, 4)`
        same format as ``boxes1``

    Returns
    -------
    iou_matrix : np.array of shape `(N, M)`
        pairwise IOU matrix with shape `(N, M)`, where the value at i_th row
        j_th column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.

    """
    # Convert from [x,y,w,h] to [x1,y1,x2,y2]
    boxes1_x1y1x2y2 = np.array([xywh_to_x1y1x2y2(box) for box in boxes1])
    boxes2_x1y1x2y2 = np.array([xywh_to_x1y1x2y2(box) for box in boxes2])

    # Original IoU computation
    lu = np.maximum(boxes1_x1y1x2y2[:, None, :2], boxes2_x1y1x2y2[:, :2])
    rd = np.minimum(boxes1_x1y1x2y2[:, None, 2:], boxes2_x1y1x2y2[:, 2:])
    intersection = np.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]

    boxes1_area = boxes1[:, 2] * boxes1[:, 3]  # width * height for boxes1
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]  # width * height for boxes2

    union_area = np.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    return np.clip(intersection_area / union_area, 0.0, 1.0)


def apply_NMS_to_predictions(y_pred, iou_threshold=0.5):
    """Apply NMS to predictions in RAMP format.

    Parameters
    ----------
    y_pred : list of list of dict
        Each dict contains 'bbox', 'class', and 'proba' keys
    iou_threshold : float
        IoU threshold for considering boxes as overlapping

    Returns
    -------
    filtered_y_pred : list of list of dict
        Predictions after NMS
    """
    filtered_y_pred = np.empty(len(y_pred), dtype=object)

    for i, predictions in enumerate(y_pred):
        if not predictions:
            filtered_y_pred[i] = []
            continue

        # Group predictions by class
        class_groups = {}
        for pred in predictions:
            class_id = pred['class']
            if class_id not in class_groups:
                class_groups[class_id] = {'boxes': [], 'scores': [], 'preds': []}
            class_groups[class_id]['boxes'].append(pred['bbox'])
            class_groups[class_id]['scores'].append(pred['proba'])
            class_groups[class_id]['preds'].append(pred)

        # Apply NMS per class
        kept_predictions = []
        for class_preds in class_groups.values():
            if class_preds['boxes']:
                boxes = np.array(class_preds['boxes'])
                scores = np.array(class_preds['scores'])

                # Sort by score
                order = scores.argsort()[::-1]
                boxes = boxes[order]
                scores = scores[order]

                keep = []
                while order.size > 0:
                    # Keep highest scoring box
                    keep.append(order[0])

                    if order.size == 1:
                        break

                    # Compute IoU of the kept box with the rest
                    ious = compute_iou(
                        boxes[0:1],  # Current highest scoring box
                        boxes[1:]    # Remaining boxes
                    )[0]

                    # Keep boxes with IoU less than threshold
                    inds = np.where(ious <= iou_threshold)[0]
                    order = order[inds + 1]
                    boxes = boxes[inds + 1]
                    scores = scores[inds + 1]

                # Add kept predictions
                for idx in keep:
                    kept_predictions.append(class_preds['preds'][idx])

        filtered_y_pred[i] = kept_predictions

    return filtered_y_pred
