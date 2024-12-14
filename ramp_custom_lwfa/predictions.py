import numpy as np
from rampwf.prediction_types.detection import Predictions as DetectionPredictions


def make_custom_predictions(iou_threshold):
    """Create class CustomPredictions using iou_threshold when bagging."""

    class CustomPredictions(DetectionPredictions):
        @classmethod
        def combine(cls, predictions_list, index_list=None):
            """Combine multiple predictions into a single one."""
            if index_list is None:
                index_list = range(len(predictions_list))

            # Get predictions arrays
            y_pred_list = [predictions_list[i].y_pred for i in index_list]
            n_images = len(y_pred_list[0])
            combined_predictions = []

            # Combine predictions for each image
            for i in range(n_images):
                all_preds = []
                for y_pred in y_pred_list:
                    if i < len(y_pred) and y_pred[i] is not None:
                        if isinstance(y_pred[i], list):
                            all_preds.extend(y_pred[i])
                        else:
                            all_preds.extend(y_pred[i].tolist())
                combined_predictions.append(all_preds)

            # Apply NMS if we have any predictions
            if any(len(preds) > 0 for preds in combined_predictions):
                from .geometry import apply_NMS_to_predictions
                filtered_predictions = apply_NMS_to_predictions(
                    combined_predictions, iou_threshold=iou_threshold
                )
            else:
                filtered_predictions = combined_predictions

            return cls(y_pred=np.array(filtered_predictions, dtype=object))

    return CustomPredictions
