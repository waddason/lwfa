import numpy as np

import torch
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import torch.optim as optim


class ObjectDetector:
    def __init__(self):
        # Initialize model with pretrained weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights)

        # Modify the first conv layer to accept 2 channels instead of 3
        # original_conv = self.model.backbone.body.conv1
        self.model.backbone.body.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Create new transform with 2-channel normalization
        min_size = 640
        max_size = 800  # very close to 798
        image_mean = [0.485, 0.456]
        # Only first 2 channels of original [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224]
        # Only first 2 channels of original [0.229, 0.224, 0.225]
        self.model.transform = GeneralizedRCNNTransform(
            min_size, max_size, image_mean, image_std
        )

        # Replace the classifier with a new one
        # for our number of classes (4 + background)
        num_classes = 5  # 4 classes + background
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.batch_size = 4

        # Class mapping
        self.int_to_cat = {
            0: "beam_from_ionisation",
            1: "laser_driven_wakefield",
            2: "beam_driven_wakefield", 
            3: "beam_from_background",
        }

    def prepare_data(self, X, y=None, batch_size=4):
        # create dataset class
        class LWFADataset(torch.utils.data.Dataset):
            def __init__(self, X, y=None, device=None):
                self.X = X
                self.y = y
                self.batch_size = batch_size
                self.device = device

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                # Get and process image
                img = self.X[idx]['data']
                img_tensor = torch.FloatTensor(img).to(self.device)
                if img_tensor.dim() == 2:
                    img_tensor = img_tensor.unsqueeze(0)
                    # Add channel dimension if missing
                img_height = img_tensor.shape[1]
                img_width = img_tensor.shape[2]

                if self.y is not None:
                    # Prepare target dict for training
                    boxes = []
                    labels = []
                    for box in self.y[idx]:
                        # Get box coordinates
                        x_center, y_center, width, height = box['bbox']

                        # Convert from [x_center, y_center, width, height]
                        # to [x1, y1, x2, y2]
                        x1 = (x_center - width/2) * img_width
                        y1 = (y_center - height/2) * img_height
                        x2 = (x_center + width/2) * img_width
                        y2 = (y_center + height/2) * img_height

                        # Add sanity checks
                        if x1 >= 0 and y1 >= 0 and x2 <= img_width and y2 <= img_height:
                            if x2 > x1 and y2 > y1:
                                boxes.append([x1, y1, x2, y2])
                                labels.append(box['class'] + 1)
                                # Add 1 since 0 is background in Faster R-CNN

                    # Create target dict
                    if boxes:
                        target = {
                            'boxes': torch.FloatTensor(boxes).to(self.device),
                            'labels': torch.tensor(
                                labels, dtype=torch.int64
                            ).to(self.device)
                        }
                    else:
                        # If no valid boxes, create empty target
                        target = {
                            'boxes': torch.FloatTensor(size=(0, 4)),
                            'labels': torch.tensor([], dtype=torch.int64)
                        }
                    return img_tensor, target
                return img_tensor

        # Create dataset
        dataset = LWFADataset(X, y, device=self.device)

        # Create data loader
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True if y is not None else False,
            collate_fn=lambda x: tuple(zip(*x)) if y is not None else torch.stack(x)
        )

        return data_loader

    def fit(self, X, y):
        # Prepare training data
        data_loader = self.prepare_data(X, y, batch_size=4)

        # Set model to training mode
        self.model.train()

        # Create optimizer
        optimizer = optim.SGD(self.model.parameters(),
                              lr=0.01,
                              momentum=0.9,
                              weight_decay=0.0005)
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95  # Reduce lr by 5% each epoch
        )

        # Training loop
        num_epochs = 1
        for epoch in range(num_epochs):
            epoch_loss = 0
            for images, targets in data_loader:
                optimizer.zero_grad()

                # Move input data to the device
                images = [image.to(self.device) for image in images]
                targets = [
                    {k: v.to(self.device) for k, v in t.items()}
                    for t in targets
                ]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                losses.backward()

                # Add gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                epoch_loss += losses.item()
            # Step the scheduler
            scheduler.step()

            # print(f'Epoch {epoch}: Loss = {epoch_loss/len(images)}')

        return self

    def predict(self, X):
        # Set model to evaluation mode
        self.model.eval()

        # Prepare data
        data_loader_test = self.prepare_data(X)

        predictions = []
        with torch.no_grad():
            for batch in data_loader_test:
                # batch shape is [B, C, H, W] where B=4 (batch_size)
                # Process each image in the batch separately
                for single_img in batch:
                    # Add batch dimension back
                    img = single_img.unsqueeze(0)  # [1, C, H, W]
                    pred = self.model(img)[0]
                    img_preds = []
                    # print(f"Boxes: {pred['boxes']}")
                    # print(f"Labels: {pred['labels']}")
                    # print(f"Scores: {pred['scores']}")
                    # Get image dimensions
                    img_height = img[0].shape[1]  # Height is dim 1
                    img_width = img[0].shape[2]   # Width is dim 2
                    for box, label, score in zip(
                        pred['boxes'], pred['labels'], pred['scores']
                    ):
                        if score > 0.25:  # Confidence threshold
                            # Convert box from pixels
                            # to normalized coordinates [0,1]
                            x1, y1, x2, y2 = box.cpu().numpy()
                            # Normalize coordinates
                            x1 = x1 / img_width
                            x2 = x2 / img_width
                            y1 = y1 / img_height
                            y2 = y2 / img_height
                            # Convert from [x1,y1,x2,y2]
                            # to [x_center,y_center,width,height]
                            width = x2 - x1
                            height = y2 - y1
                            x_center = x1 + width/2
                            y_center = y1 + height/2
                            pred_dict = {
                                'bbox': [x_center, y_center, width, height],
                                'class': int(label.cpu().numpy()) - 1,
                                'proba': float(score.cpu().numpy())
                            }
                            img_preds.append(pred_dict)

                    predictions.append(img_preds)

        return np.array(predictions, dtype=object)
