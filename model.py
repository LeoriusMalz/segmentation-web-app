import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

import torch
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights, MaskRCNN

class Predict():
    def __init__(self, image_path, score_threshold=0.6):
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        self.score_threshold = score_threshold

        transform = transforms.Compose([transforms.ToTensor()])
        self.image_tensor = transform(self.image)

    def find(self, model=maskrcnn_resnet50_fpn_v2(weights='DEFAULT', weights_backbone='DEFAULT')):
        if model.training: model.eval()
        with torch.no_grad():
            self.preds = model([self.image_tensor])[0]

    def valide(self):
        self.boxes = self.preds['boxes'][self.preds['scores'] > self.score_threshold].numpy()
        self.labels = self.preds['labels'][self.preds['scores'] > self.score_threshold].numpy()
        self.masks = self.preds['masks'][self.preds['scores'] > self.score_threshold].numpy()

        return self.labels.shape[0]

    def point(self, x, y, mask_threshold=0.5):
        for i in range(self.labels.shape[0]):
            mask = self.masks[i][0]
            if mask[x, y] > mask_threshold:
                return self.vis(i, mask_threshold)

        return self.default()

    def vis(self, num, mask_threshold=0.5):
        image_copy = self.image.copy()

        box = self.boxes[num]
        mask = self.masks[num]
        label = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta["categories"][self.labels[num]]

        color = np.random.rand(3) * 255
        mask = mask[0]
        image_copy[mask > mask_threshold] = image_copy[mask > mask_threshold] * 0.5 + color * 0.5

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)

        cv2.putText(image_copy, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.5/(1280/image_copy.shape[0]), color, 
                    max(1, round(5/(1280/image_copy.shape[0]))))

        return image_copy

    def default(self):
        return self.image

    def change_threshold(self, val):
        self.score_threshold = val

    def get_threshold(self):
        return self.score_threshold
