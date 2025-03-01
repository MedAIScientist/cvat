# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import cv2
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_image_predictor import sam_model_registry


def convert_mask_to_polygon(mask):
    contours = None
    if int(cv2.__version__.split('.')[0]) > 3:
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[0]
    else:
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[1]

    contours = max(contours, key=lambda arr: arr.size)
    if contours.shape.count(1):
        contours = np.squeeze(contours)
    if contours.size < 3 * 2:
        raise Exception('Less then three point have been detected. Can not build a polygon.')

    polygon = []
    for point in contours:
        polygon.append([int(point[0]), int(point[1])])

    return polygon

class ModelHandler:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#        self.sam_checkpoint = "/opt/nuclio/sam/sam_vit_h_4b8939.pth"
#        self.model_type = "vit_h"
        self.latest_image = None
#        sam_model = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
#        sam_model.to(device=self.device)
        self.predictor =  SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

    def handle(self, image):
        self.predictor.set_image(np.array(image))
        features = self.predictor.get_image_embedding()
        return features
