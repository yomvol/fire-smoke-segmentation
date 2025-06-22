# Fire and smoke semantic segmentation

My experiment with image segmentation to get a feel of PyTorch. The goal is to teach a neural network to spot and separate fire and smoke regions from everything else in a photo -- think of it as a module for safety cameras or environmental monitoring.

I've got my hands on a small dataset of outdoor fires (bounding boxes in COCO format) and got ground truth masks by running bboxes through Segment Anything Model by Meta. DeepLabV3+ was picked as the segmentation model since it performs well even on transparent smoke and lightweight enough to train locally. Pretrained ResNet encoder was chosen for simplicity. Then the model is fine-tuned to recognize fire and smoke regions.

## Results
Still tuning, but already getting pretty solid results. Introduced data augmentation and it helped a bit. The dataset could benefit from enrichment and some manual cleaning (generated masks sometimes have artifacts like false positive smoke)

| Metric         | Fire   | Smoke  | Mean   |
|----------------|--------|--------|--------|
| Precision      | 0.8076 | 0.7281 |   -    |
| Recall         | 0.8109 | 0.7473 |   -    |
| F1 Score       |   -    |   -    | 0.8428 |
| mAP@50         |   -    |   -    | 0.8362 |
| Dataset IoU    | 0.8920 | 0.8198 | 0.8559 |