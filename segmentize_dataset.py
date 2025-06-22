import sys
import os
import supervision as sv
import numpy as np
import cv2
import COCO_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sam2')))
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

DATASET_SUBDIR = os.path.join("images", "train")
ANNOTATIONS_FILE = os.path.join("annotations", "instances_default.json")
IMAGES_PATH = os.path.join(os.curdir, "images")
ANNOTATIONS_PATH = os.path.join(os.curdir, ANNOTATIONS_FILE)

coco_data = COCO_utils.load_coco_json(ANNOTATIONS_PATH)
CLASSES = [category.name for category in coco_data.categories if category.supercategory != 'none']
IMAGES = [image.file_name for image in coco_data.images]

# run SAM inference
SAM2_CHECKPOINT = os.path.abspath("../sam2/checkpoints/sam2.1_hiera_large.pt")
SAM2_MODEL_CFG = os.path.abspath("../sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam2 = build_sam2(str(SAM2_MODEL_CFG), str(SAM2_CHECKPOINT), device=device, apply_postprocessing=False)
mask_predictor = SAM2ImagePredictor(sam2)

for image_name in IMAGES:
    EXAMPLE_IMAGE_PATH = os.path.join(IMAGES_PATH, image_name)

    annotations = COCO_utils.COCOJsonUtility.get_annotations_by_image_path(coco_data=coco_data, image_path=image_name)
    ground_truth = COCO_utils.COCOJsonUtility.annotations2detections(annotations=annotations)

    # small hack - coco numerate classes from 1, model from 0 + we drop first redundant class from coco json
    ground_truth.class_id = ground_truth.class_id - 1
    # So in this dataset fire is class 3, smoke is class 2, background is 0, 1, 4, 5

    image_bgr = cv2.imread(EXAMPLE_IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # initiate annotator
    # box_annotator = sv.BoxAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX)
    # mask_annotator = sv.MaskAnnotator(color=sv.Color.WHITE, color_lookup=sv.ColorLookup.INDEX)

    # annotated_ground_truth = box_annotator.annotate(scene=image_bgr.copy(), detections=ground_truth)

    mask_predictor.set_image(image=image_rgb)

    # prepare output mask
    final_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    pairs = list(zip(ground_truth.xyxy, ground_truth.class_id))
    pairs_rsorted = sorted(pairs, key=lambda x: x[1], reverse=True) # since it's reverse sorted,
    # fire bboxes are always iterated over first, then smoke
    for i, (bbox, class_id) in enumerate(pairs_rsorted):
        match class_id:
            case 3:  # smoke
                class_id = 1
            case 4:  # fire
                class_id = 0
            case default:
                continue

        masks, scores, logits = mask_predictor.predict(box = bbox, multimask_output=True)
        mask = masks[np.argmax(scores)].astype(np.bool_)  # take the best mask
        final_mask[mask] = class_id + 1

    mask_save_path = EXAMPLE_IMAGE_PATH.replace("images", "masks", count=1).replace(".jpg", ".png")
    os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
    cv2.imwrite(mask_save_path, final_mask * 127)
    print(f"Processed {image_name} and saved mask to {mask_save_path}")

# [masks, scores, logits] = mask_predictor.predict(box=ground_truth.xyxy[1], multimask_output=True)
# masks = masks.astype(np.bool_)
# detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks), mask=masks, confidence=scores)

# annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

# sv.plot_images_grid(
#     images = [annotated_ground_truth, annotated_image],
#     grid_size= (1, 2),
#     titles = ["Source", "Segmented Image"]
# )