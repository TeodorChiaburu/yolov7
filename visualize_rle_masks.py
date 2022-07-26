"""Script to visualize the bboxes and seg. masks that detectron 2 will receive for training."""

import os, json, cv2
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer

# Wildbee dataset
DATASET_ROOT = "../../data/data_lstudio"
ANN_ROOT = "../beexplainable/metafiles/Bees_Christian/22_species"
TRAIN_PATH = os.path.join(DATASET_ROOT, "Bees_Christian_train")
VAL_PATH = os.path.join(DATASET_ROOT, "Bees_Christian_val")
TRAIN_JSON = os.path.join(ANN_ROOT, "bees_coco_train_1.json")
VAL_JSON = os.path.join(ANN_ROOT, "bees_coco_val_1.json")

# Open json file
with open(TRAIN_JSON, 'r') as f:
    annot_file = json.load(f)

# Pick a sample
ind = 6
file_name = annot_file['images'][ind]['file_name']
rle = annot_file['annotations'][ind]['segmentation']
bbox = annot_file['annotations'][ind]['bbox']

# Read image and visualize annotations
img = cv2.imread(TRAIN_PATH + '/' + file_name)
visualizer = Visualizer(img)
out = visualizer.overlay_instances(masks=[rle], boxes=[bbox])
plt.imshow(out.get_image())
plt.show()