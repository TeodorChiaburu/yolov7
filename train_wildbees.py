"""

Training script the wildbees dataset

"""

import os
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import (
    default_argument_parser,
    launch,
)
from detectron2.data.datasets.coco import register_coco_instances

# Which one should be used??
#from train_det import Trainer, setup
from train_inseg import Trainer, setup
#from train_transformer import Trainer, setup


def register_custom_datasets():
    # ADD YOUR DATASET CONFIG HERE
    # dataset names registered must be unique, different than any of the above

    # Wildbee dataset
    DATASET_ROOT = "../../data/data_lstudio"
    ANN_ROOT = "../beexplainable/metafiles/Bees_Christian/22_species"
    TRAIN_PATH = os.path.join(DATASET_ROOT, "Bees_Christian_train")
    VAL_PATH = os.path.join(DATASET_ROOT, "Bees_Christian_val")
    TRAIN_JSON = os.path.join(ANN_ROOT, "bees_coco_train_1.json")
    VAL_JSON = os.path.join(ANN_ROOT, "bees_coco_val_1.json")
    register_coco_instances("wildbees_train", {}, TRAIN_JSON, TRAIN_PATH)
    register_coco_instances("wildbees_val", {}, VAL_JSON, VAL_PATH)

register_custom_datasets()


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
