"""
Copyright Zexian Zeng's lab, AAIS, Peking Universit. All Rights Reserved

@author: Yufeng He
"""

from detectron2.config import CfgNode as CN

def add_DISSECT_config(cfg):
    """
    Add config for DISSECT
    """
    cfg.MODEL.DISSECT = CN()
    cfg.MODEL.DISSECT.NUM_CLASSES = 80
    cfg.MODEL.DISSECT.NUM_PROPOSALS = 500

    # RCNN Head.
    cfg.MODEL.DISSECT.NHEADS = 8
    cfg.MODEL.DISSECT.DROPOUT = 0.0
    cfg.MODEL.DISSECT.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DISSECT.ACTIVATION = 'relu'
    cfg.MODEL.DISSECT.HIDDEN_DIM = 256
    cfg.MODEL.DISSECT.NUM_CLS = 1
    cfg.MODEL.DISSECT.NUM_REG = 3
    cfg.MODEL.DISSECT.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.DISSECT.NUM_DYNAMIC = 2
    cfg.MODEL.DISSECT.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.DISSECT.CLASS_WEIGHT = 0.5
    cfg.MODEL.DISSECT.GIOU_WEIGHT = 2.0
    cfg.MODEL.DISSECT.L1_WEIGHT = 5.0
    cfg.MODEL.DISSECT.DEEP_SUPERVISION = True
    cfg.MODEL.DISSECT.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.DISSECT.USE_FOCAL = True
    cfg.MODEL.DISSECT.USE_FED_LOSS = False
    cfg.MODEL.DISSECT.ALPHA = 0.25
    cfg.MODEL.DISSECT.GAMMA = 2.0
    cfg.MODEL.DISSECT.PRIOR_PROB = 0.01

    # Dynamic K
    cfg.MODEL.DISSECT.OTA_K = 5

    # Diffusion
    cfg.MODEL.DISSECT.SNR_SCALE = 2.0
    cfg.MODEL.DISSECT.SAMPLE_STEP = 1

    # Inference
    cfg.MODEL.DISSECT.USE_NMS = True

    # Swin Backbones
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.SIZE = 'B'  # 'T', 'S', 'B'
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    cfg.MODEL.SWIN.OUT_FEATURES = (0, 1, 2, 3)  # modify

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # TTA.
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800, 800)
    cfg.TEST.AUG.CVPODS_TTA = True
    cfg.TEST.AUG.SCALE_FILTER = True
    cfg.TEST.AUG.SCALE_RANGES = ([96, 10000], [96, 10000], 
                                 [64, 10000], [64, 10000],
                                 [64, 10000], [0, 10000],
                                 [0, 10000], [0, 256],
                                 [0, 256], [0, 192],
                                 [0, 192], [0, 96],
                                 [0, 10000])
