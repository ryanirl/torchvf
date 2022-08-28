from ml_collections import config_dict


def get_config():
    cfg = config_dict.ConfigDict()

    cfg.DEVICE = "cuda:0"

    cfg.MODEL_TYPE = "h1"

    cfg.CONFIG_DIR = "training/bpcis_worm.py"
    cfg.WEIGHT_DIR = "../weights/worm/"

    cfg.EPOCHS = 1000
    cfg.BATCH_SIZE = 2
    cfg.SAVE_EVERY = 50
    cfg.EVAL_EVERY = 5
    cfg.IMAGE_EVERY = 10

    cfg.LR = 0.0002

    cfg.PRETRAINED = False
    cfg.PRETRAINED_DIR = None

    cfg.DATA = config_dict.ConfigDict()

    cfg.DATA.DIR = "../data/bpcis/"
    cfg.DATA.C = 1
    cfg.DATA.VF = True
    cfg.DATA.VF_DELIM = "_vf_10_11"
#    cfg.DATA.VF_DELIM = "_affinity_11"
    cfg.DATA.SPLIT = "worm_train"
    cfg.DATA.TRANSFORMS = "train"
    cfg.DATA.REMOVE = None
    cfg.DATA.COPY = None

    cfg.LOG = config_dict.ConfigDict()
    cfg.LOG.EVERY = 5
        
    cfg.LOSS = config_dict.ConfigDict()

    cfg.LOSS.IVP     = config_dict.ConfigDict()
    cfg.LOSS.MSE     = config_dict.ConfigDict()
    cfg.LOSS.TVERSKY = config_dict.ConfigDict()
    cfg.LOSS.BCE     = config_dict.ConfigDict()

    cfg.LOSS.IVP.APPLY = True
    cfg.LOSS.IVP.DX = 0.25
    cfg.LOSS.IVP.STEPS = 8
    cfg.LOSS.IVP.SOLVER = "euler"

    cfg.LOSS.MSE.APPLY = True

    cfg.LOSS.TVERSKY.APPLY = True
    cfg.LOSS.TVERSKY.ALPHA = 0.50
    cfg.LOSS.TVERSKY.BETA = 0.50
    cfg.LOSS.TVERSKY.FROM_LOGITS = True

    cfg.LOSS.BCE.APPLY = False
    
    return cfg
    




