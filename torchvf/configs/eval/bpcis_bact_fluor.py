from ml_collections import config_dict


def get_config():
    cfg = config_dict.ConfigDict()

    cfg.DEVICE = "cuda:0"

    cfg.MODEL_TYPE = "h1"

    cfg.CONFIG_DIR = "./configs/eval/bpcis_bact_fluor.py"

    cfg.MODEL_DIR  = "../weights/bact_fluor/h1/model.pth"
    cfg.SAVE_DIR = "../weights/bact_fluor/h1/"

    cfg.BATCH_SIZE = 1

    cfg.DATA = config_dict.ConfigDict()
    cfg.DATA.DIR = "../data/bpcis/"
    cfg.DATA.C = 1
    cfg.DATA.VF = False
    cfg.DATA.SPLIT = "bact_fluor_test"
    cfg.DATA.TRANSFORMS = "eval"
    cfg.DATA.REMOVE = [3, 5]
    cfg.DATA.COPY = None

    cfg.TILE = config_dict.ConfigDict()
    cfg.TILE.TYPE = "dynamic_overlap"
    cfg.TILE.SIZE = (256, 256)
    cfg.TILE.OVERLAP = 50
    cfg.TILE.BATCH_SIZE = 2

    cfg.SEMANTIC = config_dict.ConfigDict()
    cfg.SEMANTIC.THRESH = 0.5

    cfg.IVP = config_dict.ConfigDict()
    cfg.IVP.STEPS = 25
    cfg.IVP.DX = 0.1
    cfg.IVP.SOLVER = "euler"
    cfg.IVP.INTERP = "bilinear"

    cfg.CLUSTERING = config_dict.ConfigDict()
    cfg.CLUSTERING.EPS = 2.1
    cfg.CLUSTERING.MIN_SAMPLES = 15
    cfg.CLUSTERING.SNAP_NOISE = False

    return cfg





