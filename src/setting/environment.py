import setting.constant as const
import importlib
import os

def setup(args):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if args.gpu else "-1"

    const.MODEL = args.arch
    const.DATASET = args.dataset
    const.IMG_PROCESSING = args.dip

    const.fn_CHECKPOINT = ("%s_%s_%s" % (const.MODEL, const.DATASET, const.fn_CHECKPOINT))
    const.fn_LOGGER = ("%s_%s_%s" % (const.MODEL, const.DATASET, const.fn_LOGGER))

    arch = importlib.import_module("%s.%s.%s" % (const.dn_NN, const.dn_ARCH, const.MODEL))
    const.IMAGE_SIZE = arch.IMAGE_SIZE

    print("\n##################")
    print("Dataset:", const.DATASET)
    print("DIP:\t", const.IMG_PROCESSING)
    print("Arch:\t", const.MODEL)
    print("##################\n")