''' Default '''
DATASET = "cracktile"
IMG_PROCESSING = "cracktile"
MODEL = "unet"

IMAGE_SIZE = (256,256,1)
FILTER = "*[0-9].*"

### Blue, Green, Red
BACKGROUND_COLOR = [255, 120, 110]
SEGMENTATION_COLOR = [0, 0, 255]

p_VALIDATION = 0.15
MIN_DELTA = 1e-5
PATIENCE = 2
### Monitor: loss, acc, val_loss, val_acc
MONITOR = "val_loss"

''' Folder '''
# > project level
dn_DATA = "dataset"
# >> data level
dn_TEST = "test"
dn_TRAIN = "train"
# >>> only train level
dn_AUGMENTATION = "aug"
# >>> test and train level
dn_IMAGE = "image"
dn_LABEL = "label"

# > project level
dn_OUT = "out"
# >> out level
dn_TOLABEL = "tolabel"

# > src level
dn_NN = "nn"
# >> NN level
dn_ARCH = "arch"

dn_DIP = "dip"
# >> DIP level
dn_PROCESSING = "processing"

dn_MODEL = "model"

''' File '''
# > out level
# %s : number file
fn_PREPROCESSING = "%s_1_preprocessing.png"
fn_PREDICT = "%s_2_predict.png"
fn_ORIGINAL = "%s_3_original.png"
fn_OVERLAY = "%s_4_overlay.png"
fn_SEGMENTATION = "results.txt"

# > src level
# >> model level
fn_CHECKPOINT = "checkpoint.hdf5"
fn_LOGGER = "logger.log"