import os

#
# path and dataset parameter
#

DATA_PATH = 'data'

LP_W_CAR_PATH = os.path.join(DATA_PATH, 'lp_with_car')

CACHE_PATH = os.path.join(LP_W_CAR_PATH, 'cache')

OUTPUT_DIR = os.path.join(LP_W_CAR_PATH, 'output')

WEIGHTS_DIR = os.path.join(LP_W_CAR_PATH, 'weights')

WEIGHTS_FILE = None
# WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'save.ckpt')

CLASSES = ['plate']

FLIPPED = True


#
# model parameter
#

IMAGE_SIZE = 448

CELL_SIZE = 7

BOXES_PER_CELL = 1

ALPHA = 0.1

DISP_CONSOLE = False

OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE = 2.0
COORD_SCALE = 5.0


#
# solver parameter
#

GPU = ''

LEARNING_RATE = 0.0001

DECAY_STEPS = 30000

DECAY_RATE = 0.1

STAIRCASE = True

BATCH_SIZE = 45

MAX_ITER = 25000

SUMMARY_ITER = 10

SAVE_ITER = 1000


#
# test parameter
#

THRESHOLD = 0.01

IOU_THRESHOLD = 0.5
