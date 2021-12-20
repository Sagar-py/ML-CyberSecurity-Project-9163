## An example demonstrating visualization 
## code inspired from Bolun Wang 
## http://cs.ucsb.edu/~bolunwang

import os
import sys
import time

import random
import numpy as np

from tensorflow import set_random_seed
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

import utils
from visualizer import Visualizer


MODEL_NAME = str(sys.argv[1])
assert MODEL_NAME in ('anonymous_1', 'anonymous_2', 'multi_trigger_multi_target', 'sunglasses')

CONFIG = utils.load_config()

RANDOM_SEED = CONFIG['random_seed']
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
set_random_seed(RANDOM_SEED)

DEVICE = CONFIG['gpu_device']

DATA_DIR = CONFIG['data_dir']
MODEL_DIR = CONFIG['model_dir']
RESULT_DIR = CONFIG['result_dir']
SAVE_DIR = RESULT_DIR + '/' + MODEL_NAME
DATA_FILE = CONFIG['test_data_file']

MODEL_FILENAME = f'{MODEL_NAME}_bd_net.h5'
IMG_FILENAME_TEMPLATE = f'{MODEL_NAME}_%s_label_%d.png'
Y_TARGET = 0

# input size
IMG_ROWS = CONFIG['img_rows']
IMG_COLS = CONFIG['img_cols']
IMG_COLOR = CONFIG['img_color']
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)

NUM_CLASSES = CONFIG['num_classes']

INTENSITY_RANGE = CONFIG['visualize']['intensity_range']

# parameters for optimization
BATCH_SIZE = CONFIG['visualize']['batch_size']
LR = CONFIG['visualize']['learning_rate']
STEPS = CONFIG['visualize']['steps']
NB_SAMPLE = CONFIG['visualize']['num_mini_batch']
MINI_BATCH = NB_SAMPLE // BATCH_SIZE
INIT_COST = CONFIG['visualize']['init_cost']

REGULARIZATION = CONFIG['visualize']['regularization']

ATTACK_SUCC_THRESHOLD = CONFIG['visualize']['attack_suc_thres']
PATIENCE = 5
COST_MULTIPLIER = 2
SAVE_LAST = CONFIG['visualize']['save_last']

EARLY_STOP = CONFIG['visualize']['early_stop']
EARLY_STOP_THRESHOLD = CONFIG['visualize']['early_stop_thres']
EARLY_STOP_PATIENCE = 5 * PATIENCE

## the following part is not used in our application
## but the implementation also supports super-pixel mask
UPSAMPLE_SIZE = CONFIG['visualize']['upsample_size']
MASK_SHAPE = np.ceil(np.array(INPUT_SHAPE[0:2], dtype=float) / UPSAMPLE_SIZE)
MASK_SHAPE = MASK_SHAPE.astype(int)


def build_data_loader(X, Y):
    datagen = ImageDataGenerator()
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE)
    return generator


def visualize_trigger_w_mask(visualizer, gen, y_target,
                             save_pattern_flag=True):
    visualize_start_time = time.time()

    # initializing with random mask
    pattern = np.random.random(INPUT_SHAPE) * 255.0
    mask = np.random.random(MASK_SHAPE)

    # executing reverse engineering
    pattern, mask, mask_upsample, logs = visualizer.visualize(gen=gen, y_target=y_target, pattern_init=pattern, mask_init=mask)

    # meta data about the generated mask
    print('pattern, shape: %s, min: %f, max: %f' %
          (str(pattern.shape), np.min(pattern), np.max(pattern)))
    print('mask, shape: %s, min: %f, max: %f' %
          (str(mask.shape), np.min(mask), np.max(mask)))
    print('mask norm of label %d: %f' %
          (y_target, np.sum(np.abs(mask_upsample))))

    visualize_end_time = time.time()
    print('visualization cost %f seconds' %
          (visualize_end_time - visualize_start_time))

    if save_pattern_flag:
        save_pattern(pattern, mask_upsample, y_target)

    return pattern, mask_upsample, logs


def save_pattern(pattern, mask, y_target):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    img_filename = (
            '%s/%s' % (SAVE_DIR,
                       IMG_FILENAME_TEMPLATE % ('pattern', y_target)))
    utils.dump_image(pattern, img_filename, 'png')

    img_filename = (
            '%s/%s' % (SAVE_DIR,
                       IMG_FILENAME_TEMPLATE % ('mask', y_target)))
    utils.dump_image(np.expand_dims(mask, axis=2) * 255,
                     img_filename,
                     'png')

    fusion = np.multiply(pattern, np.expand_dims(mask, axis=2))
    img_filename = (
            '%s/%s' % (SAVE_DIR,
                       IMG_FILENAME_TEMPLATE % ('fusion', y_target)))
    utils.dump_image(fusion, img_filename, 'png')


def visualize_label_scan_bottom_right_white_4():
    print('loading dataset')
    X_test, Y_test = utils.data_loader('%s/%s' % (DATA_DIR, DATA_FILE))
    # transform numpy arrays into data generator
    test_generator = build_data_loader(X_test, Y_test)

    print('loading model')
    model_file = '%s/%s' % (MODEL_DIR, MODEL_FILENAME)
    model = load_model(model_file)

    # initialize visualizer
    visualizer = Visualizer(
        model, intensity_range=INTENSITY_RANGE, regularization=REGULARIZATION,
        input_shape=INPUT_SHAPE,
        init_cost=INIT_COST, steps=STEPS, lr=LR, num_classes=NUM_CLASSES,
        mini_batch=MINI_BATCH,
        upsample_size=UPSAMPLE_SIZE,
        attack_succ_threshold=ATTACK_SUCC_THRESHOLD,
        patience=PATIENCE, cost_multiplier=COST_MULTIPLIER,
        img_color=IMG_COLOR, batch_size=BATCH_SIZE, verbose=2,
        save_last=SAVE_LAST,
        early_stop=EARLY_STOP, early_stop_threshold=EARLY_STOP_THRESHOLD,
        early_stop_patience=EARLY_STOP_PATIENCE)

    log_mapping = {}

    # y_label list to analyze
    y_target_list = list(range(NUM_CLASSES))
    y_target_list.remove(Y_TARGET)
    y_target_list = [Y_TARGET] + y_target_list
    for y_target in y_target_list:
        print('processing label %d' % y_target)

        _, _, logs = visualize_trigger_w_mask(
            visualizer, test_generator, y_target=y_target,
            save_pattern_flag=True)

        log_mapping[y_target] = logs


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    utils.fix_gpu_memory()
    visualize_label_scan_bottom_right_white_4()


if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print('elapsed time %s s' % elapsed_time)
