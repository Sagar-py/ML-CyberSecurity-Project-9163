## detecting outliers in the model
## Inspired from http://cs.ucsb.edu/~bolunwang


import os
import sys
import time
import utilities
import numpy as np
from keras.preprocessing import image

MODEL_NAME = str(sys.argv[1])
assert MODEL_NAME in ('sunglasses', 'anonymous_1', 'anonymous_2', 'multi_trigger_multi_target')

CONFIG = utilities.load_config()

RESULT_DIR = CONFIG['result_dir']
SAVE_DIR = RESULT_DIR + '/' + MODEL_NAME
IMG_FILENAME_TEMPLATE = f'{MODEL_NAME}_%s_label_%d.png'

# input size
IMG_ROWS = CONFIG['img_rows']
IMG_COLS = CONFIG['img_cols']
IMG_COLOR = CONFIG['img_color']
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)

NUM_CLASSES = CONFIG['num_classes']


def outlier_detection(l1_norm_list, idx_mapping):
    consistency_constant = 1.4826  # if normal distribution
    median = np.median(l1_norm_list)
    mad = consistency_constant * np.median(np.abs(l1_norm_list - median))
    min_mad = np.abs(np.min(l1_norm_list) - median) / mad

    print('median: %f, MAD: %f' % (median, mad))
    print('anomaly index: %f' % min_mad)

    flag_list = []
    for y_label in idx_mapping:
        if l1_norm_list[idx_mapping[y_label]] > median:
            continue
        if np.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad > 2:
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))

    if len(flag_list) > 0:
        flag_list = sorted(flag_list, key=lambda x: x[1])

    print('flagged label list: %s' %
          ', '.join(['%d: %2f' % (y_label, l_norm)
                     for y_label, l_norm in flag_list]))


def analyze_pattern_norm_dist():
    mask_flatten = []
    idx_mapping = {}

    for y_label in range(NUM_CLASSES):
        mask_filename = IMG_FILENAME_TEMPLATE % ('mask', y_label)
        if os.path.isfile('%s/%s' % (SAVE_DIR, mask_filename)):
            img = image.load_img(
                '%s/%s' % (SAVE_DIR, mask_filename),
                color_mode='grayscale',
                target_size=INPUT_SHAPE)
            mask = image.img_to_array(img)
            mask /= 255
            mask = mask[:, :, 0]

            mask_flatten.append(mask.flatten())

            idx_mapping[y_label] = len(mask_flatten) - 1

    l1_norm_list = [np.sum(np.abs(m)) for m in mask_flatten]

    print('%d labels found' % len(l1_norm_list))

    outlier_detection(l1_norm_list, idx_mapping)


if __name__ == '__main__':
    print('%s start' % sys.argv[0])

    start_time = time.time()
    analyze_pattern_norm_dist()
    elapsed_time = time.time() - start_time
    print('elapsed time %.2f s' % elapsed_time)
