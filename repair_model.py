import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_model_optimization as tfmot

import utilities

MODEL_NAME = str(sys.argv[1])
assert MODEL_NAME in ('sunglasses', 'anonymous_1', 'anonymous_2', 'multi_trigger_multi_target')

CONFIG = utilities.load_config()

DEVICE = CONFIG['gpu_device']

DATA_DIR = CONFIG['data_dir']
MODEL_DIR = CONFIG['model_dir']

VAL_DATA_FILE = CONFIG['val_data_file']
TEST_DATA_FILE = CONFIG['test_data_file']
POI_DATA_FILE = CONFIG[f'{MODEL_NAME}_data_file']

MODEL_FILENAME = f'{MODEL_NAME}_bd_net.h5'
WEIGHT_FILENAME = f'{MODEL_NAME}_bd_weights.h5'
OUTPUT_FILENAME = f'{MODEL_NAME}_pruned_net.h5'
REPAIR_FILENAME = f'{MODEL_NAME}_repair_net.h5'

BATCH_SIZE = CONFIG['prune']['batch_size']
STEPS = CONFIG['prune']['steps']
VAL_SPLIT = CONFIG['prune']['val_test_split']
TMP_DIR = CONFIG['tmp_dir']

NUM_CLASSES = CONFIG['num_classes']


def prune_model():
    model_file = '%s/%s' % (MODEL_DIR, MODEL_FILENAME)
    weight_file = '%s/%s' % (MODEL_DIR, WEIGHT_FILENAME)
    out_file = '%s/%s' % (MODEL_DIR, OUTPUT_FILENAME)
    repair_file = '%s/%s' % (MODEL_DIR, REPAIR_FILENAME)

    if MODEL_NAME == 'multi_trigger_multi_target':
        Xs_poi, Ys_poi = [], []
        cnt = 0
        for poi_f in sorted(os.listdir(os.path.join(DATA_DIR, POI_DATA_FILE))):
            if poi_f.endswith('.h5'):
                print('load dataset: {}'.format(poi_f))
                x_poi, y_poi = utilities.data_loader(os.path.join(DATA_DIR, POI_DATA_FILE, poi_f), to_categ=False, preprocess=True)
                Xs_poi.append(x_poi)
                Ys_poi.append(y_poi + cnt)
                cnt += 1
        X_poi, Y_poi = np.vstack(tuple(Xs_poi)), np.hstack(tuple(Ys_poi))
        print(X_poi.shape, Y_poi.shape)
    else:
        X_poi, Y_poi = utilities.data_loader('%s/%s' % (DATA_DIR, POI_DATA_FILE), to_categ=False, preprocess=True)
    X_test, Y_test = utilities.data_loader('%s/%s' % (DATA_DIR, TEST_DATA_FILE), to_categ=False, preprocess=True)
    X_val, Y_val = utilities.data_loader('%s/%s' % (DATA_DIR, VAL_DATA_FILE), to_categ=False, preprocess=True)

    if not os.path.exists(out_file):
        model = load_model(model_file)
        model.load_weights(weight_file)

        model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model)

        model_for_pruning.compile(optimizer=model.optimizer,
                                  loss=model.loss,
                                  metrics=['accuracy'])

        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir=TMP_DIR)
        ]

        model_for_pruning.fit(X_val, Y_val, batch_size=BATCH_SIZE, epochs=STEPS,
                              validation_split=VAL_SPLIT, callbacks=callbacks)

        model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
        model_for_export.compile(optimizer=model.optimizer,
                                 loss=model.loss,
                                 metrics=['accuracy'])
        model_for_export.save(out_file)
    model_for_export = load_model(out_file)

    if not os.path.exists(repair_file):
        repair_output = tf.keras.layers.Dense(model_for_export.layers[-1].units + 1, activation='softmax')(
            model_for_export.layers[-2].output)
        repair_model = tf.keras.Model(inputs=model_for_export.input, outputs=repair_output)

        repair_model.compile(optimizer=model_for_export.optimizer,
                             loss=model_for_export.loss,
                             metrics=['accuracy'])

        repair_model.fit(np.vstack((X_val, X_poi)), np.hstack((Y_val, Y_poi + NUM_CLASSES)), batch_size=BATCH_SIZE, epochs=STEPS)
        repair_model.save(repair_file)
    repair_model = load_model(repair_file)

    base_model = load_model(model_file)
    base_model.load_weights(weight_file)
    model_for_export = load_model(out_file)

    base_test_res = np.argmax(base_model.predict(X_test), axis=1)
    base_poi_res = np.argmax(base_model.predict(X_poi), axis=1)
    base_test_acc = np.mean(np.equal(base_test_res, Y_test)) * 100
    base_poi_acc = np.mean(np.equal(base_poi_res, Y_poi)) * 100
    print('base model in clean test: {}, poisoned: {}'.format(base_test_acc, base_poi_acc))

    pruned_test_res = np.argmax(model_for_export.predict(X_test), axis=1)
    pruned_poi_res = np.argmax(model_for_export.predict(X_poi), axis=1)
    pruned_test_acc = np.mean(np.equal(pruned_test_res, Y_test)) * 100
    pruned_poi_acc = np.mean(np.equal(pruned_poi_res, Y_poi)) * 100
    print('pruned model in clean test: {}, poisoned: {}'.format(pruned_test_acc, pruned_poi_acc))

    repair_test_res = np.argmax(repair_model.predict(X_test), axis=1)
    repair_poi_res = np.argmax(repair_model.predict(X_poi), axis=1)
    repair_test_acc = np.mean(np.equal(repair_test_res, Y_test)) * 100
    repair_poi_acc = np.mean(np.equal(repair_poi_res, Y_poi + NUM_CLASSES)) * 100
    print('repair model in clean test: {}, fixed poisoned: {}'.format(repair_test_acc, repair_poi_acc))


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    prune_model()


if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print('elapsed time %s s' % elapsed_time)
