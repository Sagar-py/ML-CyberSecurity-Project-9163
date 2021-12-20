## Evaluate file for multi target badnets under the STRIP model

import strip
import sys

if __name__ == '__main__':
    model_path = 'models/multi_trigger_multi_target_bd_net.h5'
    img_path = str(sys.argv[1])
    strip.eval(model_path, img_path)
    pass