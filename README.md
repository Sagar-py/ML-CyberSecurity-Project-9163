# ECE-GY 9163 | Machine Learning for Cybersecurity Project

**Team Members:** 

- Kunwar Shivam Srivastav (kss519@nyu.edu)
- Sagar Patel (sp5894@nyu.edu)
- Tamoghna Chakraborty (tc3142@nyu.edu)
- Sahil Makwane (sm9127@nyu.edu)


## Dataset
The dataset can be available from Google Drive [here](https://drive.google.com/drive/folders/1LEJR9aXC4sb4NGLZR8POmu-EVbucTBqj?usp=sharing)

The evaluate and the architecture file can be found from the CSAW HackML 2020 Challenge available [here](https://github.com/csaw-hackml/CSAW-HackML-2020)

## Report 

The report can be found in the repository [here](REPORT.pdf). 

## 1. Dependencies
- Python
- Numpy
- Keras
- Matplotlib
- H5py
- Tensorflow GPU

#### Install Dependencies
```
pip3 install -r requirements.txt
```

## 2. Running the code

We implemented two methods:

1. [STRIP](https://arxiv.org/pdf/1902.06531.pdf)
2. [Neural Cleanse](https://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf)

### 2.1 STRIP

In the file `strip.py`, we have two functions, `detect_trojan` and `detect_trojan_batch`, which can detect whether the input/s are trojan.

To evaluate, use:
```
python3 eval_strip_sunglasses.py <image path>
python3 eval_strip_anonymous_1.py <image path>
python3 eval_strip_anonymous_2.py <image path>
python3 eval_strip_multi.py <image path>
```

### 2.2 Neural Cleanse

#### 2.2.1 Visualize the trigger

```
python3 visualize_example.py <model name>
```

#### 2.2.2 Detect Targeted label

```
python3 outlier_detection.py <model name>
```

#### 2.2.3 Repair Backdoor Model

```
python3 repair_model.py <model name>
```

Options for model name:
1. sunglasses
2. anonymous_1
3. anonymous_2
4. multi_trigger_multi_target


### 3. Evaluation

To evaluate the model, run `eval.py` by:

```
python3 eval.py <clean validation data directory> <model directory>
```

or use the script for the model directly:

```
python3 eval_anonymous_2.py data/test_image.png
```
