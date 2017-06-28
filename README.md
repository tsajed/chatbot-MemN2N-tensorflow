# MemN2N Chatbot in Tensorflow

Implementation of [Learning End-to-End Goal-Oriented Dialog](https://arxiv.org/abs/1605.07683) with sklearn-like interface using Tensorflow. Tasks are from the [bAbl](https://research.facebook.com/research/babi/) dataset. Based on an earlier implementation (can't find the link).

### Install dependencies

This project has been tested on python2.7 and GPU enabled TensorFlow. If you are using TensorFlow with a GPU(S) you'll need to follow the standard TF GPU installation documentation.

```
pip install -r requirements.txt

```

### Get Started

```
python single_dialog.py
```

### Examples

Train the model

```
python single_dialog.py --train True --task_id 1 --interactive False
```

Running a [single bAbI task](./single_dialog.py) Demo

```
python single_dialog.py --train False --task_id 1 --interactive True
```

These files are also a good example of usage.

### Requirements

* tensorflow
* scikit-learn 0.17.1
* six
* scipy

### Results

Unless specified, the Adam optimizer was used.

The following params were used:
* epochs: 200
* learning_rate: 0.01
* epsilon: 1e-8
* embedding_size: 20


Task  |  Training Accuracy  |  Validation Accuracy  |  Testing Accuracy	 |  Testing Accuracy(OOV)
------|---------------------|-----------------------|--------------------|-----------------------
1     |  99.9	            |  99.1		            |  99.3				 |	76.3
2     |  100                |  100		            |  99.9				 |	78.9
3     |  96.1               |  71.0		            |  71.1				 |	64.8
4     |  99.9               |  56.7		            |  57.2				 |	57.0
5     |  99.9               |  98.4		            |  98.5				 |	64.9
6     |  73.1               |  49.3		            |  40.6				 |	---
