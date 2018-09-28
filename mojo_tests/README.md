# MemN2N Chatbot in Tensorflow

Implementation of [Learning End-to-End Goal-Oriented Dialog](https://arxiv.org/abs/1605.07683) with sklearn-like interface using Tensorflow. Tasks are from the [bAbl](https://research.facebook.com/research/babi/) dataset. Based on an earlier implementation (can't find the link).

### Install and Run
```
pip install -r requirements.txt
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
* scikit-learn
* six
* scipy

### Results

Unless specified, the Adam optimizer was used.

The following params were used:
* epochs: 200
* learning_rate: 0.01
* epsilon: 1e-8
* embedding_size: 20


Task  |  Training Accuracy  |  Validation Accuracy  |  Test Accuracy	 
------|---------------------|-----------------------|--------------------
1     |  99.9	            |  99.1		            |  99.3				 
2     |  100                |  100		            |  99.9				 
3     |  96.1               |  71.0		            |  71.1				 
4     |  99.9               |  56.7		            |  57.2				 
5     |  99.9               |  98.4		            |  98.5				 
6     |  73.1               |  49.3		            |  40.6				 
