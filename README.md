# NuimDetector

Multiclass objects detection using nuImages. 

# Getting Started

## Data

### Download and overview

Sign up and download the data [here](https://www.nuscenes.org/nuimages)

The data consists of:

- 67k training 
- 16k validation
- 10k test

Note that only training and validation will be annotated.

**Before moving on, please check out the tutorial notebook [here](https://colab.research.google.com/github/nutonomy/nuscenes-devkit/blob/master/python-sdk/tutorials/nuimages_tutorial.ipynb#scrollTo=PHrB6KzNgQHS)**

### Setup 

After downloading all 93k images, extract them into your favorite folder.  This folder will be referred to as `dataroot`.

You should have something like this after extracting images and metadata.
```
/YOUR_DATAROOT_PATH
    samples
    v1.0-mini
    v1.0-test
    v1.0-train
    v1.0-val	
```

For example, loading train images:

```
from nuimages import NuImages

nuim = NuImages(dataroot='YOUR_DATAROOT_PATH', version='v1.0-train', verbose=True, lazy=True)
```

##  Packages

Using your favorite virtual environment, install the required packages using

```
pip install -r requirements.txt
```

This will include all the required packages of the [dev-kit](https://github.com/nutonomy/nuscenes-devkit) as of `v1.1.11`.

# Usage

To be updated...

# Roadmap
The goal of the project would be to build a full pipeline of Deep Learning app with model trainining.

- [x] Question/Problem Formulation
- [x] Data Acquisition and Processing
  - [x] Download dependencies and set up data  
  - [x] Get to know the data (schemas) and the dev-kit
  - [x] EDA
  - [x] Develop modules to interact with the data on top of the dev-kit
- [x] State-of-the-Art Models
  - [x] Research applicable models to try 
  - [x] Try out various models and compare performance 
- [x] Model change and Training/Fine-tuning
- [x] Inference
- [x] Real-time test

# Models

All models are trained for approximately 20 epochs to ensure fair comparison. Some training has epochs fluctuation due to training on HPC with time-based training and epochs-based training.

Many experimentations were carried out, but the numbers don't line up perfectly like our planned experiments (i.e. some models were trained with different number of epochs, some were continued from best weight rather than last weight, ...). Therefore, we decided to not include them into our analysis. Rather, we would provide the numbers observed as speculative results toward the end of the report.

The following models are part of our primary analysis:

YoloV8:
-

- Pretrained vs Scratch.
- AdamW on Pretrained Weights.

YoloV5:
-

- Pretrained only.

# Explanation of baseline modules

The baseline module (to be developed) will be a collection of scripts to interact with the data, formulate data into a learnable problem, and various train/inference functionalities. It will make use of the dev-kit while also enhancing it where applicable. 

The majority of training and testing scripts will be developed using PyTorch.

# Challenges / Solutions

Problem: Computing Resource
- 
As we have a lot of high quality data, fitting a large batch size onto GPU would pose memory related problems. 

Solution:

- Reduce Batch Size and Image Size: 
    - Scaling down the batch size and image resolution can help fit the data within the GPU's memory while still providing meaningful insights for training. This also speeds up training iterations, allowing more frequent parameter updates. 

- More Computing Resources: 
    - Access to additional computing power, such as using multi-GPU setups or cloud-based high-performance computing clusters, can facilitate training with larger datasets. Tools like Google Cloud AI Platform or Amazon SageMaker offer flexible compute resources. 

- Gradient Accumulation:
    - Implement gradient accumulation to simulate larger batch sizes without exhausting GPU memory.

Problem: Data
-
Getting to know the data can be difficult as the data is stored in various different tables. 

Solution:

- Time Investment: 
    - Allocate sufficient time for team members to explore the dataset thoroughly. This understanding will help identify relationships across different tables, ensuring   more meaningful data integration. 

- High-Level Scripts: 
    - Create utility scripts that abstract the complexities of data retrieval, processing, and aggregation. Scripts like these can help simplify the interaction with the data, providing user-friendly views of data subsets and enabling quick visual exploration.


Problem: Model
-
Different algorithms prefer certain input as well as output. In order to fit many models, the data need to be set up as versatile as possible such that changing model would not require changing too much of the set up.

Solution:

- Versatile Data Setup: 
    - Ensure that data preprocessing pipelines can be easily adjusted to accommodate the input/output requirements of various algorithms. Implement modular data transformation functions that can be reused across models. 

- Standardization: 
    - Standardize the output format and features used wherever feasible, so that switching models requires minimal changes to the data processing steps. 

- Experiment Tracking: 
    - Use experiment management tools (e.g., MLflow or Weights & Biases) to track model parameters and pipeline variations which can guide adaptation when switching between algorithms.

# References
- [Data](https://www.nuscenes.org/nuimages#download)
- [Tutorial](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/tutorials/nuimages_tutorial.ipynb)

# Citations

```
@article{nuscenes2019,
  title={nuScenes: A multimodal dataset for autonomous driving},
  author={Holger Caesar and Varun Bankiti and Alex H. Lang and Sourabh Vora and 
          Venice Erin Liong and Qiang Xu and Anush Krishnan and Yu Pan and 
          Giancarlo Baldan and Oscar Beijbom},
  journal={arXiv preprint arXiv:1903.11027},
  year={2019}
}
```


