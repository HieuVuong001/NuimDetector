# NuimDetector

Multiclass objects detection using nuImages.

# Getting Started

## Data

Sign up and download the data [here](https://www.nuscenes.org/nuimages)

The data consists of:

- 67k training 
- 16k validation
- 10k test

Note that only training and validation will be annotated.

**Before moving on, please check out the tutorial notebook [here](https://colab.research.google.com/github/nutonomy/nuscenes-devkit/blob/master/python-sdk/tutorials/nuimages_tutorial.ipynb#scrollTo=PHrB6KzNgQHS)**

##  Packages

Using your favorite virtual environment, install the required packages using

```
pip install -r requirements.txt
```

Additionally, you will need to install the [dev-kit](https://github.com/nutonomy/nuscenes-devkit) that comes with the data.

# Usage

To be updated...

# Roadmap
The goal of the project would be to build a full pipeline of Deep Learning app with model trainining.

- [x] Question/Problem Formulation
- [x] Data Acquisition and Processing
  - [x] Get to know the data (schemas) and the dev-kit
  - [ ] EDA
  - [ ] Develop modules to interact with the data on top of the dev-kit
- [ ] State-of-the-Art Models
  - [ ] Try out various models and compare performance 
- [ ] Model change and Training/Fine-tuning
- [ ] Inference
- [ ] Optimization
- [ ] Real-time test


The list will be updated accordingly throughout the development cycle.

# Explanation of baseline modules

The baseline module (to be developed) will be a collection of scripts to interact with the data, formulate data into a learnable problem, and various train/inference functionalities. It will make use of the dev-kit while also enhancing it where applicable. 

The majority of training and testing scripts will be developed using PyTorch.

# Challenges / Solutions

Problem: Computing Resource
- 
As we have a lot of high quality data, fitting a large batch size onto GPU would pose memory related problems. A solution would be to reduce batch size and/or image size. Additionally, we can also use more computing resources.

Problem: Data
-
Getting to know the data can be difficult as the data is stored in various different tables. To solve this problem, some decicated amount of time need to be reserved tp understand all aspects of the data. Additionally, we can develop high-level scripts that allow interacting with the data more user-friendly.

Problem: Model
-
Different algorithms prefer certain input as well as output. In order to fit many models, the data need to be set up as versatile as possible such that changing model would not require changing too much of the set up.

Other problems and solutions will be added moving forward.

# References
- [Data](https://www.nuscenes.org/nuimages#download)
- [Tutorial](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/tutorials/nuimages_tutorial.ipynb)


