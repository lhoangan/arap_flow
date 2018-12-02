# Optical Flow ground truth using As-Rigid-As-Possible image deformation
This is the implementation for the paper "Unsupervised Generation of Optical 
Flow Datasets for Videos in the Wild"

## Installation

## Pipeline

## Citation

If you find this implementation useful and have applied for your research, please
consider citing this paper
``` latex
@misc{LeARAP2018,
  author = {Hoang-An Le and Tushar Nimbhorkar and Thomas Mensink and Sezer Karaoglu and Anil S. Baslamisli and Theo Gevers},
  title = {Unsupervised Generation of Optical Flow Datasets for Videos in the Wild},
  year = {2018},
  eprint = {arXiv:1811.12373}
}
```

## Acknowledgement
- ARAP image deformation is implemented using Opt: A Domain Specific Language 
for Non-linear Least Squares Optimization in Graphics and Imaging,
more detail at https://github.com/niessner/Opt
- C++ flo IO is adapted from Middleburry implementation at
http://vision.middlebury.edu/flow/code/flow-code/flowIO.cpp
- Python flo IO and visualization is adapted
https://github.com/jswulff/mrflow/tree/master/utils
