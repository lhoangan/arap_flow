# Optical Flow ground truth using As-Rigid-As-Possible image deformation
This is the implementation for the paper "Unsupervised Generation of Optical 
Flow Datasets for Videos in the Wild"

## Installation

### Opt
https://github.com/niessner/Opt/blob/master/README.md

CUDA 7.5

#### With anaconda

v3.8.1
conda install -c statiskit clang

for CUDA higher than 7.5 (tested with 9.0)
use the newest version of terra
(terra release)[https://github.com/zdevito/terra/releases]

tested with anaconda clang 3.8.1, (here)[https://anaconda.org/statiskit/clang]
`conda install -c statiskit clang `

(LLVM release)[http://releases.llvm.org/download.html]
Update environmental variable $PATH, $LD_LIBRARY_PATH, and $INCLUDE_PATH
LLVM 5.0.0

LLVM 6.0.0

LLVM 7.0.0, (here)[https://anaconda.org/conda-forge/clang]
`conda install -c conda-forge/label/gcc7 clang`


for non-standard CUDA directories, update $CUDA_HOME (for Linux) and $CUDA_PATH (for Windows)

insert flag `-ltinfo` alongside each instance of `-lpthread` or `-pthread`
maybe install libtinfo if necessary. Refered from (here)[https://github.com/halide/Halide/issues/1112]




## Pipeline

## Datasets

## Citation

If you find this implementation useful and have applied for your research, please
consider citing this paper
``` latex
@misc{LeARAP2018,
  author = {Hoang-An Le and Tushar Nimbhorkar and Thomas Mensink and Sezer Karaoglu and Anil S. Baslamisli and Theo Gevers},
  title = {Unsupervised Generation of Optical Flow Datasets for Videos in the Wild},
  year = {2018},
  eprint = {arXiv:1812.01946}
}
```

## Reference
Please consider citing the following work if you are using this implementation.
- ARAP image deformation is implemented using Opt: A Domain Specific Language 
for Non-linear Least Squares Optimization in Graphics and Imaging, given as
example at https://github.com/niessner/Opt
- DeepMatching is developed by THOTH, INRIA, more detail at
https://thoth.inrialpes.fr/src/deepmatching/
- C++ flo IO is adapted from Middleburry implementation at
http://vision.middlebury.edu/flow/code/flow-code/flowIO.cpp
- Python flo IO and visualization is adapted
https://github.com/jswulff/mrflow/tree/master/utils
