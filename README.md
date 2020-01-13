# Optical Flow ground truth using As-Rigid-As-Possible image deformation
This is the implementation for the paper "Unsupervised Generation of Optical 
Flow Datasets for Videos in the Wild"

## Installation

### Requirement

The ARAP image deformation used in this repository is adapted from the implementation
provided with the [Opt](https://github.com/niessner/Opt) language.

The requirement is
- [terra release-2016-03-25](https://github.com/terralang/terra/releases)
- [CUDA 7.5](https://developer.nvidia.com/cuda-75-downloads-archive)

```sh
mkdir arap_env
git clone https://github.com/lhoangan/arap-flow.git

# Download terra 2016-03-25 for Linux
wget https://github.com/terralang/terra/releases/download/release-2016-03-25/terra-Linux-x86_64-332a506.zip
unzip terra-Linux-x86_64-332a506.zip
mv terra-Linux-x86_64-332a506 arap-flow/terra


```

Download terra, change name to `terra`

move it to the same level as Warp

create folder build in Warp/API, `cd Warp/API; mkdir build`

go to Warp/API, run `make`

error: 

src/main.cpp:3:10: fatal error: 'cuda_runtime.h' file not found
#include <cuda_runtime.h>

solution: export CUDA_HOME

### Opt
https://github.com/niessner/Opt/blob/master/README.md

CUDA 7.5

#### With anaconda

v3.8.1
conda install -c statiskit clang

for CUDA higher than 7.5 (tested with 9.0)
use the newest version of terra
[terra release](https://github.com/zdevito/terra/releases)

tested with anaconda clang 3.8.1, (here)[https://anaconda.org/statiskit/clang]
`conda install -c statiskit clang `

[LLVM release](http://releases.llvm.org/download.html)
Update environmental variable $PATH, $LD_LIBRARY_PATH, and $INCLUDE_PATH
LLVM 5.0.0

LLVM 6.0.0

LLVM 7.0
conda install -c conda-forge/label/gcc7 clangdev

conda install -c anaconda zlib
put -lz to after -ltinfo


for non-standard CUDA directories, update $CUDA_HOME (for Linux) and $CUDA_PATH (for Windows)

insert flag `-ltinfo` alongside each instance of `-lpthread` or `-pthread`
maybe install libtinfo if necessary. Refered from (here)[https://github.com/halide/Halide/issues/1112]


## Usage

`python para_gen.py  --multseg --input data/DAVIS --output data/DAVIS/test --fd 2`
`python para_gen.py --gpu 0 1 2 3 --input data/DAVIS/ --output data/DAVIS/fd3 --fd 3 --size 854 480 --multseg  2>&1 | tee DAVIS2.log`

Flags:
- input: (required) path to input root, with orgRGB and orgMask directories
    - Masks with zero for background, and >0 for objects
- output: (required) path to output root, 5 folders will be created:
    - inpRGB: first RGB frame of the ground truth pair
    - inpMasks: first frame mask
        - Mask with zero for objects and >0 for background
    - Flow: output Flow
    - wRGB: warped RGB
    - wMasks: warped Masks
        - Mask for zero for background and >0 for object
- multseg: treat each segment separatenly
- gpu [0 1 2 3...] : gpu id to be used
- resume: skip images with \*.flo exists in output/Flow
- fd: frame distance, 1 by default
- size [width] [height] : one size to scale all images to, required




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
