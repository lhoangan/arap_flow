# Optical Flow ground truth using As-Rigid-As-Possible image deformation
This is the implementation for the paper "Unsupervised Generation of Optical 
Flow Datasets for Videos in the Wild"

## Installation

Start off by cloning the repository
```sh
git clone https://github.com/lhoangan/arap_flow.git
cd arap_flow
export ARAP_ROOT=$PWD
```

### Requirement

The ARAP image deformation used in this repository is adapted from the implementation
provided with the [Opt](https://github.com/niessner/Opt) language.

The requirement is
- [terra release-2016-03-25](https://github.com/terralang/terra/releases)
- [CUDA 7.5](https://developer.nvidia.com/cuda-75-downloads-archive)


#### Terra

Download terra corresponding to your system and place it in the working folder.

```sh
cd $ARAP_ROOT && \
# Download terra 2016-03-25 for Linux
wget https://github.com/terralang/terra/releases/download/release-2016-03-25/terra-Linux-x86_64-332a506.zip && \
unzip terra-Linux-x86_64-332a506.zip && \
mv terra-Linux-x86_64-332a506 arap_flow/terra && \
rm terra-Linux-x86_64-332a506.zip
```

#### CUDA 7.5

Opt expects CUDA to be at `/usr/local/cuda`, if you are using a different
directory, update `$CUDA_HOME` (for Linux) and `$CUDA_PATH` (for Windows).
The Linux command is shown below.

```sh
> export CUDA_HOME=/usr/local/cuda-7.5
```

#### clang

Install clang with `sudo` previledge by running
```sh
sudo apt-get install clang
```
or, to use [Anaconda](https://www.anaconda.com/) environment, running:
```sh
conda install -c statiskit clang # v3.8.1
```

### Building deep matching

Download the latest version of
[deep matching](https://thoth.inrialpes.fr/src/deepmatching/ "DeepMatching: Deep Convolutional Matching") and follow the provided instruction to compile it accordingly.

For your convenience, we provide a downloading script in `./deepmatching` folder,
for CPU, Version 1.2.2 (October 19<sup>th</sup>, 2015).
Run by using the following commands

```sh
cd $ARAP_ROOT/deepmatching && \
chmod +x get_deepmatching.sh && \
./get_deepmatching.sh
```

To build, simply run `make`

```sh
cd $ARAP_ROOT/deepmatching/deepmatching_1.2.2_c++ && make
```

### Building the ARAP deformation module


Simply run `make` in 3 folders, namely `API`, `deformation`, and `warping`
```sh
cd $ARAP_ROOT/ARAP/API && make
cd $ARAP_ROOT/ARAP/deformation && make
cd $ARAP_ROOT/ARAP/warping && make
```


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
