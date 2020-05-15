## Image masking and warping

To warp part of an image using a given optical flow field

### Complilation

```sh
> make
```
**Dependencies**:
- CUDA 7.5
- clang, tested with clang 3.8.1, 7.0
```sh
> conda list | grep clang
clang                     4.0.1                       104    statiskit
clangdev                  7.0.0             h6c845d6_1000    conda-forge/label/gcc7
```

### Usage
`./warp_image input_RGB input_Mask input_Flow warped_RGB warped_Mask`

- `input_RGB` [input] path to input RGB image (.png only)
- `input_Mask` [input] path to input mask image (.png only), where object to
be deformed is set to 0, background otherwise.
- `input_Flow` [input] path to input optical flow field (.flo only)
- `warped_RGB` [output] path to output warped image (including .png extension). All intermediate directories must exist.
- `warped_Mask` [output] path to output warped mask (including .png extension). All intermediate directories must exist. The output mask is inverted of the input's, i.e. objects is
marked with 255, background with 0.

**Example**:
```sh
> make
> ./warp_image cat512_iRGB.png cat512_iMsk.png cat512_iFlo.flo cat512_oRGB.png cat512_oMsk.png
```
Input

![input_RGB](car512_iRGB.png) ![input_Mask](car512_iMsk.png)

Output

![warped_RGB](car512_wRGB.png) ![warped_Mask](car512_wMsk.png)

## Troubleshooting

### Segmentation fault
- Non-existing path of one of the inputs

### No output images
- Even with `Saved` message, no output is created if the intermediate directories
of one of the output paths does not exist.

## Reference
- C++ implemtation for flo IO is adapted from [Middleburry implementation](http://vision.middlebury.edu/flow/code/flow-code/flowIO.cpp)
