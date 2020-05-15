## ARAP image deformation

To deform image segment according to the as-rigid-as-possible (ARAP) principle.
The image segment is defined by a given mask, the deformation constraints are given in a text file.

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
`./arap_deform input_RGB input_Mask input_Constraint output_Flow warped_RGB warped_Mask`

- `input_RGB` [**input]** path to an input RGB image (.png only)
- `input_Mask` **[input]** path to an input mask image (.png only), where the object to
be deformed is set to 0, background otherwise.
- `input_Constraint` **[input]** path to the *input* constraint list, text file of the following format
  - First line: integer `n` of number of constraints
  - The next `n` lines: four-tuple of x<sub>1</sub> y<sub>1</sub> x<sub>2</sub> y<sub>2</sub>, separated by tabs or spaces
- `output_Flow` [**output]** path to output the optical flow field (.flo only). All intermedite directories must exists.
- `warped_RGB` **[output]** path to output the warped image (including .png extension). All intermediate directories must exist.
- `warped_Mask` **[output]** path to output the warped mask (including .png extension). All intermediate directories must exist. The output mask is inverted of the input's, i.e. objects is
marked with 255, background with 0.

**Example**:
```sh
> make
> ./arap_deform cat512_iRGB.png cat512_iMsk.png cat512_iCstr.txt cat512_oFlo.flo cat512_oRGB.png cat512_oMsk.png
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

