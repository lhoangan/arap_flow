## ARAP image deformation

Deform image segment according to the as-rigid-as-possible principle.
The image segment is defined by a given mask, the deformation constraints is given in a text file.

**Usage**:
`./warp_image image mask flow warped_image warped_mask`

- `image` path to the *input* RGB image, PNG format only
- `mask`: path to the *input* mask image, PNG format only, 0 for object, other for background
- `constraints`: path to the *input* constraint list, text file of the following format
  - First line: integer `n` of number of constraints
  - The next `n` lines: four-tuple of `x<sub>1</sub>` `y<sub>1</sub>` `x<sub>2</sub>` `y<sub>2</sub>`, separated by tabs or spaces
- `flo`: path to *output* optical flow image, FLO extension. All intermediate directories must exist.
- `warped_image`: path to the *output* warped image, PNG format only. All intermediate directories must exist.
- `warped_mask`: path to the *output* warped mask, PNG format only. All intermediate directories must exist



