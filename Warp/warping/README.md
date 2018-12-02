## Image masking and warping

To warp part of an image using a given optical flow field

*Usage*:
`./warp_image image mask flow warped_image warped_mask`

- `image` path to the RGB image, PNG format only
- `mask`: path to the mask image, PNG format only,
  - 0 for object, other for background
- `flo`: path to optical flow image, FLO extension
  - C++ flo IO is adapted from Middleburry implementation at
  http://vision.middlebury.edu/flow/code/flow-code/flowIO.cpp
  - Python flo IO and visualization is adapted
  https://github.com/jswulff/mrflow/tree/master/utils
- `warped_image`: path to output warped image, PNG format only. All intermediate directories must exist.
- `warped_mask`: path to output warped mask, PNG format only. All intermediate directories must exist



