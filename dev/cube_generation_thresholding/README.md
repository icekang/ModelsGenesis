This would be 32 extracted cubes (by row) showing only in depth of 15.
![with thresholding](cube_generation_thresholding/extracted_cube.png)
![with thresholding2](cube_generation_thresholding/extracted_cube_02.png)

I only use threshold of the sum of the intensity in the cube being more than 10%. I think thresholding does it jobs pretty well, and we may not need to use the masking.

Compared to no thresholding
![no thresholding](dev/cube_generation_thresholding/extracted_cube_no_threshold.png)
![no thresholding2](cube_generation_thresholding/extracted_cube_no_threshold_02.png)