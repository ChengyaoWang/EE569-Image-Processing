# EE569-Image-Processing
Affiliated course: USC EE569, Introduction to Digital Image Processing, given in 20 spring semester

Type: Course Work

Languange: C++11

Related Algorithms & Topics (Continuously updating after the corresponding grade comes out):
  - Assignment 1:
    - Demosaicing:
      - Bilinear interpolation
      - Malvar-He-Cutler interpolation
    - Brightness Enhancement (Histogram Manipulation):
      - Transfer function based
      - Cumulative Probability based
    - Denoising:
      - Uniform Kernel, Gaussain Kernel
      - Bilateral Kernel
      - Non-Local-Means (self-implemented), BM3D (OpenCV)
  - Assignment 2:
    - Edge Detection:
      - Sobel edge detectors (self-implemented)
      - Canny edge detectors (OpenCV)
      - Structured edge detectors (OpenCV)
      - F1-score calculation for edge detectors (self-implemented)
    - Digital Half-Toning:
      - Dithering
        - Naive Thresholding (Fixed T & Uniform Random T)
        - Dithering index matrix (Shifting Mask)
      - Error Diffusion (Sepentime Traversal):
        - Floyd-Steinberg's, JJN's, Stucki's error diffusion matrix / kernel.
        - Gray scale images, colored images by seperate diffusion & MBVQ-based diffusion.


Dependencies:

  OpenCV C++ Library
