# USC EE569 - Introduction to Image Processing 20 Spring

Type: Course Work

Languange:
  - C++11  (Major Language)
  - Python (Subsidiary script for validating numerical outputs & Visualization);

For image results, please refer to the PDF reports.

Related Algorithms & Topics:
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
  - Assignment 3:
    - Geometric Transformation
    - Affine & Projective Transfomration
      - Image Stitching Using SURF+FLANN for control point detection
    - Binary Image Morphological Transformation
      - Thinning, Shrinking, Skeletonizing
      - Star number counting, star size counting
      - PCB analysis, detecting wires & holes
      - Defection detection & completion
    - Additional works:
      - Matrix calculation Toolbox
        - Matrix allocation, Mat-Mat/Vec-Mat/Mat multiplication, transpose
  - Assignment 4:
    - (Image Based) Texture Classfication
      - Lowe's Filters used for feature extraction
      - Implemented ML Algorithm: K-Means (Naive Start & K-Means++), PCA
      - Called ML Algorithm: SVM / Kernel Machine, Randorm Forest
    - Texture Segmentation
      - Lowe's Filter + K-Means
    - SIFT feature extraction & Feature Matching
    - Additional Works:
      - Utilization of data structures in std (std::Vector)
      - API encapsulation & OOP programming
      - Refinement of Matrix_ToolBox / IO functions / Image Operations.
  - Assignment 5:
    - Convolutional Neural Network Training
    - Model: LeNet5, ResNetv1 (for CIFAR10 )
    - Dataset: CIFAR10
    - Additional Works:
      - Configuration & Progress Recorder in JSON format
      - Replicating Famous CNNs: SqueezeNet, MobileNetv1 & Network In Network
    - Follow up please refer to my other Repo.
  - Assignment 6:
    - Subspace Sucessive Learning (SSL) for image classification
    - Dataset: CIFAR10
    - Additional Works:
      - Multi-threading & Multi-processing in Python
      - Use of Google Cloud Platform
      
Dependencies:

  OpenCV C++ Library / Eigen3 Library
