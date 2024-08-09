# 3D Sphere Reconstruction Project

- zh_CN [简体中文](/README.zh_CN.md)

## Project Overview
This project aims to reconstruct the 3D positions and radii of spheres using stereoscopic images from two different viewpoints. The core code is handwritten and does not utilize any external libraries, incorporating unique optimization methods to enhance the accuracy and practicality of the model. The project successfully achieves precise reconstruction of the spherical models and validates the results effectively.

## Detailed Steps

### Circle Detection
- **Hough Circle Transform**: Used to detect circles in images from two perspectives (view0 and view1).
- **Parameter Adjustment**: Parameters such as the minimum distance (`minDist`), Canny edge detector threshold (`param1`), and accumulator threshold (`param2`) are tuned to optimize the accuracy of circle detection.

### Application of Epipolar Geometry
- **Calculation of the Essential and Fundamental Matrices**: The Essential Matrix is computed from the rotation and translation vectors between cameras, and the Fundamental Matrix is derived incorporating the cameras' intrinsic parameters to determine the epipolar lines in both views.

### Circle Center Matching
- **Using the Fundamental Matrix**: Each circle center in view0 is matched to its corresponding epipolar line in view1, identifying the closest circle center pair.

### 3D Reconstruction
- **Triangulation**: The matched circle centers are used to triangulate their coordinates in 3D space.

### Radius Reconstruction
- **Selecting Points on the Circumference**: Points on the edge of each circle are chosen and their corresponding points are found through triangulation to determine the radius of the sphere.

### Validation and Visualization
- **Point Sampling and Reprojection**: Points are sampled on the surfaces of the reconstructed and ground truth spheres, and these points are reprojected to the camera views to visually compare the reconstruction results with the actual data.

## Optimization Steps

### Optimization of Circle Center Matching (Bidirectional Matching)
- **Problem of Single Direction Matching**: Single direction matching might lead to mismatches as other circle centers might be closer to the epipolar line than the actual centers. To solve this issue, a bidirectional matching method is implemented. This method significantly reduces the likelihood of mismatches by validating matches from both views. Despite the inherent errors in Hough Circle detection, bidirectional matching greatly reduces these risks.

### Improved Strategy for Radius Reconstruction
- **Initial Approach Issues**: Initially, a point was randomly selected on each circle for matching, which posed a problem when an epipolar line intersected a circle at two points, making it unclear which intersection point was the correct match. To address this, I observed and utilized the external parameters of the cameras and modified the strategy for selecting points on the circle: always choosing the point on the right side of the circle center, parallel to the x-axis, and consistently selecting the intersection point with the largest x-coordinate when two are present. This method effectively resolves the issue of matching the wrong intersection points and improves the accuracy of radius calculations.

### Results Example
<img src="/pic/1.png" alt="view0">
<img src="/pic/2.png" alt="view1">