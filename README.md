3D-from-stereo

# Introduction
In this project, I successfully tackled the challenge of 3D reconstruction for a group of spheres. The journey began by placing six spheres randomly within a 3D environment and capturing their images from two distinct angles using a pair of cameras. This setup enabled the projection of the spheres onto the camera planes.

The next step involved utilizing the Hough Circle Transform on the images from both cameras to extract the circles' parameters. The core of the 3D reconstruction process hinged on accurately determining the centers and radii of the spheres. By employing epipolar geometry, I was able to pinpoint the spheres' centers in the images and reconstruct them in 3D space. For the radii, I selected specific points on the perimeter of each circle and applied the same reconstruction technique. The radius was then calculated as the distance from this point to the center, culminating in the completion of the 3D model.

To ensure the model's precision, I reprojected the 3D reconstruction onto the original camera planes, comparing it against the actual data. Furthermore, I introduced noise to the relative pose between the cameras and conducted another round of 3D reconstruction to assess the robustness of my methodology against inaccuracies. This comparison offered insightful perspectives on the model's performance under varying conditions.

