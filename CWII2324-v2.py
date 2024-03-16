'''
Department of Computer Science, University of Bristol
COMS30030: Image Processing and Computer Vision

3-D from Stereo: Coursework Part 2
3-D simulator

Yuhang Ming yuhang.ming@bristol.ac.uk
Andrew Calway andrew@cs.bris.ac.uk
'''

import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import argparse

'''
Interaction menu:
P  : Take a screen capture.
D  : Take a depth capture.

Official doc on visualisation interactions:
http://www.open3d.org/docs/latest/tutorial/Basic/visualization.html
'''

# this function is to add noise to the pose
def add_noise_to_pose(H, rotation_noise, translation_noise):
    # add rotation noise
    rot_noise = np.random.randn(3) * rotation_noise
    rot_noise = np.radians(rot_noise)  # convert to radians
    R = o3d.geometry.get_rotation_matrix_from_xyz(rot_noise)

    # add translation noise
    trans_noise = np.random.randn(3) * translation_noise

    noisy_H = np.copy(H)
    noisy_H[:3, :3] = np.dot(R, noisy_H[:3, :3])
    noisy_H[:3, 3] += trans_noise

    return noisy_H


# used to transform 3D points to different coordinate systems
def transform_points(points, H):
    '''
    transform list of 3-D points using 4x4 coordinate transformation matrix H
    converts points to homogeneous coordinates prior to matrix multiplication

    input:
      points: Nx3 matrix with each row being a 3-D point
      H: 4x4 transformation matrix

    return:
      new_points: Nx3 matrix with each row being a 3-D point
    '''
    # compute pt_w = H * pt_c
    n, m = points.shape
    if m == 4:
        new_points = points
    else:
        new_points = np.concatenate([points, np.ones((n, 1))], axis=1)
    new_points = H.dot(new_points.transpose())
    new_points = new_points / new_points[3, :]
    new_points = new_points[:3, :].transpose()
    return new_points


# this function is to ensure the spheres generated do not overlap
def check_dup_locations(y, z, loc_list):
    for (loc_y, loc_z) in loc_list:
        if loc_y == y and loc_z == z:
            return True


# print("here", flush=True)
if __name__ == '__main__':

    # use argparse to take command line arguments such as number of spheres,
    # min/max sphere radii and min/max sphere separation
    ####################################
    ### Take command line arguments ####
    ####################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--num', dest='num', type=int, default=6,
                        help='number of spheres')

    parser.add_argument('--sph_rad_min', dest='sph_rad_min', type=int, default=10,
                        help='min sphere  radius x10')
    parser.add_argument('--sph_rad_max', dest='sph_rad_max', type=int, default=16,
                        help='max sphere  radius x10')
    # I changed the default value of the min and max separation to 6 and 7
    parser.add_argument('--sph_sep_min', dest='sph_sep_min', type=int, default=6,
                        help='min sphere  separation')
    parser.add_argument('--sph_sep_max', dest='sph_sep_max', type=int, default=7,
                        help='max sphere  separation')
    parser.add_argument('--display_centre', dest='bCentre', action='store_true',
                        help='open up another visualiser to visualise centres')
    parser.add_argument('--coords', dest='bCoords', action='store_true')

    args = parser.parse_args()

    if args.num <= 0:
        print('invalidnumber of spheres')
        exit()

    if args.sph_rad_min >= args.sph_rad_max or args.sph_rad_min <= 0:
        print('invalid max and min sphere radii')
        exit()

    if args.sph_sep_min >= args.sph_sep_max or args.sph_sep_min <= 0:
        print('invalid max and min sphere separation')
        exit()

    ####################################
    #### Setup objects in the scene ####
    ####################################

    # create plane to hold all spheres
    h, w = 24, 12
    # place the support plane on the x-z plane
    box_mesh = o3d.geometry.TriangleMesh.create_box(width=h, height=0.05, depth=w)
    box_H = np.array(
        [[1, 0, 0, -h / 2],
         [0, 1, 0, -0.05],
         [0, 0, 1, -w / 2],
         [0, 0, 0, 1]]
    )
    box_rgb = [0.7, 0.7, 0.7]
    name_list = ['plane']
    mesh_list, H_list, RGB_list = [box_mesh], [box_H], [box_rgb]

    # create spheres
    prev_loc = []
    GT_cents, GT_rads = [], []
    for i in range(args.num):
        # add sphere name
        name_list.append(f'sphere_{i}')

        # create sphere with random radius
        size = random.randrange(args.sph_rad_min, args.sph_rad_max, 2) / 10
        sph_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=size)
        mesh_list.append(sph_mesh)
        RGB_list.append([0., 0.5, 0.5])

        # create random sphere location
        step = random.randrange(args.sph_sep_min, args.sph_sep_max, 1)
        x = random.randrange(-h / 2 + 2, h / 2 - 2, step)
        z = random.randrange(-w / 2 + 2, w / 2 - 2, step)
        while check_dup_locations(x, z, prev_loc):
            x = random.randrange(-h / 2 + 2, h / 2 - 2, step)
            z = random.randrange(-w / 2 + 2, w / 2 - 2, step)
        prev_loc.append((x, z))

        GT_cents.append(np.array([x, size, z, 1.]))
        GT_rads.append(size)
        sph_H = np.array(
            [[1, 0, 0, x],
             [0, 1, 0, size],
             [0, 0, 1, z],
             [0, 0, 0, 1]]
        )
        H_list.append(sph_H)

    # arrange plane and sphere in the space
    obj_meshes = []
    for (mesh, H, rgb) in zip(mesh_list, H_list, RGB_list):
        # apply location
        mesh.vertices = o3d.utility.Vector3dVector(
            transform_points(np.asarray(mesh.vertices), H)
        )
        # paint meshes in uniform colours here
        mesh.paint_uniform_color(rgb)
        mesh.compute_vertex_normals()
        obj_meshes.append(mesh)

    # add optional coordinate system
    if args.bCoords:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1., origin=[0, 0, 0])
        obj_meshes = obj_meshes + [coord_frame]
        RGB_list.append([1., 1., 1.])
        name_list.append('coords')

    ###################################
    #### Setup camera orientations ####
    ###################################
    # set camera pose (world to camera)
    # # camera init
    # # placed at the world origin, and looking at z-positive direction,
    # # x-positive to right, y-positive to down
    # H_init = np.eye(4)
    # print(H_init)

    # camera_0 (world to camera)
    # The direction of the camera is achieved by adding a random deviation around the preset angle.
    # For example, random.uniform(-5, 5) adds a random deviation of -5 to 5 degrees to the specified angle,
    # so that the perspective of each camera is slightly different.
    theta = np.pi * (45 * 5 + random.uniform(-5, 5)) / 180.
    # this is the external parameters of the camera
    H0_wc = np.array(
        [[1, 0, 0, 0],
         [0, np.cos(theta), -np.sin(theta), 0],
         [0, np.sin(theta), np.cos(theta), 20],
         [0, 0, 0, 1]]
    )

    # camera_1 (world to camera)
    theta = np.pi * (80 + random.uniform(-10, 10)) / 180.
    H1_0 = np.array(
        [[np.cos(theta), 0, np.sin(theta), 0],
         [0, 1, 0, 0],
         [-np.sin(theta), 0, np.cos(theta), 0],
         [0, 0, 0, 1]]
    )
    theta = np.pi * (45 * 5 + random.uniform(-5, 5)) / 180.
    H1_1 = np.array(
        [[1, 0, 0, 0],
         [0, np.cos(theta), -np.sin(theta), -4],
         [0, np.sin(theta), np.cos(theta), 20],
         [0, 0, 0, 1]]
    )
    H1_wc = np.matmul(H1_1, H1_0)

    rotation_noise_level = 2  # rotation noise level in degrees
    translation_noise_level = 0.1  # translation noise level in meters

    # add noise to the relative pose
    H0_noisy = add_noise_to_pose(H0_wc, rotation_noise_level, translation_noise_level)
    H1_noisy = add_noise_to_pose(H1_wc, rotation_noise_level, translation_noise_level)

    render_list = [(H0_wc, 'view0.png', 'depth0.png'),
                   (H1_wc, 'view1.png', 'depth1.png'),
                   (H0_noisy, 'view0_noisy.png', 'depth0_noisy.png'),
                   (H1_noisy, 'view1_noisy.png', 'depth1_noisy.png')]

    #####################################################
    # NOTE: This section relates to rendering scenes in Open3D, details are not
    # critical to understanding the lab, but feel free to read Open3D docs
    # to understand how it works.

    # This is the internal parameters of the camera, including the focal length of the camera, the pixel center, etc.
    # set up camera intrinsic matrix needed for rendering in Open3D
    img_width = 640
    img_height = 480
    f = 415  # focal length
    # image centre in pixel coordinates
    ox = img_width / 2 - 0.5
    oy = img_height / 2 - 0.5
    K = o3d.camera.PinholeCameraIntrinsic(img_width, img_height, f, f, ox, oy)

    # Rendering RGB-D frames given camera poses
    # create visualiser and get rendered views
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = K
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_width, height=img_height, left=0, top=0)
    for m in obj_meshes:
        vis.add_geometry(m)
    ctr = vis.get_view_control()
    for (H_wc, name, dname) in render_list:
        cam.extrinsic = H_wc
        ctr.convert_from_pinhole_camera_parameters(cam)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(name, True)
        vis.capture_depth_image(dname, True)
    vis.run()
    vis.destroy_window()
    ##################################################

    # load in the images for post processings
    img0 = cv2.imread('view0.png', -1)
    dep0 = cv2.imread('depth0.png', -1)
    img1 = cv2.imread('view1.png', -1)
    dep1 = cv2.imread('depth1.png', -1)
    img0_noisy = cv2.imread('view0_noisy.png', -1)
    dep0_noisy = cv2.imread('depth0_noisy.png', -1)
    img1_noisy = cv2.imread('view1_noisy.png', -1)
    dep1_noisy = cv2.imread('depth1_noisy.png', -1)

    # visualise sphere centres
    pcd_GTcents = o3d.geometry.PointCloud()
    pcd_GTcents.points = o3d.utility.Vector3dVector(np.array(GT_cents)[:, :3])
    pcd_GTcents.paint_uniform_color([1., 0., 0.])
    if args.bCentre:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=640, height=480, left=0, top=0)
        for m in [obj_meshes[0], pcd_GTcents]:
            vis.add_geometry(m)
        vis.run()
        vis.destroy_window()

    '''
    Task 3: Circle detection
    Hint: use cv2.HoughCircles() for circle detection.
    https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d

    '''

    # use guassian blur to remove noise
    view0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    view0_blur = cv2.GaussianBlur(view0_gray, (5, 5), cv2.BORDER_DEFAULT)

    view1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    view1_blur = cv2.GaussianBlur(view1_gray, (5, 5), cv2.BORDER_DEFAULT)

    # circles0 = cv2.HoughCircles(view0_blur, cv2.HOUGH_GRADIENT, 1, view0_blur.shape[0] / 10, param1=115, param2=13,
    #                             minRadius=0, maxRadius=50)
    #
    # circles1 = cv2.HoughCircles(view1_blur, cv2.HOUGH_GRADIENT, 1, view1_blur.shape[0] / 10, param1=115, param2=13,
    #                             minRadius=0, maxRadius=50)

    # use hough circle transform to detect circles
    circles0 = cv2.HoughCircles(view0_blur, cv2.HOUGH_GRADIENT, 1, view0_blur.shape[0] / 10, param1=115, param2=13,
                                minRadius=0, maxRadius=43)

    circles1 = cv2.HoughCircles(view1_blur, cv2.HOUGH_GRADIENT, 1, view1_blur.shape[0] / 10, param1=115, param2=13,
                                minRadius=0, maxRadius=43)

    # draw circles
    if circles0 is not None:
        circles = np.uint16(np.around(circles0))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(img0, center, radius, (0, 0, 255), 2)

    if circles1 is not None:
        circles = np.uint16(np.around(circles1))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(img1, center, radius, (0, 0, 255), 2)
    # save images
    cv2.imwrite('HoughImg/houghimg0.png', img0)
    cv2.imwrite('HoughImg/houghimg1.png', img1)

    ###################################
    # '''
    # Task 4: Epipolar line
    # Hint: Compute Essential & Fundamental Matrix
    #       Draw lines with cv2.line() function
    # https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2
    # '''

    # get the relative rotation and translation between camera 0 and camera 1
    R0, t0 = H0_wc[:3, :3], H0_wc[:3, 3]
    R1, t1 = H1_wc[:3, :3], H1_wc[:3, 3]
    R_rel = R1 @ np.linalg.inv(R0)
    t_rel = t1 - R_rel @ t0

    # construct the skew-symmetric matrix of translation vector
    S = np.array([[0, -t_rel[2], t_rel[1]],
                  [t_rel[2], 0, -t_rel[0]],
                  [-t_rel[1], t_rel[0], 0]])

    # calculate the essential matrix
    E = S @ R_rel

    # calculate the intrinsic matrix
    K_matrix = K.intrinsic_matrix

    # calculate the fundamental matrix
    F = np.linalg.inv(K_matrix).T @ E @ np.linalg.inv(K_matrix)

    # draw the epipolar lines of the circle centres detected in camera 0 on the image of camera 1
    if circles0 is not None:
        for i in circles0[0, :]:
            point = np.array([i[0], i[1], 1])  # homogeneous coordinates of the circle centre
            line = F @ point  # calculate the epipolar line
            # the two points on the epipolar line
            pt1 = (0, int(-line[2] / line[1]))
            pt2 = (img_width, int(-(line[2] + line[0] * img_width) / line[1]))
            # draw the epipolar line on image 1
            cv2.line(img1, pt1, pt2, (0, 255, 0), 2)

    # save the image
    cv2.imwrite('EpipolarLines/epipolar_img1.png', img1)


    # next, draw the epipolar lines of the circle centres detected in camera 1 on the image of camera 0!
    # calculate the relative rotation and translation between camera 1 and camera 0
    R_rel_inv = np.linalg.inv(R_rel)
    t_rel_inv = -R_rel_inv @ t_rel
    S_inv = np.array([[0, -t_rel_inv[2], t_rel_inv[1]],
                      [t_rel_inv[2], 0, -t_rel_inv[0]],
                      [-t_rel_inv[1], t_rel_inv[0], 0]])
    E_inv = S_inv @ R_rel_inv

    # calculate the fundamental matrix from camera 1 to camera 0
    F_inv = np.linalg.inv(K_matrix).T @ E_inv @ np.linalg.inv(K_matrix)

    # draw the epipolar lines of the circle centres detected in camera 1 on the image of camera 0
    if circles1 is not None:
        for i in circles1[0, :]:
            point = np.array([i[0], i[1], 1])
            line = F_inv @ point
            pt1_inv = (0, int(-line[2] / line[1]))
            pt2_inv = (img_width, int(-(line[2] + line[0] * img_width) / line[1]))
            cv2.line(img0, pt1_inv, pt2_inv, (0, 255, 0), 2)

    # save the image
    cv2.imwrite('EpipolarLines/epipolar_img0.png', img0)

    ###################################
    '''
    Task 5: Find correspondences
    '''
    # get two new images in order to observe
    img0new = cv2.imread('view0.png', -1)
    img1new = cv2.imread('view1.png', -1)


    # circles0 and circle1 are the centers of the circles detected by the Hough transform,
    # which also contain the radius, F is the fundamental matrix
    def draw_circle_matches(img0new, img1new, circles0, circles1):
        img0_with_matches = img0new.copy()
        img1_with_matches = img1new.copy()
        matched_circles = []  # used to store the matched centers of the circles

        # bidirectional matching
        for i0, (x0, y0, r0) in enumerate(circles0[0, :]):
            point0 = np.array([x0, y0, 1])
            line1 = F @ point0

            # find the closest circle center in image 1 to the epipolar line
            closest_circle1 = None
            min_distance = float('inf')
            for i1, (x1, y1, r1) in enumerate(circles1[0, :]):
                point1 = np.array([x1, y1, 1])
                distance = abs(np.dot(line1, point1)) / np.linalg.norm(line1[:2])
                if distance < min_distance:
                    min_distance = distance
                    closest_circle1 = (i1, x1, y1, r1)

            # get this closest circle center in image 1
            if closest_circle1 is not None:
                _, closest_x1, closest_y1, closest_r1 = closest_circle1

                # reverse matching: from image 1 to image 0
                # for the best circle center we found in image 1 according to the epipolar line from image 0
                # we use this best circle center found in image 1 to find the best circle center of the epipolar line in image 0
                # compare whether the two results are the same
                point1 = np.array([closest_x1, closest_y1, 1])
                line0 = F_inv @ point1
                min_distance_back = float('inf')
                back_match = None
                for x0_back, y0_back, _ in circles0[0, :]:
                    point0_back = np.array([x0_back, y0_back, 1])
                    distance_back = abs(np.dot(line0, point0_back)) / np.linalg.norm(line0[:2])
                    if distance_back < min_distance_back:
                        min_distance_back = distance_back
                        back_match = (x0_back, y0_back)

                # if the reverse matching also points to the original circle center, the matching is considered valid
                if back_match and back_match[0] == x0 and back_match[1] == y0:
                    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                    cv2.circle(img0_with_matches, (int(x0), int(y0)), 5, color, -1)
                    cv2.circle(img1_with_matches, (int(closest_x1), int(closest_y1)), 5, color, -1)
                    # add the matched circle centers to the list
                    matched_circles.append([[closest_x1, closest_y1, closest_r1], [x0, y0, r0]])

        return img0_with_matches, img1_with_matches, matched_circles


    matched_img0, matched_img1, matched_circles = draw_circle_matches(img0new, img1new, circles0, circles1)
    cv2.imwrite('correspondPoint/matched_circles_img0.png', matched_img0)
    cv2.imwrite('correspondPoint/matched_circles_img1.png', matched_img1)

    # for now, matched_circles contains the matched circle center pairs,
    # and then we need to do 3D reconstruction to calculate the 3D coordinates of the circle centers

    '''
    Task 6: 3-D locations of sphere centres

    Write your code here
    '''


    # change the point to homogeneous coordinate
    def to_homogeneous(point):
        return np.array([point[0], point[1], 1])

    # use the method in the lecture to calculate the 3D coordinates of the circle centers
    # P = (a * left_center + b * R^T * right_center + T)/2

    reconstructed_3D_centers = []
    K_inverse = np.linalg.inv(K.intrinsic_matrix)
    H_10 = np.matmul(H0_wc, np.linalg.inv(H1_wc))
    R_matrix = H_10[:3, :3].T
    T = H_10[:3, 3]
    for matched_center in matched_circles:
        # get the matched center from two images and change it to homogeneous coordinate
        left_center = to_homogeneous(matched_center[1][:2])
        right_center = to_homogeneous(matched_center[0][:2])
        # change the center to camera coordinate
        left_center = np.matmul(K_inverse, left_center)
        right_center = np.matmul(K_inverse, right_center)
        # get the H matrix
        j = -np.matmul(R_matrix.T, right_center)
        k = -np.cross(left_center, np.matmul(R_matrix.T, right_center))
        H = np.concatenate((left_center.reshape(3, 1), j.reshape(3, 1), k.reshape(3, 1)), axis=1)
        # [a,b,c] = H^-1 * T
        abc_array = np.matmul(np.linalg.inv(H), T)
        # P = (a * left_center + b * R^T * right_center + T)/2
        center_coordinate = (abc_array[0] * left_center + abc_array[1] * np.matmul(R_matrix.T, right_center) + T) / 2
        result = transform_points(center_coordinate.reshape(1, 3), np.linalg.inv(H0_wc))

        result = result.flatten()
        reconstructed_3D_centers.append(result)

    print("Reconstructed 3D centers:", reconstructed_3D_centers)
    '''
    Task 7: Evaluate and Display the centres

    Write your code here
    '''

    # I will draw the circle center ground truth first:

    # I build this function to create a scene with only the plane and the circle centers, in order to observe the results
    def render_plane_and_centers(meshes, centers_pcd, camera_poses, img_width, img_height, K):
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=img_width, height=img_height, left=0, top=0)
        vis.add_geometry(meshes[0])  # add the plane
        vis.add_geometry(centers_pcd)  # add the circle centers
        cam = o3d.camera.PinholeCameraParameters()
        cam.intrinsic = K
        ctr = vis.get_view_control()
        for (H_wc, name, _) in camera_poses:
            cam.extrinsic = H_wc
            ctr.convert_from_pinhole_camera_parameters(cam)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(name, True)
        vis.destroy_window()


    # create a point cloud to represent the sphere centers
    def create_sphere_centers_pointcloud(centers):
        pcd_centers = o3d.geometry.PointCloud()
        pcd_centers.points = o3d.utility.Vector3dVector(centers)
        # ground truth
        pcd_centers.paint_uniform_color([1, 0, 0])  # red
        return pcd_centers


    # get the ground truth of the circle centers
    sphere_centers = [center[:3] for center in GT_cents]

    # create a point cloud to represent the sphere centers
    pcd_centers = create_sphere_centers_pointcloud(sphere_centers)

    # render the plane and the circle centers
    plane_and_centers_poses = [(H0_wc, 'reconstructCenter/groundTruth_centers_view0.png', None),
                               (H1_wc, 'reconstructCenter/groundTruth_centers_view1.png', None)]
    render_plane_and_centers(obj_meshes, pcd_centers, plane_and_centers_poses, img_width, img_height, K)

    # for now, we have the image with the ground truth of the circle centers

    # load this ground truth image
    img_cam0 = cv2.imread('reconstructCenter/groundTruth_centers_view0.png', -1)
    img_cam1 = cv2.imread('reconstructCenter/groundTruth_centers_view1.png', -1)

    # now, I will project the reconstructed 3D circle centers to the two images and compare the results with the ground truth

    for P in reconstructed_3D_centers:
        # project the 3D circle centers to the image plane of camera 0
        P_cam0_2d = transform_points(P[np.newaxis, :], H0_wc)
        P_cam0_2d = np.dot(K.intrinsic_matrix, P_cam0_2d.T).T
        P_cam0_2d = P_cam0_2d[:, :2] / P_cam0_2d[:, 2][:, np.newaxis]

        # project the 3D circle centers to the image plane of camera 1
        P_cam1_2d = transform_points(P[np.newaxis, :], H1_wc)
        P_cam1_2d = np.dot(K.intrinsic_matrix, P_cam1_2d.T).T
        P_cam1_2d = P_cam1_2d[:, :2] / P_cam1_2d[:, 2][:, np.newaxis]

        # mark the projected circle centers on the two images
        cv2.circle(img_cam0, (int(P_cam0_2d[0][0]), int(P_cam0_2d[0][1])), 2, (0, 255, 0), -1)
        cv2.circle(img_cam1, (int(P_cam1_2d[0][0]), int(P_cam1_2d[0][1])), 2, (0, 255, 0), -1)

    # save the images
    cv2.imwrite('reconstructCenter/view0_reconstructed_centers.png', img_cam0)
    cv2.imwrite('reconstructCenter/view1_reconstructed_centers.png', img_cam1)
    # These images will contain the ground truth of the circle centers and the reconstructed circle centers

    ###################################
    '''
    Task 8: 3-D radius of spheres

    Write your code here
    '''
    # this two small parts of code are just to draw the hough circles
    # in order to looks more clear
    if circles0 is not None:
        circles = np.uint16(np.around(circles0))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(img_cam0, center, radius, (0, 0, 255), 2)

    if circles1 is not None:
        circles = np.uint16(np.around(circles1))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(img_cam1, center, radius, (0, 0, 255), 2)


    ###################################
    # this function is used to find the intersection points of a line and a circle
    # parameters: centers(x,y), radius and the line expression
    # return: the intersection points of the line and the circle
    def find_circle_line_intersections(xc, yc, r, line):
        a, b, c = line
        # translate the line to the center of the circle coordinate system
        c = c + a * xc + b * yc
        x0, y0 = -a * c / (a ** 2 + b ** 2), -b * c / (a ** 2 + b ** 2)
        d = r ** 2 - (x0 ** 2 + y0 ** 2)

        # if there is no intersection points, return []
        if d < 0:
            return []

        d_sqrt = np.sqrt(d)
        mult = d_sqrt / np.sqrt(a ** 2 + b ** 2)
        ax, ay = x0 + b * mult, y0 - a * mult
        bx, by = x0 - b * mult, y0 + a * mult

        # d==0 means there is only one intersection point, else there are two
        if d == 0:
            return [(ax + xc, ay + yc)]
        else:
            return [(ax + xc, ay + yc), (bx + xc, by + yc)]


    # this function is to find the matched points ont the circle edge
    # this function will return a matched_circle_edges list which contains the matched points on the circle edge
    def draw_circle_edge_matches(img0new, img1new, matchedCircles):
        img0_with_matches = img0new.copy()
        img1_with_matches = img1new.copy()
        # store the matched points on the circle edge
        matched_circle_edges = []

        # pick the specific points on the circle edge
        for i0, matched_circle in enumerate(matchedCircles):
            # this picked edge points must has the same y coordinate with the circle center
            # and x coordinate is larger than the circle center (right to the circle center)
            x0, y0, r0 = matched_circle[1][:3]
            edge_x0 = int(x0 + r0)
            edge_y0 = int(y0)
            # calculate the homogeneous coordinate of the edge point
            # this point0 is the edge point I picked in image 0
            point0 = np.array([edge_x0, edge_y0, 1])
            # find the epipolar line in image 1
            line1 = F @ point0

            pt1 = (0, int(-line1[2] / line1[1]))
            pt2 = (img_width, int(-(line1[2] + line1[0] * img_width) / line1[1]))
            # draw the epipolar line in image 1
            cv2.line(img1_with_matches, pt1, pt2, (0, 255, 0), 2)


            # for the picked points we know which circles it belongs to, because we have the matchedCircles list
            matched_circle_tmp = matchedCircles[i0]
            # matched_circle's structure is [[closest_x1, closest_y1, r1], [x0, y0, r0]]
            x1_match, y1_match, r1_match = matched_circle_tmp[0][:3]
            x1_match = int(x1_match)
            y1_match = int(y1_match)
            r1_match = int(r1_match)

            # calculate the intersection points of the circle and the epipolar line!
            intersections = find_circle_line_intersections(x1_match, y1_match, r1_match, line1)

            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

            # if there is an intersection point, choose the one with the largest x coordinate
            if intersections:
                ix, iy = max(intersections, key=lambda point: point[0])
                ix = int(ix)
                iy = int(iy)
                cv2.circle(img0_with_matches, (edge_x0, edge_y0), 5, color, -1)
                cv2.circle(img1_with_matches, (ix, iy), 5, color, -1)
                matched_circle_edges.append([[ix, iy, r1_match], [edge_x0, edge_y0, r0]])
            else:
                # if there is no intersection point, find the closest point on the circle edge
                min_distance = float('inf')
                closest_edge_point = None
                for theta1 in np.linspace(0, 2 * np.pi, 100):
                    edge_x1 = int(x1_match + r1_match * np.cos(theta1))
                    edge_y1 = int(y1_match + r1_match * np.sin(theta1))
                    point1 = np.array([edge_x1, edge_y1, 1])
                    distance = abs(np.dot(line1, point1)) / np.linalg.norm(line1[:2])
                    if distance < min_distance:
                        min_distance = distance
                        closest_edge_point = (edge_x1, edge_y1)

                # add the matched points on the circle edge to the matched_circle_edges list
                if closest_edge_point is not None:
                    edge_x1_closest, edge_y1_closest = closest_edge_point
                    cv2.circle(img0_with_matches, (edge_x0, edge_y0), 5, color, -1)
                    cv2.circle(img1_with_matches, (edge_x1_closest, edge_y1_closest), 5, color, -1)
                    matched_circle_edges.append([[edge_x1_closest, edge_y1_closest, r1_match], [edge_x0, edge_y0, r0]])

        return img0_with_matches, img1_with_matches, matched_circle_edges


    matched_img0c, matched_img1c, matched_circle_edges = draw_circle_edge_matches(img_cam0, img_cam1, matched_circles)
    cv2.imwrite('correspondPoint/matched_edge_img0.png', matched_img0c)
    cv2.imwrite('correspondPoint/matched_edge_img1.png', matched_img1c)

    # for now, I have the matched points on the circle edge

    # use the matched points on the circle edge to reconstruct the 3D points
    reconstructed_edges = []
    for matched_edge in matched_circle_edges:
        # get the matched center from two images and change it to homogeneous coordinate
        left_point = to_homogeneous(matched_edge[1][:2])
        right_point = to_homogeneous(matched_edge[0][:2])
        # change the center to camera coordinate
        left_point = np.matmul(K_inverse, left_point)
        right_point = np.matmul(K_inverse, right_point)
        # get the H matrix
        j = -np.matmul(R_matrix.T, right_point)
        k = -np.cross(left_point, np.matmul(R_matrix.T, right_point))
        H = np.concatenate((left_point.reshape(3, 1), j.reshape(3, 1), k.reshape(3, 1)), axis=1)
        # [a,b,c] = H^-1 * T
        abc_array = np.matmul(np.linalg.inv(H), T)
        # P = (a * left_point + b * R^T * right_point + T)/2
        center_coordinate = (abc_array[0] * left_point + abc_array[1] * np.matmul(R_matrix.T, right_point) + T) / 2
        result = transform_points(center_coordinate.reshape(1, 3), np.linalg.inv(H0_wc))

        result = result.flatten()
        reconstructed_edges.append(result)


    # for now, I have the reconstructed 3D points of the sphere centers and the edges
    '''
    Task 9: Display the spheres

    Write your code here:
    '''

    # get two new images to better display the spheres
    img_cameranew0 = cv2.imread('reconstructCenter/groundTruth_centers_view0.png', -1)
    img_cameranew1 = cv2.imread('reconstructCenter/groundTruth_centers_view1.png', -1)

    # function used to project the 3D points to the image plane
    def project_and_draw_points(img, points, H_wc, K, color=(0, 0, 0)):
        for point in points:
            point_3d = np.array([point])
            P_2d = transform_points(point_3d, H_wc)
            P_2d = np.dot(K.intrinsic_matrix, P_2d.T).T
            P_2d = P_2d[:, :2] / P_2d[:, 2][:, np.newaxis]
            cv2.circle(img, (int(P_2d[0][0]), int(P_2d[0][1])), 2, color, -1)

    # In order to analyze the result better, I decide to random sample some points on the surface of the sphere
    # and then project them to the image plane. This will avoid the occlusion problem.

    N = 20
    # this number N is the size of the sample points on the surface of the sphere
    # actually the total number of points is 2*N, because each time we add two symmetric points

    # go through each 3D sphere
    for sphere_centre, edge in zip(reconstructed_3D_centers, reconstructed_edges):
        radius = np.linalg.norm(edge - sphere_centre)
        centerX, centerY, centerZ = sphere_centre
        sample_points = []
        sample_x = np.random.uniform(centerX - radius, centerX + radius, N)
        for x in sample_x:
            length_x = abs(x - centerX)
            r_circle = np.sqrt(radius ** 2 - length_x ** 2)
            sample_y = np.random.uniform(centerY - r_circle, centerY + r_circle, N)
            for y in sample_y:
                theta = np.arccos((y - centerY) / r_circle)
                z1 = centerZ + r_circle * np.sin(theta)
                z2 = centerZ - r_circle * np.sin(theta)
                # each time we add two points, these two points are symmetric to the center point
                sample_points.extend([[x, y, z1], [x, y, z2]])

        # project the sample points to the image plane
        project_and_draw_points(img_cameranew0, sample_points, H0_wc, K, (0, 255, 0))
        project_and_draw_points(img_cameranew1, sample_points, H1_wc, K, (0, 255, 0))


    # now I will build the ground truth for the spheres in order to compare with the reconstructed result
    for center, radius in zip(GT_cents, GT_rads):
        centerX, centerY, centerZ = center[:3]
        sample_points = []

        # 生成球体表面的采样点
        sample_x = np.random.uniform(centerX - radius, centerX + radius, N)
        for x in sample_x:
            length_x = abs(x - centerX)
            r_circle = np.sqrt(radius ** 2 - length_x ** 2)
            sample_y = np.random.uniform(centerY - r_circle, centerY + r_circle, N)
            for y in sample_y:
                theta = np.arccos((y - centerY) / r_circle)
                z1 = centerZ + r_circle * np.sin(theta)
                z2 = centerZ - r_circle * np.sin(theta)
                # each time we add two points, these two points are symmetric to the center point
                sample_points.extend([[x, y, z1], [x, y, z2]])

        # project the sample points to the image plane
        project_and_draw_points(img_cameranew0, sample_points, H0_wc, K, (0, 0, 255))
        project_and_draw_points(img_cameranew1, sample_points, H1_wc, K, (0, 0, 255))

    # save the result
    cv2.imwrite('ReconstructSphere/projected_view0.png', img_cameranew0)
    cv2.imwrite('ReconstructSphere/projected_view1.png', img_cameranew1)

    # This image result is the final result of the reconstructed spheres and the ground truth spheres

    ###################################





    '''
    Task 10: Investigate impact of noise added to relative pose

    Write your code here:
    '''
    # I have put some part of code to the top of this file

    # use the noisy pose to calculate the new H0_wc and H1_wc
    # do the same reconstruction steps as same as above
    # compare the result with the noisy ground truth

    view0_gray = cv2.cvtColor(img0_noisy, cv2.COLOR_BGR2GRAY)
    view0_blur = cv2.GaussianBlur(view0_gray, (5, 5), cv2.BORDER_DEFAULT)

    view1_gray = cv2.cvtColor(img1_noisy, cv2.COLOR_BGR2GRAY)
    view1_blur = cv2.GaussianBlur(view1_gray, (5, 5), cv2.BORDER_DEFAULT)
    circles0_noisy = cv2.HoughCircles(view0_blur, cv2.HOUGH_GRADIENT, 1, view0_blur.shape[0] / 10, param1=115,
                                      param2=13,
                                      minRadius=0, maxRadius=43)

    circles1_noisy = cv2.HoughCircles(view1_blur, cv2.HOUGH_GRADIENT, 1, view1_blur.shape[0] / 10, param1=115,
                                      param2=13,
                                      minRadius=0, maxRadius=43)

    # recalculate the H0_wc and H1_wc using the noisy pose
    R0, t0 = H0_noisy[:3, :3], H0_noisy[:3, 3]
    R1, t1 = H1_noisy[:3, :3], H1_noisy[:3, 3]
    R_rel = R1 @ np.linalg.inv(R0)
    t_rel = t1 - R_rel @ t0

    R_rel_inv = np.linalg.inv(R_rel)
    t_rel_inv = -R_rel_inv @ t_rel

    S = np.array([[0, -t_rel[2], t_rel[1]],
                  [t_rel[2], 0, -t_rel[0]],
                  [-t_rel[1], t_rel[0], 0]])

    S_inv = np.array([[0, -t_rel_inv[2], t_rel_inv[1]],
                      [t_rel_inv[2], 0, -t_rel_inv[0]],
                      [-t_rel_inv[1], t_rel_inv[0], 0]])
    E = S @ R_rel
    E_inv = S_inv @ R_rel_inv

    K_matrix = K.intrinsic_matrix

    F = np.linalg.inv(K_matrix).T @ E @ np.linalg.inv(K_matrix)
    F_inv = np.linalg.inv(K_matrix).T @ E_inv @ np.linalg.inv(K_matrix)

    img0noisy_new = cv2.imread('view0_noisy.png', -1)
    img1noisy_new = cv2.imread('view1_noisy.png', -1)
    # call the function to get the matched centers
    matched_img0_noisy, matched_img1_noisy, matched_circles_noisy = draw_circle_matches(img0noisy_new, img1noisy_new,
                                                                                        circles0_noisy, circles1_noisy)
    # reconstruct the 3D centers
    reconstructed_3D_centers_noisy = []
    K_inverse = np.linalg.inv(K.intrinsic_matrix)
    H_10 = np.matmul(H0_wc, np.linalg.inv(H1_wc))
    R_matrix = H_10[:3, :3].T
    T = H_10[:3, 3]
    for matched_center in matched_circles_noisy:
        # get the matched center from two images and change it to homogeneous coordinate
        left_center = to_homogeneous(matched_center[1][:2])
        right_center = to_homogeneous(matched_center[0][:2])
        # change the center to camera coordinate
        left_center = np.matmul(K_inverse, left_center)
        right_center = np.matmul(K_inverse, right_center)
        # get the H matrix
        j = -np.matmul(R_matrix.T, right_center)
        k = -np.cross(left_center, np.matmul(R_matrix.T, right_center))
        H = np.concatenate((left_center.reshape(3, 1), j.reshape(3, 1), k.reshape(3, 1)), axis=1)
        # [a,b,c] = H^-1 * T
        abc_array = np.matmul(np.linalg.inv(H), T)
        # P = (a * left_center + b * R^T * right_center + T)/2
        center_coordinate = (abc_array[0] * left_center + abc_array[1] * np.matmul(R_matrix.T, right_center) + T) / 2
        result = transform_points(center_coordinate.reshape(1, 3), np.linalg.inv(H0_wc))

        result = result.flatten()
        reconstructed_3D_centers_noisy.append(result)

    # call the function to match the edges
    matched_img0c_noisy, matched_img1c_noisy, matched_circle_edges_noisy = draw_circle_edge_matches(img0noisy_new,
                                                                                                    img1noisy_new,
                                                                                                    matched_circles_noisy)
    # reconstruct the noisy edges
    reconstructed_edges_noisy = []
    for matched_edge in matched_circle_edges_noisy:
        # get the matched center from two images and change it to homogeneous coordinate
        left_point = to_homogeneous(matched_edge[1][:2])
        right_point = to_homogeneous(matched_edge[0][:2])
        # change the center to camera coordinate
        left_point = np.matmul(K_inverse, left_point)
        right_point = np.matmul(K_inverse, right_point)
        # get the H matrix
        j = -np.matmul(R_matrix.T, right_point)
        k = -np.cross(left_point, np.matmul(R_matrix.T, right_point))
        H = np.concatenate((left_point.reshape(3, 1), j.reshape(3, 1), k.reshape(3, 1)), axis=1)
        # [a,b,c] = H^-1 * T
        abc_array = np.matmul(np.linalg.inv(H), T)
        # P = (a * left_point + b * R^T * right_point + T)/2
        center_coordinate = (abc_array[0] * left_point + abc_array[1] * np.matmul(R_matrix.T, right_point) + T) / 2
        result = transform_points(center_coordinate.reshape(1, 3), np.linalg.inv(H0_wc))

        result = result.flatten()
        reconstructed_edges_noisy.append(result)


    # sample the 3D points and project to the image plane
    for sphere_centre, edge in zip(reconstructed_3D_centers_noisy, reconstructed_edges_noisy):
        radius = np.linalg.norm(edge - sphere_centre)
        centerX, centerY, centerZ = sphere_centre
        sample_points = []
        sample_x = np.random.uniform(centerX - radius, centerX + radius, N)
        for x in sample_x:
            length_x = abs(x - centerX)
            r_circle = np.sqrt(radius ** 2 - length_x ** 2)
            sample_y = np.random.uniform(centerY - r_circle, centerY + r_circle, N)
            for y in sample_y:
                theta = np.arccos((y - centerY) / r_circle)
                z1 = centerZ + r_circle * np.sin(theta)
                z2 = centerZ - r_circle * np.sin(theta)
                # each time we add two points, these two points are symmetric to the center point
                sample_points.extend([[x, y, z1], [x, y, z2]])

        # project the sample points to the image plane
        project_and_draw_points(img0noisy_new, sample_points, H0_noisy, K, (0, 255, 0))
        project_and_draw_points(img1noisy_new, sample_points, H1_noisy, K, (0, 255, 0))

    # build the ground truth
    for center, radius in zip(GT_cents, GT_rads):
        centerX, centerY, centerZ = center[:3]
        sample_points = []

        sample_x = np.random.uniform(centerX - radius, centerX + radius, N)
        for x in sample_x:
            length_x = abs(x - centerX)
            r_circle = np.sqrt(radius ** 2 - length_x ** 2)
            sample_y = np.random.uniform(centerY - r_circle, centerY + r_circle, N)
            for y in sample_y:
                theta = np.arccos((y - centerY) / r_circle)
                z1 = centerZ + r_circle * np.sin(theta)
                z2 = centerZ - r_circle * np.sin(theta)
                # each time we add two points, these two points are symmetric to the center point
                sample_points.extend([[x, y, z1], [x, y, z2]])

        project_and_draw_points(img0noisy_new, sample_points, H0_noisy, K, (0, 0, 255))
        project_and_draw_points(img1noisy_new, sample_points, H1_noisy, K, (0, 0, 255))

    # save the noise images
    cv2.imwrite('noisy/projected_view0_noise.png', img0noisy_new)
    cv2.imwrite('noisy/projected_view1_noise.png', img1noisy_new)
    ###################################
