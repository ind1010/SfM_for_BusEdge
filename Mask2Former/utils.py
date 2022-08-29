#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 16:17:00 2021

@author: tombu
"""
import numpy as np
from collections import OrderedDict 
from scipy.spatial import transform
import os

def read_images(path, subdir = 'sparse', file = 'images.txt'):
    """
    input: root path
    returns: dictionaries of image file names to their colmap IDs
                             image file names to their 3D poses
                             image file names to the 2D features 
    """     
    img2pose = {}
    imgto2dfeatures = {}
    img2id = {}
    path  = os.path.join(path, subdir, file)
    with open(path) as f:
        lines = f.read().splitlines()

    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    for i,line in enumerate(lines):
        if line[0] == '#':
            continue
        else:
            if i % 2 == 0:
                fields = line.split(' ')
                #NAME
                image_name = os.path.basename(fields[-1])
                #IMAGE_ID
                image_id = int(fields[0])
                quat = fields[1:5]
                quaternion = np.array([float(q) for q in quat])
                trans = fields[5:8]
                translation = np.array([float(t) for t in trans])
                camera_id = int(fields[8])
                #maps the image_name (jpg file) to it's pose
                img2pose[image_name] = [quaternion, translation, image_id, camera_id]
                #maps the name to an image id
                img2id[image_name] = image_id
            else:
                fields = line.split(' ')
                points_2d = np.array([float(pt) for pt in fields])
                points_2d = np.reshape(points_2d, (-1, 3))
                #maps the name to 2d points in the image
                imgto2dfeatures[image_name] = points_2d
    return img2pose, img2id, imgto2dfeatures

def read_3D(path):
    """
    input: path to root folder
    return: dictionary that maps 3D point ID to it's X, Y, Z value
    """
    pID2XYZ = OrderedDict()
    # 3D point list with one line of data per point:
    #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    #TRACK[] is a list of images and their points which contain the 3D point
    path  = os.path.join(path, 'sparse', 'points3D.txt')

    with open(path) as f:
        lines = f.read().splitlines()
    for i,line in enumerate(lines):
        if line[0] == '#':
            continue
        else: 
            fields = line.split(' ')
            points_attr = np.array([float(pt) for pt in fields if pt.isnumeric()])
            #we discard the tracks, and the RGB value
            pID2XYZ[points_attr[0]] = np.array([points_attr[1],points_attr[2],points_attr[3]])
    return pID2XYZ


def filter_3d_points(path, id3d, subdir = 'sparse', no_filter = False):
    """
    Filter the 3D points by their IDs

    Parameters
    ----------
    path : str
        path to root.
    id3d : Nx1 int array
        3D point IDs that are desired.

    Returns
    -------
    valid_pts : Nx8 array
        POINT3D_ID, X, Y, Z, R, G, B, ERROR, camera id, 2d point id, camera id, 2d point
    """
    path  = os.path.join(path, subdir, 'points3D.txt')
    points3d = np.loadtxt(path, usecols = (0, 1, 2, 3, 4, 5, 6, 7))
    N = points3d.shape[0]
    points3d = np.hstack([points3d, np.ones((N, 5))])

    if no_filter:
        return points3d
    #returns intersection values, left_indices, right_indices
    _, idx, _ = np.intersect1d(points3d[:, :1].flatten(), id3d.flatten(), return_indices=True)
    valid_pts = points3d[idx, :]
    return valid_pts

def read_cameras(path, subdir = 'sparse'):
    """
    Parameters
    ----------
    path : str
        path to root.

    Returns
    -------
    cameras : np array
        Nx11 where N is the number of cameras.
        fx, fy, ux, uy, k1, k2, p1, p2, CAMERA_ID, WIDTH, HEIGHT

    """
    path  = os.path.join(path, subdir, 'cameras.txt')
    with open(path) as f:
        lines = f.read().splitlines()
    cameras = []
    for i,line in enumerate(lines):
        if line[0] == '#':
            continue
        else: 
            fields = line.split(' ')
            camera_type = fields[1]
            k1, k2, p1, p2, fx, fy, ux, uy = 0, 0, 0, 0, 0, 0, 0, 0
            CAMERA_ID, WIDTH, HEIGHT = fields[0], fields[2], fields[3]
            if camera_type == 'SIMPLE_RADIAL':
                f, cx, cy, k = fields[4], fields[5], fields[6], fields[7]
                fx = fy = f
                ux = cx
                uy = cy
                k1 = k
            elif camera_type == 'RADIAL':
                f, cx, cy, k1, k2 = fields[4], fields[5], fields[6], fields[7], fields[8]
                fx = fy = f
                ux = cx
                uy = cy
            elif camera_type == 'OPENCV':
                fx, fy, ux, uy, k1, k2, p1, p2 = fields[4], fields[5], fields[6], fields[7], fields[8], fields[9], fields[10], fields[11]
            elif camera_type == 'PINHOLE':
                fx, fy, cx, cy = fields[4], fields[5], fields[6], fields[7]
                ux = cx
                uy = cy
            else:
                raise Exception('camera type not recognized')
            cameras.append([float(x) for x in [fx, fy, ux, uy, k1, k2, p1, p2, CAMERA_ID, WIDTH, HEIGHT]])
    cameras = np.array(cameras)
    return cameras

def get_intrinsic(parameters):
    """
    Parameters
    ----------
    parameters : float
        #fx, fy, ux, uy, k1, k2, p1, p2

    Returns
    -------
    intrinsic : 3x3 matrix
        camera intrinsic matrix.
    """
    intrinsic = np.eye(3, 3)
    intrinsic[0, 0] = parameters[0]
    intrinsic[1, 1] = parameters[1]
    intrinsic[0, 2] = parameters[2]
    intrinsic[1, 2] = parameters[3]
    return intrinsic
    
def get_extrinsic(pose):
    """
    Parameters
    ----------
    pose : float
        #QW, QX, QY, QZ, TX, TY, TZ
    Returns
    -------
    extrinsic : 3x4 matrix
        camera extrinsic matrix.
    """
    q = pose[:4]
    #make scalar last
    q = q[[1, 2, 3, 0]]
    #get rotation matrix from the quaternion. world to cam
    R_w2c = transform.Rotation.from_quat(q)
    
    # world to cam translation
    t= pose[4:]
    extrinsic = np.zeros((3, 4))        
    extrinsic[0:3, 0:3] = R_w2c.as_matrix()
    extrinsic[0:3, 3] = t     
    return extrinsic

def project_3d_points(points_id, extrinsic, parameters, force_pinhole = False):
    """
    Parameters
    ----------
    points_id : Nx4 array where N is the number of points. X, Y, Z coordinates, ID
    
    parameters : float
    #fx, fy, ux, uy, k1, k2, p1, p2, CAMERA_ID, WIDTH, HEIGHT

    Returns
    -------
    pixel : Nx2 array where N is the number of points. X, Y coordinates
    id3d : Nx1 array of IDs of 3D points
    """
    N = points_id.shape[0]
    id3d = points_id[:, 3:4]
    points = points_id[:, :3]
    #homogeneous coordinates
    points = np.hstack([points, np.ones((N, 1))]).T
    
    #project into camera frame. Used to filter out points behind the camera at the end
    points_in_camera_frame = extrinsic @ points
    points_in_camera_frame = points_in_camera_frame.T
    

    if force_pinhole == False:
        pixelw = extrinsic @ points
        #divide points by z to normalize
        pixelw = pixelw/pixelw[2:3, :]
        pixelw = pixelw.T
        
        #distort
        k1 = parameters[4]#-0.292631
        k2 = parameters[5]#0.063197
        p1 = parameters[6]#0.000750
        p2 = parameters[7]#0.000354
        k3 = 0
        fx = parameters[0]
        fy = parameters[1]
        ux = parameters[2]
        uy = parameters[3]
        
        x = pixelw[:, 0]
        xCorrected = pixelw[:, 0]
        y = pixelw[:, 1]
        yCorrected = pixelw[:, 1]
        
        r2 = x*x + y*y
        xCorrected = x * (1. + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
        yCorrected = y * (1. + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
        xCorrected = xCorrected + (2. * p1 * x * y + p2 * (r2 + 2. * x * x))
        yCorrected = yCorrected + (p1 * (r2 + 2. * y * y) + 2. * p2 * x * y)
        
        xCorrected = xCorrected * fx + ux
        yCorrected = yCorrected * fy + uy
        
        pixel = np.stack([xCorrected, yCorrected], axis = 1)
        
        # w0 = parameters[9]
        # h0 = parameters[10]
        # inds = np.where((pixel[:, 0] < w0) & (pixel[:, 0] >= 0) & (pixel[:, 1] < h0) & (pixel[:, 1] >= 0) & (points_in_camera_frame[:, 2] >= 0))[0]
        # pixel = pixel[inds, :]
        # id3d = id3d[inds, :]
        # return pixel, id3d
    else:
        intrinsic = get_intrinsic(parameters)
        pixel = intrinsic @ extrinsic @ points
        pixel = pixel/pixel[2, :]
        pixel = pixel.T
        
        # w0 = parameters[9]
        # h0 = parameters[10]
        # inds = np.where((pixel[:, 0] < w0) & (pixel[:, 0] >= 0) & (pixel[:, 1] < h0) & (pixel[:, 1] >= 0) & (points_in_camera_frame[:, 2] >= 0))[0]
        # pixel = pixel[inds, :]    
        # id3d = id3d[inds, :]
        # return pixel, id3d
    w0 = parameters[9]
    h0 = parameters[10]
    inds = np.where((pixel[:, 0] < w0) & (pixel[:, 0] >= 0) & (pixel[:, 1] < h0) & (pixel[:, 1] >= 0) & (points_in_camera_frame[:, 2] >= 0))[0]
    pixel = pixel[inds, :]    
    id3d = id3d[inds, :]
    return pixel, id3d

def get_subdir(path):
	"""
	Given a path, retrieve subdirectories
	"""
	li = list(filter(os.path.isdir, [os.path.join(path,x) for x in os.listdir(path)]))
	return [os.path.basename(x) for x in li]
