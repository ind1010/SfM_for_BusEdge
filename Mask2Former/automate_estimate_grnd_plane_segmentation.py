# common libraries
import numpy as np
import os
import argparse
from scipy.spatial import transform
import shutil
from sklearn.neighbors import NearestNeighbors
import cv2
import torch, torchvision

# detectron2 imports
import detectron2
from detectron2.utils.visualizer import _PanopticPrediction
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.projects.deeplab import add_deeplab_config
coco_metadata = MetadataCatalog.get("mapillary_vistas_panoptic_val")
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="mask2former")

# import Mask2Former
# from mask2former import add_maskformer2_config

# other scripts
from utils import filter_3d_points
from utils import get_subdir


def init_model():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    # coco
    # cfg.merge_from_file("configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
    # cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl'
    # mapillary vistas
    cfg.merge_from_file("configs/mapillary-vistas/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_300k.yaml")
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/mapillary_vistas/panoptic/maskformer2_swin_large_IN21k_384_bs16_300k/model_final_132c71.pkl'
    # cityscapes
    # cfg.merge_from_file("configs/cityscapes/semantic-segmentation/swin/maskformer2_swin_small_bs16_90k.yaml")
    # cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/maskformer2_swin_tiny_bs16_90k/model_final_2d58d4.pkl'
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
    predictor = DefaultPredictor(cfg)
    return cfg, predictor

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
                image_name = fields[-1]
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

def get_mask(predictor, im_path, cfg):
    im = cv2.imread(im_path)
    sem_seg = predictor(im)["sem_seg"]
    
    # pred = _PanopticPrediction(panoptic_seg.to("cpu"), segments_info, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))

    # classes = np.array([x[1]['category_id'] for x in pred.semantic_masks()])
    # instances = np.array([x[0] for x in pred.semantic_masks()])
    stuff_classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).stuff_classes
    
    idx = (np.arange(len(stuff_classes)) == stuff_classes.index('Road'))
    stuff_instances = sem_seg[idx]
    
    if len(stuff_instances) == 0:
        pred_mask = np.ones_like(im)[:, :, 0]
    else:
        instances = stuff_instances
        pred_mask = instances[0]
        pred_mask = pred_mask.astype(int)
    return pred_mask

def read_predictions(args, parent, imgto2dfeatures, cfg, predictor):
    """
    get the 3d ID of the ground points from image segmentations
    """
    total_ground_points = []
    for image_name, points_2d in imgto2dfeatures.items():
        if os.path.exists(os.path.join(args.ground_path, parent, image_name)):
            pred_mask = cv2.imread(os.path.join(args.ground_path, parent, image_name))
            pred_mask = pred_mask[:, :, 0]
            # print(os.path.join(args.ground_path, parent, parent))
        else:
            # print(os.path.join(args.ground_path, parent))
            img_path = os.path.join(args.image_path, parent, image_name)
            pred_mask = get_mask(predictor, img_path, cfg)
            pred_mask = pred_mask * 255
            os.makedirs(os.path.join(args.ground_path, parent, os.path.dirname(image_name)), exist_ok=True)    
            cv2.imwrite(os.path.join(args.ground_path, parent, image_name), np.stack([pred_mask, pred_mask, pred_mask], axis = 2))            
        image_points = points_2d[points_2d[:, 2] != -1]
        image_points = image_points.astype(int)
        image_points[:, [1, 0]] = image_points[:, [0, 1]]
        image_points_bool = pred_mask[image_points[:, 0], image_points[:, 1]]
        ground_points = image_points[image_points_bool > 0]
        if len(ground_points) != 0:
            total_ground_points.append(ground_points[:, 2])
    return np.concatenate(total_ground_points)

def remove_outlier(points3d_instance):
    nbrs = NearestNeighbors(n_neighbors = 5)
    X = points3d_instance[:, 1:4]
    nbrs.fit(X)
    distances, indexes = nbrs.kneighbors(X)
    distances = distances[:, 1:]
    mean = np.mean(distances.mean(axis = 1))
    std = np.std(distances.mean(axis = 1))
    inlier_index = distances.mean(axis = 1) < (mean)
    return points3d_instance[inlier_index]

def rotation_matrix_from_vectors(vec1, vec2):
    """

    
    Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def align_with_plane(coef):
    "given parameters of a plane, provide the transformation to make things align with it"
    tf1 = np.eye(4)
    tf1[2, 3] = coef[-1] /coef[2] 
    vt = rotation_matrix_from_vectors(coef[:3], np.array([0,0,1]))
    tf2 = np.eye(4)
    tf2[:3, :3] = vt[:3, :3]
    tf = tf2 @ tf1
    return tf

def normalize(colmap_3d_one_cam):
    minimum = np.min(colmap_3d_one_cam[:, 0])
    colmap_3d_one_cam[:, 0] = colmap_3d_one_cam[:, 0] - minimum
    maximum = np.max(colmap_3d_one_cam[:, 0])
    colmap_3d_one_cam[:, 0] = colmap_3d_one_cam[:, 0] * 1/ maximum
    
    minimum = np.min(colmap_3d_one_cam[:, 1])
    colmap_3d_one_cam[:, 1] = colmap_3d_one_cam[:, 1] - minimum
    maximum = np.max(colmap_3d_one_cam[:, 1])
    colmap_3d_one_cam[:, 1] = colmap_3d_one_cam[:, 1] * 1/ maximum
    
    minimum = np.min(colmap_3d_one_cam[:, 2])
    colmap_3d_one_cam[:, 2] = colmap_3d_one_cam[:, 2] - minimum
    maximum = np.max(colmap_3d_one_cam[:, 2])
    if maximum != 0:
        colmap_3d_one_cam[:, 2] = colmap_3d_one_cam[:, 2] * 1/ maximum
    return colmap_3d_one_cam

def transform_cam(tf, lines):
    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    for i,line in enumerate(lines):
        if line[0] == '#':
            continue
        else:
            if i % 2 == 0:
                fields = line.split(' ')
                
                #get the image pose information
                quat = fields[1:5]
                quat = np.array([float(q) for q in quat])
                trans = fields[5:8]
                t = np.array([float(t) for t in trans])    
                
                #get pose in world frame
                quat = [quat[1], quat[2], quat[3], quat[0]]
                R = transform.Rotation.from_quat(quat)
                R2 = R.as_matrix().T
                tvec2 = -R.as_matrix().T.dot(t) 
            
                # transform the pose into the new coordinate
                tf2 = np.eye(4)
                tf2[:3, :3] = R2
                tf2[:3, 3] = tvec2
                tf2 = tf @ tf2 
                
                #pose in world frame still
                R2 = tf2[:3, :3]
                tvec2 = tf2[:3, 3]
                
                #save rotation matrix in ego frame
                quat = transform.Rotation.from_matrix(R2.T)
                quat = quat.as_quat()
                quat = [quat[3], quat[0], quat[1], quat[2]]
                quat = [str(q) for q in quat]
                fields[1:5] = quat
                
                #save translation matrix in ego frame
                tvec2 = -R2.T.dot(tvec2) 
                trans = [str(t) for t in tvec2.tolist()]
                fields[5:8] = trans
                
                lines[i] = ' '.join(fields)
    return lines


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        # points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)
        points = points / np.linalg.norm(points[:3, :], axis = 0, keepdims=True).repeat(3, 0).reshape(3, nbr_points)
    return points

def dist(data, coef):
    A = np.ones((data.shape[0], 1))
    A = np.concatenate([data, A], axis = 1)
    d = np.abs(np.dot(A, coef)) / np.sqrt(np.sum((coef ** 2)[:3]))
    return d

def fit_plane(data):
    A = np.ones((data.shape[0], 1))
    A = np.concatenate([data, A], axis = 1)
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    coef = vt[3, :]    
    return coef

def ransac(data, threshold = 0.001):
    best_cnt = 0
    iteration = 2000
    print(data.shape)
    for i in range(iteration):
        sample = data[np.random.randint(data.shape[0], size=4), :]
        coef = fit_plane(sample)
        cnt = np.sum(dist(data, coef) < threshold)
        if cnt > best_cnt:
            best_cnt = cnt
            best_coef = coef
    return best_coef

def create_image_pose(img2pose):
    img_pose = []
    for img_fc in img2pose.keys():
        q = img2pose[img_fc][0]
        quat = [q[1], q[2], q[3], q[0]]
        R = transform.Rotation.from_quat(quat)
        t = img2pose[img_fc][1]

        #point in world frame
        colmap_pt = - R.as_matrix().T.dot(t) 
        img_pose.append(colmap_pt)
    img_pose = np.array(img_pose)

    
    #make it into colmap's format
    r, c = img_pose.shape
    rng = np.random.default_rng(12345)
    rints = rng.integers(low=1, high=1000000, size=img_pose.shape[0])
    img_pose = np.hstack([rints[:, np.newaxis], img_pose, np.zeros((r, 1)), np.ones((r, 1)) * 255, np.zeros((r, 1))])
    r, c = img_pose.shape
    img_pose = np.hstack([img_pose, np.ones((r, 13-c))])
    
    return img_pose

def create_plane_points(img_pose):
    x_range = max(img_pose[:, 1]) - min(img_pose[:, 1])
    y_range = max(img_pose[:, 2]) - min(img_pose[:, 2])
    x = np.linspace(min(img_pose[:, 1]) - x_range * 0.1, max(img_pose[:, 1])  + x_range * 0.1, 200)
    y = np.linspace(min(img_pose[:, 2])- y_range * 0.1, max(img_pose[:, 2])  + y_range * 0.1, 200)
    xx, yy = np.meshgrid(x, y)
    z = np.zeros_like(xx)
    
    plane = np.stack([xx.flatten(), yy.flatten(), z.flatten()], axis = 1)    
    #make it into colmap's format
    r, c = plane.shape
    rng = np.random.default_rng(12345)
    rints = rng.integers(low=1, high=1000000, size=plane.shape[0])
    plane = np.hstack([rints[:, np.newaxis], plane, np.zeros((r, 1)), np.ones((r, 1)) * 255, np.zeros((r, 1))])
    r, c = plane.shape
    plane = np.hstack([plane, np.ones((r, 13-c))])
    return plane    

def gen_points_output(args, parent, points3d_final):
    fmt = '%d', '%.18f', '%.18f', '%.18f', '%d', '%d', '%d', '%.18f', '%d', '%d', '%d', '%d', '%d'
    np.savetxt(os.path.join(parent, args.output, 'points3D.txt'), points3d_final, fmt = fmt) 
  
def gen_image_output(args, parent, tf):
    #convert all poses of the images 
    path  = os.path.join(args.path, parent, 'images.txt')
    with open(path) as f:
        lines = f.read().splitlines()
    lines = transform_cam(tf, lines)
    with open(os.path.join(parent, args.output, 'images.txt') , 'w') as filehandle:
        filehandle.writelines("%s\n" % place for place in lines)    

  
class Argument():
    def __init__(self, path, image_path, output, ground_path):
        self.path = path
        self.image_path = image_path
        self.output = output
        self.ground_path = ground_path

def estimate_ground_plane(path, image_path, output, ground_path):
    args = Argument(path, image_path, output, ground_path)
    cfg, predictor = init_model()
    args.cfg = cfg
    args.predictor = predictor 

    
    #load the points
    path  = os.path.join(args.path, 'points3D.txt')
    #sparse point clouds have error, trac, etc. which we don't need
    points3d_sparse = np.loadtxt(path, usecols = (0, 1, 2, 3, 4, 5, 6, 7))
    r, c = points3d_sparse.shape
    points3d_sparse = np.hstack([points3d_sparse, np.ones((r, 13-c))])


    #get the pose of each imge
    img2pose, img2id, imgto2dfeatures = read_images(args.path, subdir = '', file = 'images.txt')
    img_pose = create_image_pose(img2pose)
    
    #get ground points
    instance_points = read_predictions(args, '', imgto2dfeatures, args.cfg, args.predictor)
    id3d = np.unique(instance_points)[:, np.newaxis]
    points3d_instance = filter_3d_points(args.path, id3d, subdir = '')
    points3d_instance = remove_outlier(points3d_instance)

    #estimate the ground plane
    coef = ransac(points3d_instance[:, 1:4])    
    tf = align_with_plane(coef)

    #transform both points and cameras
    points3d_sparse[:, 1:4] = view_points(points3d_sparse[:, 1:4].T, tf, normalize=False).T
    img_pose[:, 1:4] = view_points(img_pose[:, 1:4].T, tf, normalize=False).T

    #add the plane points
    plane = create_plane_points(img_pose)
    points3d_final = np.concatenate([points3d_sparse, plane], axis = 0)

    #create output
    os.makedirs(args.output, exist_ok=True)    
    shutil.copy2(os.path.join(args.path, 'cameras.txt'), os.path.join(args.output, 'cameras.txt'))
    gen_points_output(args, '', points3d_final)
    gen_image_output(args, '', tf)    
    






    