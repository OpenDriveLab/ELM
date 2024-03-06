import pickle
import numpy as np
from lidar_box3d import BaseInstance3DBoxes
from pyquaternion import Quaternion
import torch
import json
import tqdm
import cv2
import os
import mmcv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from nuscenes import NuScenes



map_loc = { 1: 0, 5: 0, 7: 0, 8: 0, 10: 0, 11: 0, 13: 0, 19: 0, 20: 0, 0: 0, 29: 0, 31: 0, 9: 1, 14: 2, 15: 3, 16: 3, 17: 4, 18: 5, 21: 6, 2: 7, 3: 7, 4: 7, 6: 7, 12: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 30: 16}
fore_loc = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
back_loc = [11, 12, 13, 14, 15, 16]


class PrepareQA:
    def __init__(self):
        # current information
        self.current_boxes = None
        self.current_coor = None
        self.current_gt_inds = None


    def forward(self, info, lidarseg, category):
        """
        input: info: list of info for 7 frames
        It includes 6 frames for history and 1 frame for current
        We want to generate 5 questions and answers for the past locations of the current objects
        """
        # Load box and labels: gt_bboxes_3d:(Nx9), gt_labels_3d:(N,)
        mask = info['valid_flag']
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_labels_3d = info['gt_names'][mask]

        gt_velocity = info['gt_velocity'][mask]
        nan_mask = np.isnan(gt_velocity[:, 0])
        gt_velocity[nan_mask] = [0.0, 0.0]
        gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        gt_bboxes_3d = BaseInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5))

        # (N, 8, 3)
        file_client_args=dict(backend='disk')
        file_client = mmcv.FileClient(**file_client_args)
        pts_bytes = file_client.get(os.path.join('data', info['lidar_path'].split('data/')[-1]))
        points = np.frombuffer(pts_bytes, dtype=np.float32)

        # lidarseg
        pts_bytes = file_client.get(os.path.join('data/nuscenes/seg', lidarseg[f'{info["lidarseg_token"]}']['filename']))
        seg_points = np.frombuffer(pts_bytes, dtype=np.uint8)

        points = torch.from_numpy(np.copy(points).reshape(-1, 5)[:, :3])
        seg_points = torch.from_numpy(np.copy(seg_points).reshape(-1, 1))

        corners = gt_bboxes_3d.corners
        
        # extrinsics
        lidar2lidarego = np.eye(4, dtype=np.float32)
        lidar2lidarego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        lidar2lidarego[:3, 3] = info['lidar2ego_translation']
        lidar2lidarego = torch.from_numpy(lidar2lidarego)

        lidarego2global = np.eye(4, dtype=np.float32)
        lidarego2global[:3, :3] = Quaternion(info['ego2global_rotation']).rotation_matrix
        lidarego2global[:3, 3] = info['ego2global_translation']
        lidarego2global = torch.from_numpy(lidarego2global)

        # intrinsic
        cam2camego = np.eye(4, dtype=np.float32)
        cam2camego[:3, :3] = Quaternion(info['cams']["CAM_FRONT"]['sensor2ego_rotation']).rotation_matrix
        cam2camego[:3, 3] = info['cams']['CAM_FRONT']['sensor2ego_translation']
        cam2camego = torch.from_numpy(cam2camego)
    
        camego2global = np.eye(4, dtype=np.float32)
        camego2global[:3, :3] = Quaternion(info['cams']['CAM_FRONT']['ego2global_rotation']).rotation_matrix
        camego2global[:3, 3] = info['cams']['CAM_FRONT']['ego2global_translation']
        camego2global = torch.from_numpy(camego2global)

        cam2img = np.eye(4, dtype=np.float32)
        cam2img[:3, :3] = info['cams']['CAM_FRONT']['cam_intrinsic']
        cam2img = torch.from_numpy(cam2img)

        lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
            lidarego2global.matmul(lidar2lidarego))
        lidar2img = cam2img.matmul(lidar2cam)

        points_img = points.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
        points_img = torch.cat([points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]], 1)

        corner_mask = (corners[:, 0, 1] < 0) & (corners[:, 1, 1] < 0) & (corners[:, 2, 1] < 0) & (corners[:, 3, 1] < 0) & (corners[:, 4, 1] < 0) & (corners[:, 5, 1] < 0) & (corners[:, 6, 1] < 0) & (corners[:, 7, 1] < 0)
        corners = corners[~corner_mask]
        if corners.shape[0] == 0:
            corners = torch.zeros((1, 8, 3))
        corners = corners.reshape(-1, 3)
        
        corners_img = corners.matmul(lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
        corners_img = torch.cat([corners_img[:, :2] / corners_img[:, 2:3], corners_img[:, 2:3]], 1)
        corners_img = corners_img.reshape(-1, 8, 3)[:,:,:2]

        # limit to the front view
        height, width = 900, 1600
        coor = points_img[:, :2]
        depth = points_img[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (depth >= 1.0)
        points_img = points_img[kept1]
        points = points[kept1]
        seg_points = seg_points[kept1]

        box_loc = corners_img.numpy()
        uv_loc = points_img.numpy()
        points = points.numpy()
        uv_loc = np.concatenate([uv_loc, points], axis=1)


        self.current_coor = coor
        self.current_corners = corners
        self.depth = depth

        # fileter the foreground points
        vectorized_map = np.vectorize(lambda x: map_loc[x])
        seg_points = vectorized_map(seg_points.numpy())

        seg_points = seg_points.squeeze()
        mask = (seg_points >=1) & (seg_points <= 10)
        f_loc = uv_loc[mask]

        mask = (seg_points >=11)
        b_loc = uv_loc[mask]

        uv_loc = f_loc
        if uv_loc.shape[0] > 1:
            radius = 0.3 #100
            downsampled_points = []
            point_sampled = uv_loc[0]
            point_sampled = point_sampled[np.newaxis, :]
            for i in range(len(uv_loc)):
                point = uv_loc[i]
                loc = point_sampled[:,3:6]
                distances = ((loc[:,0] - point[3])**2 + (loc[:,1] - point[4])**2 + (loc[:,2] - point[5])**2)**(1/3)
                within_radius = np.any(distances < radius)
                if not within_radius:
                    point_sampled = np.concatenate([point_sampled, point[np.newaxis, :]], axis=0)
            f_loc = point_sampled


        uv_loc = b_loc
        if uv_loc.shape[0] > 1:
            radius = 2 #100
            downsampled_points = []
            point_sampled = uv_loc[0]
            point_sampled = point_sampled[np.newaxis, :]
            for i in range(len(uv_loc)):
                point = uv_loc[i]
                loc = point_sampled[:,3:6]
                distances = ((loc[:,0] - point[3])**2 + (loc[:,1] - point[4])**2 + (loc[:,2] - point[5])**2)**(1/3)
                within_radius = np.any(distances < radius)
                if not within_radius:
                    point_sampled = np.concatenate([point_sampled, point[np.newaxis, :]], axis=0)
            b_loc = point_sampled

        image = mpimg.imread(cam_path)

        fig, ax = plt.subplots()
        ax.imshow(image)

        plt.scatter(f_loc[:, 0], f_loc[:, 1], c=f_loc[:, 2], cmap='viridis', alpha=1, s=1)
        plt.scatter(b_loc[:, 0], b_loc[:, 1], c=b_loc[:, 2], cmap='viridis', alpha=1, s=1)
        cbar = plt.colorbar()
        cbar.set_label('Color')
        plt.title('Scatter Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()


        output = f_loc

        return output


def add_token(infos):
    nuscenes_version = 'v1.0-trainval'
    dataroot = 'data/nuscenes/'
    nuscenes = NuScenes(nuscenes_version, dataroot)
    for set in ['train', 'val']:
        for id in range(len(infos)):
            info = infos[id]
            # get sweep adjacent frame info
            sample = nuscenes.get('sample', info['token'])
            lidarseg_token = sample['data']['LIDAR_TOP']
            info['lidarseg_token'] = lidarseg_token

    return infos


if __name__ == '__main__':
    split = "train"

    nuscenes_version = 'v1.0-trainval'
    nusc = NuScenes(version=nuscenes_version, dataroot='data/nuscenes', verbose=True)

    with open(f"data/nuscenes/nuscenes_infos_{split}.pkl", "rb") as f:
        infos = pickle.load(f)["infos"]
    infos = add_token(infos)
    print(f"Add token finished {infos[0]['lidarseg_token']}")
    
    with open(f"data/nuscenes/seg/v1.0-trainval/category.json", "rb") as f:
        category = json.load(f)
    with open(f"data/nuscenes/seg/v1.0-trainval/lidarseg.json", "rb") as f:
        lidarseg = json.load(f)
    segments = {}
    for seg in lidarseg:
        segments[seg['token']] = seg

    length = len(infos)
    QAGenerate = PrepareQA()

    for i, info in enumerate(tqdm.tqdm(infos)):
        output = QAGenerate.forward(info, segments, category)
    