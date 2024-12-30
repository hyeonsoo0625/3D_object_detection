import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

def lidar_bbox(sample_token):
    BASE_DIR = 'data/3d-object-detection-for-autonomous-vehicles/'

    train = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))

    object_columns = ['sample_id', 'object_id', 'center_x', 'center_y', 'center_z', 'width', 'length', 'height', 'yaw', 'class_name']
    objects = []
    for sample_id, ps in train.values[:]:
        object_params = ps.split()
        n_objects = len(object_params)
        for i in range(n_objects // 8):
            x, y, z, w, l, h, yaw, c = tuple(object_params[i * 8: (i + 1) * 8])
            objects.append([sample_id, i, x, y, z, w, l, h, yaw, c])
    train_objects = pd.DataFrame(objects, columns=object_columns)

    numerical_cols = ['object_id', 'center_x', 'center_y', 'center_z', 'width', 'length', 'height', 'yaw']
    train_objects[numerical_cols] = np.float32(train_objects[numerical_cols].values)

    sample_data = check_file('sample_data', BASE_DIR)
    calibrated_sensor = check_file('calibrated_sensor', BASE_DIR)
    sample_data_object = sample_data[sample_data['sample_token'] == sample_token]

    bin_objects = sample_data_object[sample_data_object['filename'].str.endswith('.bin')]

    for bin_name in bin_objects['filename']:
        lidar_points = load_lidar_data(os.path.join(BASE_DIR, 'train_lidar', bin_name.split('/')[-1]))
        
        object_axis = train_objects[train_objects['sample_id'] == sample_data_object[sample_data_object['filename'] == bin_name]['sample_token'].values[0]]

        bounding_volumes_array = np.array(object_axis[['center_x', 'center_y', 'center_z', 'width', 'length', 'height', 'yaw']].values)

        ego_pose = check_file('ego_pose', BASE_DIR)
        ego_pose_token = sample_data_object[sample_data_object['filename'] == bin_name]['ego_pose_token'].values[0]
        ego_pose_data = ego_pose[ego_pose['token'] == ego_pose_token]
        calibrated_sensor_data = calibrated_sensor[calibrated_sensor['token'] == bin_objects[bin_objects['filename'] == bin_name]['calibrated_sensor_token'].values[0]]
        sensor_rotation = np.array(calibrated_sensor_data['rotation'].values[0])
        sensor_translation = np.array(calibrated_sensor_data['translation'].values[0])

        sensor = check_file('sensor', BASE_DIR)
        sensor_data = sensor[sensor['token'] == calibrated_sensor_data['sensor_token'].values[0]]['channel']
        print('sensor_data : ', sensor_data)
        translation_ego_pose = np.array(ego_pose_data['translation'].values[0])
        rotation_ego_pose = np.array(ego_pose_data['rotation'].values[0])

        rotation_matrix = Quaternion(rotation_ego_pose).rotation_matrix

        rotation_matrix2 = Quaternion(sensor_rotation).rotation_matrix

        lidar_points[:, 0:3] = (rotation_matrix2 @ lidar_points[:, 0:3].T).T
        lidar_points[:, 0:3] += sensor_translation
        lidar_points[:, 0:3] = (rotation_matrix @ lidar_points[:, 0:3].T).T
        lidar_points[:, 0:3] += translation_ego_pose
            
        visualize_lidar_with_bounding_boxes(lidar_points, bounding_volumes_array)

def check_file(file, BASE_DIR):
        file_path = os.path.join(BASE_DIR, 'train_data', f'{file}.json')

        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df


def load_lidar_data(file_path):
    lidar_data = np.fromfile(file_path, dtype=np.float32)
    lidar_data = lidar_data.reshape(-1, 5)
    return lidar_data

def visualize_lidar_with_bounding_boxes(points, bounding_boxes):
    fig = go.Figure(data=[
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color=points[:, 3],
                colorscale='Viridis',
                opacity=0.5
            )
        )
    ])

    for box in bounding_boxes:
        cx, cy, cz, w, l, h, yaw = box
        corners = get_bounding_box_corners(cx, cy, cz, w, l, h, yaw)

        fig.add_trace(
            go.Mesh3d(
                x=corners[:, 0],
                y=corners[:, 1],
                z=corners[:, 2],
                color='red',
                opacity=0.3,
                alphahull=0
            )
        )
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='manual',
            aspectratio=dict(x=200, y=200, z=15),
            camera=dict(
                eye=dict(x=2800, y=700, z=-18)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    fig.show()

def get_bounding_box_corners(cx, cy, cz, l, w, h, yaw):
    rotation_mat = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    corners = np.array([
        [-w / 2, -l / 2, -h / 2],
        [w / 2, -l / 2, -h / 2],
        [w / 2, l / 2, -h / 2],
        [-w / 2, l / 2, -h / 2],
        [-w / 2, -l / 2, h / 2],
        [w / 2, -l / 2, h / 2],
        [w / 2, l / 2, h / 2],
        [-w / 2, l / 2, h / 2],
    ])
    rotated_corners = np.dot(corners, rotation_mat.T)
    translated_corners = rotated_corners + np.array([cx, cy, cz])

    return translated_corners


#####################################################################
""" Rendering Scene """
def render_scene(scene_token, base_dir='data/3d-object-detection-for-autonomous-vehicles/'):
    # Load required data
    sample = check_file('sample', base_dir)
    scene = check_file('scene', base_dir)
    
    # Get all sample tokens for the given scene, sorted by timestamp
    scene_data = scene[scene['token'] == scene_token]
    first_sample_token = scene_data['first_sample_token'].values[0]
    sample_sequence = []
    
    current_token = first_sample_token
    while current_token:
        sample_data = sample[sample['token'] == current_token]
        sample_sequence.append(current_token)
        current_token = sample_data['next'].values[0] if len(sample_data['next'].values) > 0 else None

    # Prepare frames for animation
    frames = []
    for sample_token in sample_sequence:
        lidar_data, bounding_boxes = get_lidar_and_boxes(sample_token, base_dir)
        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=lidar_data[:, 0],
                    y=lidar_data[:, 1],
                    z=lidar_data[:, 2],
                    mode='markers',
                    marker=dict(
                        size=1,
                        color=lidar_data[:, 3],
                        colorscale='Viridis',
                        opacity=0.5
                    )
                ),
                *[
                    go.Mesh3d(
                        x=box[:, 0],
                        y=box[:, 1],
                        z=box[:, 2],
                        color='red',
                        opacity=0.3,
                        alphahull=0
                    )
                    for box in [get_bounding_box_corners(*box) for box in bounding_boxes]
                ]
            ],
            name=f"frame_{sample_token}"
        )
        frames.append(frame)

    # Create animated plot
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode='markers',
                marker=dict(size=1, opacity=0.5)
            )
        ],
        layout=go.Layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='manual',
                aspectratio=dict(x=200, y=200, z=15),
                camera=dict(
                    eye=dict(x=2800, y=700, z=-18)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    buttons=[
                        dict(label='Play',
                             method='animate',
                             args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True)]),
                        dict(label='Pause',
                             method='animate',
                             args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')])
                    ]
                )
            ]
        ),
        frames=frames
    )

    fig.show()

def get_lidar_and_boxes(sample_token, base_dir):
    sample_data = check_file('sample_data', base_dir)

    train = pd.read_csv(os.path.join(base_dir, 'train.csv'))

    object_columns = ['sample_id', 'object_id', 'center_x', 'center_y', 'center_z', 'width', 'length', 'height', 'yaw', 'class_name']
    objects = []
    for sample_id, ps in train.values[:]:
        object_params = ps.split()
        n_objects = len(object_params)
        for i in range(n_objects // 8):
            x, y, z, w, l, h, yaw, c = tuple(object_params[i * 8: (i + 1) * 8])
            objects.append([sample_id, i, x, y, z, w, l, h, yaw, c])
    train_objects = pd.DataFrame(objects, columns=object_columns)

    numerical_cols = ['object_id', 'center_x', 'center_y', 'center_z', 'width', 'length', 'height', 'yaw']
    train_objects[numerical_cols] = np.float32(train_objects[numerical_cols].values)

    calibrated_sensor = check_file('calibrated_sensor', base_dir)
    ego_pose = check_file('ego_pose', base_dir)

    # Process sample
    sample_data_object = sample_data[sample_data['sample_token'] == sample_token]
    bin_objects = sample_data_object[sample_data_object['filename'].str.endswith('.bin')]

    lidar_points = []
    bounding_boxes = []

    for bin_name in bin_objects['filename']:
        lidar_points = load_lidar_data(os.path.join(base_dir, 'train_lidar', bin_name.split('/')[-1]))
        object_axis = train_objects[train_objects['sample_id'] == sample_token]
        bounding_boxes = object_axis[['center_x', 'center_y', 'center_z', 'width', 'length', 'height', 'yaw']].values

    return lidar_points, bounding_boxes
