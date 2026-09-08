import cv2
import time
import numpy as np
import viser
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
import fpsample


def depth2pc(depth, camera_intrinsics, camera_pose=np.eye(4)):
    height, width = depth.shape
    fx, fy, cx, cy = camera_intrinsics
    z = depth.flatten()

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.flatten()
    v = v.flatten()

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points_camera = np.stack((x, y, z), axis=1)[z > 0]

    points_camera_h = np.concatenate((points_camera, np.ones((points_camera.shape[0], 1))), axis=1)  # Nx4
    points_world = (camera_pose @ points_camera_h.T).T[:, :3]

    return points_world


def point_cloud_downsample(point_cloud, num_points):
    # bounding box filter
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    bounding_box_mask = (x > 0.2) & (x < 0.8) & (y > -0.23) & (y < 0.25) & (z > 0.3085) & (z < 0.65)
    point_cloud = point_cloud[bounding_box_mask]


    # FPS sampling
    if len(point_cloud) < num_points:
        point_cloud = np.concatenate([point_cloud] * (num_points // len(point_cloud) + 1), axis=0)
    sample_idx = fpsample.bucket_fps_kdtree_sampling(point_cloud, num_points)
    point_cloud = point_cloud[sample_idx]

    return point_cloud


camera_intrinsics = (1348.70988187, 1348.70988187, 967.53239972, 549.01237165)
X_root_camera = np.array([
    [-0.99926635,  0.02147153,  0.03171328,  0.52130602],
    [-0.00456881,  0.7553141 , -0.65534703,  0.44153803],
    [-0.03802479, -0.65501113, -0.75466187,  0.75009051],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])
# rot_mat = X_root_camera[:3, :3]
# rot_euler = R.from_matrix(rot_mat).as_euler('xyz', degrees=True)
# print(rot_euler)
# rot_euler[0] -= 0.2
# rot_euler[1] += 0.5
# rot_mat = R.from_euler('xyz', rot_euler, degrees=True).as_matrix()
# print(rot_mat)
# X_root_camera[:3, :3] = rot_mat


class RealSense(object):
    def __init__(
        self,
        fps=30,
        depth_width=320,
        depth_height=240,
        num_points=1024
    ):
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.num_points = num_points

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)
        self.align = rs.align(rs.stream.color)

    def start(self):
        profile = self.pipeline.start(self.config)

        # get intrinsics
        frames = self.pipeline.wait_for_frames()
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        depth_frame = frames.get_depth_frame()
        depth_intrinsics = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
        self.depth_intrinsics = (depth_intrinsics.fx, depth_intrinsics.fy, depth_intrinsics.ppx, depth_intrinsics.ppy)

    def stop(self):
        self.pipeline.stop()

    def get_frame(self, require_pc=False):
        while True:
            frames = self.pipeline.wait_for_frames()

            timestamp = frames.get_timestamp() / 1000  # ms -> s
            depth_frame = frames.get_depth_frame()

            if depth_frame:
                break

        depth_image = np.array(depth_frame.get_data())

        point_cloud = point_cloud_downsample(depth2pc(
            depth_image * self.depth_scale,
            self.depth_intrinsics,
            X_root_camera
        ), self.num_points) if require_pc else None

        return {
            'timestamp': timestamp,
            'depth': depth_image,
            'depth_scale': self.depth_scale,
            'point_cloud': point_cloud
        }


if __name__ == '__main__':
    camera = RealSense()
    camera.start()
    
    server = viser.ViserServer(host='127.0.0.1', port=8080)

    while True:
        frame = camera.get_frame(require_pc=True)
        # print('timestamp:', frame['timestamp'])

        server.scene.add_frame(
            f'camera_pose',
            wxyz=R.from_matrix(X_root_camera[:3, :3]).as_quat()[[3, 0, 1, 2]],
            position=X_root_camera[:3, 3],
            axes_length=0.2,
            axes_radius=0.006
        )

        server.scene.add_point_cloud(
            'pc',
            frame['point_cloud'],
            point_size=0.005,
            point_shape="circle",
            colors=(0, 0, 255)
        )
        time.sleep(0.033)
