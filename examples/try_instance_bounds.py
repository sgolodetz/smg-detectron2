import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from detectron2.structures import Instances
from typing import List, Tuple

from smg.detectron2 import InstanceSegmenter
from smg.open3d import VisualisationUtil
from smg.openni import OpenNICamera
from smg.utility import GeometryUtil


def make_instance_bounds(instance: InstanceSegmenter.Instance, depth_image: np.ndarray,
                         ws_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    combined_mask: np.ndarray = np.where((instance.pred_mask != 0) & (depth_image != 0), 255, 0).astype(np.uint8)
    combined_mask = cv2.erode(combined_mask, (5, 5))

    cv2.imshow("Combined Mask", combined_mask)
    cv2.waitKey(1)

    plt.hist(depth_image[np.where(combined_mask)], 50)
    plt.title(f"{instance.pred_class}")
    plt.xlabel("Depth")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()

    masked_ws_points: np.ndarray = np.where(np.atleast_3d(combined_mask), ws_points, np.full(ws_points.shape, np.nan))
    return np.array([
        np.nanmin(masked_ws_points[:, :, 0]),
        np.nanmin(masked_ws_points[:, :, 1]),
        np.nanmin(masked_ws_points[:, :, 2]),
    ]), np.array([
        np.nanmax(masked_ws_points[:, :, 0]),
        np.nanmax(masked_ws_points[:, :, 1]),
        np.nanmax(masked_ws_points[:, :, 2]),
    ])


def main() -> None:
    segmenter: InstanceSegmenter = InstanceSegmenter.make_mask_rcnn()

    with OpenNICamera(mirror_images=True) as camera:
        while True:
            colour_image, depth_image = camera.get_images()
            cv2.imshow("Colour Image", colour_image)
            cv2.imshow("Depth Image", depth_image / 2)

            raw_instances: Instances = segmenter.segment_raw(colour_image)
            segmented_image: np.ndarray = segmenter.draw_raw_instances(raw_instances, colour_image)
            cv2.imshow("Segmented Image", segmented_image)

            c: int = cv2.waitKey(1)
            if c == ord('v'):
                break

        intrinsics: Tuple[float, float, float, float] = camera.get_colour_intrinsics()
        pcd: o3d.geometry.PointCloud = VisualisationUtil.make_rgbd_image_point_cloud(
            segmented_image, depth_image, intrinsics
        )
        to_visualise: List[o3d.geometry.Geometry] = [pcd]

        height, width = depth_image.shape
        ws_points: np.ndarray = np.zeros((height, width, 3), dtype=float)
        GeometryUtil.compute_world_points_image_fast(depth_image, np.eye(4), intrinsics, ws_points)

        instances: List[InstanceSegmenter.Instance] = segmenter.parse_raw_instances(raw_instances)
        for instance in instances:
            mins, maxs = make_instance_bounds(instance, depth_image, ws_points)
            to_visualise.append(o3d.geometry.AxisAlignedBoundingBox(mins, maxs))

        VisualisationUtil.visualise_geometries(to_visualise)


if __name__ == "__main__":
    main()
