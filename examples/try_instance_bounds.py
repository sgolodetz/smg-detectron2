import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import sys

from detectron2.structures import Instances
from sklearn.mixture import GaussianMixture
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

    depths = depth_image[np.where(combined_mask)]
    reshaped_depths = depths.reshape(-1, 1)
    gm: GaussianMixture = GaussianMixture().fit(reshaped_depths)  # Check for ValueErrors
    scores: np.ndarray = gm.score_samples(reshaped_depths)
    probs = np.exp(scores)
    probs = (probs - np.min(probs)) / (np.max(probs) - np.min(probs))
    depths = depths[np.where(probs >= 0.5)]
    min_depth = np.min(depths)
    max_depth = np.max(depths)
    combined_mask = np.where((depth_image >= min_depth) & (depth_image <= max_depth), combined_mask, 0).astype(np.uint8)
    print(f"{instance.pred_class}: {min_depth}, {max_depth}, {np.count_nonzero(combined_mask)}")

    print(f"{instance.pred_class}: {list(depths)}")
    plt.hist(depths, 50)
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
    np.set_printoptions(threshold=sys.maxsize)

    segmenter: InstanceSegmenter = InstanceSegmenter.make_mask_rcnn()

    with OpenNICamera(mirror_images=True) as camera:
        while True:
            colour_image, depth_image = camera.get_images()
            cv2.imshow("Colour Image", colour_image)
            cv2.imshow("Depth Image", depth_image / 2)

            from scipy import ndimage
            gradient_image: np.ndarray = ndimage.gaussian_gradient_magnitude(depth_image, sigma=3)
            cv2.imshow("Gradient Image", gradient_image)
            # print(np.min(gradient_image), np.mean(gradient_image), np.max(gradient_image))
            gradient_mask: np.ndarray = np.where(gradient_image >= 0.1, 255, 0).astype(np.uint8)
            cv2.imshow("Gradient Mask", gradient_mask)

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

        ws_points: np.ndarray = GeometryUtil.compute_world_points_image_fast(depth_image, np.eye(4), intrinsics)

        # depth_image = np.where(gradient_mask != 0, depth_image, 0.0).astype(float)

        instances: List[InstanceSegmenter.Instance] = segmenter.parse_raw_instances(raw_instances)
        for instance in instances:
            mins, maxs = make_instance_bounds(instance, depth_image, ws_points)
            box: o3d.geometry.AxisAlignedBoundingBox = o3d.geometry.AxisAlignedBoundingBox(mins, maxs)
            box.color = (1.0, 0.0, 1.0)
            to_visualise.append(box)

        VisualisationUtil.visualise_geometries(to_visualise)


if __name__ == "__main__":
    main()
