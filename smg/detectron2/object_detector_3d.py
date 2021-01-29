from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from sklearn.mixture import GaussianMixture
from typing import List, Optional, Tuple

from smg.utility import GeometryUtil

from .instance_segmenter import InstanceSegmenter


class ObjectDetector3D:
    """TODO"""

    # NESTED TYPES

    class Object:
        """A detected object."""

        # CONSTRUCTOR

        def __init__(self, instance: InstanceSegmenter.Instance, pred_box_3d: Tuple[np.ndarray, np.ndarray]):
            self.__instance: InstanceSegmenter.Instance = instance
            self.__pred_box_3d: Tuple[np.ndarray, np.ndarray] = pred_box_3d

        # PROPERTIES

        @property
        def pred_box_2d(self) -> Tuple[float, float, float, float]:
            return self.__instance.pred_box

        @property
        def pred_box_3d(self) -> Tuple[np.ndarray, np.ndarray]:
            return self.__pred_box_3d

        @property
        def pred_class(self) -> str:
            return self.__instance.pred_class

        @property
        def pred_mask(self) -> np.ndarray:
            return self.__instance.pred_mask

        @property
        def score(self) -> float:
            return self.__instance.score

    # CONSTRUCTOR

    def __init__(self, segmenter: InstanceSegmenter):
        self.__segmenter: InstanceSegmenter = segmenter

    # PUBLIC METHODS

    def detect_objects(self, colour_image: np.ndarray, depth_image: np.ndarray, pose: np.ndarray,
                       intrinsics: Tuple[float, float, float, float]) -> List[Object]:
        return self.lift_instances_to_objects(self.__segmenter.segment(colour_image), depth_image, pose, intrinsics)

    def lift_instances_to_objects(self, instances: List[InstanceSegmenter.Instance], depth_image: np.ndarray,
                                  pose: np.ndarray, intrinsics: Tuple[float, float, float, float]) -> List[Object]:
        objects: List[ObjectDetector3D.Object] = []
        ws_points: np.ndarray = GeometryUtil.compute_world_points_image_fast(depth_image, pose, intrinsics)

        for instance in instances:
            pred_box_3d: Optional[Tuple[np.ndarray, np.ndarray]] = self.__predict_box_3d(
                instance, depth_image, ws_points
            )

            if pred_box_3d is not None:
                objects.append(ObjectDetector3D.Object(instance, pred_box_3d))

        return objects

    # PRIVATE STATIC METHODS

    @staticmethod
    def __predict_box_3d(instance: InstanceSegmenter.Instance, depth_image: np.ndarray, ws_points: np.ndarray) \
            -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            combined_mask: np.ndarray = np.where(
                (instance.pred_mask != 0) & (depth_image != 0), 255, 0
            ).astype(np.uint8)
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
            combined_mask = np.where(
                (depth_image >= min_depth) & (depth_image <= max_depth), combined_mask, 0
            ).astype(np.uint8)
            print(f"{instance.pred_class}: {min_depth}, {max_depth}, {np.count_nonzero(combined_mask)}")

            print(f"{instance.pred_class}: {list(depths)}")
            plt.hist(depths, 50)
            plt.title(f"{instance.pred_class}")
            plt.xlabel("Depth")
            plt.ylabel("Count")
            plt.grid(True)
            plt.show()

            masked_ws_points: np.ndarray = np.where(np.atleast_3d(combined_mask), ws_points,
                                                    np.full(ws_points.shape, np.nan))
            mins: np.ndarray = np.array([
                np.nanmin(masked_ws_points[:, :, 0]),
                np.nanmin(masked_ws_points[:, :, 1]),
                np.nanmin(masked_ws_points[:, :, 2]),
            ])
            maxs: np.ndarray = np.array([
                np.nanmax(masked_ws_points[:, :, 0]),
                np.nanmax(masked_ws_points[:, :, 1]),
                np.nanmax(masked_ws_points[:, :, 2]),
            ])
            centre: np.ndarray = (mins + maxs) / 2
            scale: float = 1.25
            mins = centre + scale * (mins - centre)
            maxs = centre + scale * (maxs - centre)
            return mins, maxs
        except ValueError:
            return None
