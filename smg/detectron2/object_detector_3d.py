from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np

from sklearn.mixture import GaussianMixture
from typing import List, Optional, Tuple

from smg.utility import GeometryUtil

from .instance_segmenter import InstanceSegmenter


class ObjectDetector3D:
    """A 3D object detector based on back-projecting the instances detected by one of the Detectron2 models."""

    # NESTED TYPES

    class Object3D:
        """A detected 3D object."""

        # CONSTRUCTOR

        def __init__(self, instance: InstanceSegmenter.Instance, box_3d: Tuple[np.ndarray, np.ndarray]):
            """
            Construct a detected object.

            :param instance:    The 2D instance on which the object is based.
            :param box_3d:      The 3D bounding box predicted for the object.
            """
            self.__box_3d: Tuple[np.ndarray, np.ndarray] = box_3d
            self.__instance: InstanceSegmenter.Instance = instance

        # PROPERTIES

        @property
        def box_2d(self) -> Tuple[float, float, float, float]:
            """
            Get the 2D bounding box predicted for the object.

            :return:    The 2D bounding box predicted for the object.
            """
            return self.__instance.box

        @property
        def box_3d(self) -> Tuple[np.ndarray, np.ndarray]:
            """
            Get the axis-aligned 3D bounding box predicted for the object.

            :return:    The axis-aligned 3D bounding box predicted for the object, as a (mins, maxs) tuple.
            """
            return self.__box_3d

        @property
        def label(self) -> str:
            """
            Get the class label predicted for the object.

            :return:    The class label predicted for the object.
            """
            return self.__instance.label

        @property
        def mask(self) -> np.ndarray:
            """
            Get the binary mask predicted for the object.

            :return:    The binary mask predicted for the object.
            """
            return self.__instance.mask

        @property
        def score(self) -> float:
            """
            Get the score predicted for the object.

            :return:    The score predicted for the object (a float in [0,1]).
            """
            return self.__instance.score

    # CONSTRUCTOR

    def __init__(self, segmenter: InstanceSegmenter, *, debug: bool = False):
        """
        Construct a 3D object detector based on back-projecting the instances detected by an instance segmenter.

        :param segmenter:   The instance segmenter.
        :param debug:       Whether to show debug visualisations.
        """
        self.__segmenter: InstanceSegmenter = segmenter
        self.__debug: bool = debug

    # PUBLIC METHODS

    def detect_objects(self, colour_image: np.ndarray, depth_image: np.ndarray, pose: np.ndarray,
                       intrinsics: Tuple[float, float, float, float]) -> List[Object3D]:
        """
        Try to detect the 3D objects present in an RGB-D image.

        .. note::
            We assume that the colour and depth images {have been/are naturally} aligned, and that the two cameras
            have the same intrinsics. This is a reasonable assumption when e.g. (i) using a Kinect with auto-align
            turned on, or (ii) predicting a depth image to augment a colour image.

        :param colour_image:    The colour part of the RGB-D image.
        :param depth_image:     The depth part of the RGB-D image.
        :param pose:            The camera pose (we assume that the colour and depth images have been aligned).
        :param intrinsics:      The camera intrinsics, as an (fx, fy, cx, cy) tuple.
        :return:                The 3D objects detected in the RGB-D image.
        """
        return self.lift_to_3d(self.__segmenter.segment(colour_image), depth_image, pose, intrinsics)

    def lift_to_3d(self, instances: List[InstanceSegmenter.Instance], depth_image: np.ndarray,
                   pose: np.ndarray, intrinsics: Tuple[float, float, float, float]) -> List[Object3D]:
        """
        Lift a set of 2D instances to 3D objects by generating 3D bounding boxes for them.

        :param instances:   The 2D instances.
        :param depth_image: The depth image.
        :param pose:        The camera pose.
        :param intrinsics:  The camera intrinsics, as an (fx, fy, cx, cy) tuple.
        :return:            The 3D objects.
        """
        objects: List[ObjectDetector3D.Object3D] = []
        ws_points: np.ndarray = GeometryUtil.compute_world_points_image_fast(depth_image, pose, intrinsics)

        for instance in instances:
            box_3d: Optional[Tuple[np.ndarray, np.ndarray]] = self.__predict_box_3d(instance, depth_image, ws_points)
            if box_3d is not None:
                objects.append(ObjectDetector3D.Object3D(instance, box_3d))

        return objects

    # PRIVATE METHODS

    def __predict_box_3d(self, instance: InstanceSegmenter.Instance, depth_image: np.ndarray,
                         ws_points: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Try to predict a 3D bounding box for an existing 2D instance.

        .. note::
            The world-space points image is the result of back-projecting the depth image and transforming
            it by the pose, so it may seem surprising that we pass in both here. However, in practice, the
            depth image is needed to mask out pixels with invalid depth (this information cannot be obtained
            from the world-space points image), and whilst we could in principle recompute the world-space
            points imgae from the depth image for each object, that would be inefficient.
        .. note::
            It's possible for this to fail if there aren't enough pixels in the 2D instance mask that have
            a valid depth. In that case, None will be returned.

        :param instance:    The 2D instance.
        :param depth_image: The depth image.
        :param ws_points:   The world-space points image.
        :return:            The 3D bounding box for the instance, if possible, or None otherwise.
        """
        try:
            # The pixels we can use to predict the 3D bounding box for this object are those that are within the
            # 2D instance mask and also have valid depth.
            usable_mask: np.ndarray = np.where(
                (instance.mask != 0) & (depth_image != 0), 255, 0
            ).astype(np.uint8)

            # In practice, the 2D instance mask may be a bit inaccurate around the edges of the object, so we erode
            # it slightly to help mitigate this a bit.
            usable_mask = cv2.erode(usable_mask, (5, 5))

            # If we're debugging, show the usable pixels mask.
            if self.__debug:
                cv2.imshow("Usable Pixels Mask", usable_mask)
                cv2.waitKey(1)

            # Make an array consisting of the depths of all the usable pixels.
            depths: np.ndarray = depth_image[np.where(usable_mask)]

            # Fit a Gaussian to the depth distribution.
            reshaped_depths: np.ndarray = depths.reshape(-1, 1)
            gm: GaussianMixture = GaussianMixture().fit(reshaped_depths)

            # Compute a probability that each depth belongs to this distribution.
            scores: np.ndarray = gm.score_samples(reshaped_depths)
            probs: np.ndarray = np.exp(scores)
            probs = (probs - np.min(probs)) / (np.max(probs) - np.min(probs))

            # Filter out any depths whose probability of belonging to the distribution is below a threshold.
            depths = depths[np.where(probs >= 0.5)]
            min_depth: float = np.min(depths)
            max_depth: float = np.max(depths)
            usable_mask = np.where(
                (depth_image >= min_depth) & (depth_image <= max_depth), usable_mask, 0
            ).astype(np.uint8)

            # If we're debugging, print out some useful information and show the depth distribution.
            if self.__debug:
                print(f"{instance.label}:")
                print(f"  * Depth Range: [{min_depth}, {max_depth}], Usable Pixels: {np.count_nonzero(usable_mask)}")
                print(f"  * Depths: {list(depths)}")

                plt.hist(depths)
                plt.title(f"{instance.label}")
                plt.xlabel("Depth")
                plt.ylabel("Count")
                plt.grid(True)
                plt.show()

            # Compute the minimum and maximum bounds of the usable world-space points.
            usable_ws_points: np.ndarray = np.where(
                np.atleast_3d(usable_mask), ws_points, np.full(ws_points.shape, np.nan)
            )
            mins: np.ndarray = np.array([np.nanmin(usable_ws_points[:, :, i]) for i in range(3)])
            maxs: np.ndarray = np.array([np.nanmax(usable_ws_points[:, :, i]) for i in range(3)])

            # Scale the bounding box up slightly relative to its centre (in practice, we over-prune the depths above
            # to avoid accidentally including outliers, and this can be used to compensate for that slightly).
            centre: np.ndarray = (mins + maxs) / 2
            scale: float = 1.25
            mins = centre + scale * (mins - centre)
            maxs = centre + scale * (maxs - centre)

            return mins, maxs
        except ValueError:
            return None
