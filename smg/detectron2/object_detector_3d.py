from __future__ import annotations

import numpy as np

from typing import List, Optional, Tuple

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

    def detect_objects(self, colour_image: np.ndarray, depth_image: np.ndarray,
                       intrinsics: Tuple[float, float, float, float]) -> List[Object]:
        pass

    # PRIVATE STATIC METHODS

    @staticmethod
    def __make_box_3d(instance: InstanceSegmenter.Instance, depth_image: np.ndarray, ws_points: np.ndarray) \
            -> Optional[Tuple[np.ndarray, np.ndarray]]:
        pass
