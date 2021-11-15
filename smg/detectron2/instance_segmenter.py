from __future__ import annotations

import logging
import numpy as np
import torch

# Suppress some annoying internal warnings produced by Detectron2 - there's nothing much we can do about them.
logging.captureWarnings(True)
logging.getLogger("py.warnings").setLevel(logging.CRITICAL)
logging.getLogger("fvcore.common.file_io").setLevel(logging.CRITICAL)

from typing import Any, Dict, List, Tuple

from detectron2 import model_zoo
from detectron2.config import CfgNode, get_cfg
from detectron2.data import Metadata, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer


class InstanceSegmenter:
    """An instance segmenter based on one of the Detectron2 models."""

    # NESTED TYPES

    class Instance:
        """A segmented instance."""

        # CONSTRUCTOR

        def __init__(self, box: Tuple[float, float, float, float], label: str, mask: np.ndarray, score: float):
            """
            Construct a segmented instance.

            :param box:    The bounding box predicted for the instance.
            :param label:  The class label predicted for the instance.
            :param mask:   The binary mask predicted for the instance.
            :param score:  The score predicted for the instance (a float in [0,1]).
            """
            self.__box: Tuple[float, float, float, float] = box
            self.__label: str = label
            self.__mask: np.ndarray = mask
            self.__score: float = score

        # PROPERTIES

        @property
        def box(self) -> Tuple[float, float, float, float]:
            """
            Get the bounding box predicted for the instance.

            :return:    The bounding box predicted for the instance.
            """
            return self.__box

        @property
        def label(self) -> str:
            """
            Get the class label predicted for the instance.

            :return:    The class label predicted for the instance.
            """
            return self.__label

        @property
        def mask(self) -> np.ndarray:
            """
            Get the binary mask predicted for the instance.

            :return:    The binary mask predicted for the instance.
            """
            return self.__mask

        @property
        def score(self) -> float:
            """
            Get the score predicted for the instance.

            :return:    The score predicted for the instance (a float in [0,1]).
            """
            return self.__score

    # CONSTRUCTOR

    def __init__(self, cfg: CfgNode):
        """
        Construct an instance segmenter based on one of the Detectron2 models.

        :param cfg: The model configuration.
        """
        self.__cfg: CfgNode = cfg
        self.__stream: torch.cuda.Stream = torch.cuda.Stream()

        with torch.cuda.stream(self.__stream):
            self.__predictor: DefaultPredictor = DefaultPredictor(cfg)

    # PUBLIC STATIC METHODS

    @staticmethod
    def make_mask_rcnn(*, config_path: str = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
                       score_threshold: float = 0.75) -> InstanceSegmenter:
        """
        Make an instance segmenter that uses a Mask R-CNN model.

        :param config_path:     The path to the model configuration file.
        :param score_threshold: The minimum detection score needed to detect an instance.
        :return:                The instance segmenter.
        """
        cfg: CfgNode = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config_path))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)
        return InstanceSegmenter(cfg)

    # PUBLIC METHODS

    def draw_raw_instances(self, raw_instances: Instances, image: np.ndarray) -> np.ndarray:
        """
        Visualise the raw instances output by Detectron2 for an image.

        :param image:           The image that was segmented by Detectron2.
        :param raw_instances:   The raw instances output by Detectron2 for that image.
        :return:                A visualisation of the raw instances output by Detectron2 for the image.
        """
        v: Visualizer = Visualizer(image[:, :, ::-1], self.__get_metadata())
        out = v.draw_instance_predictions(raw_instances.to("cpu"))
        return out.get_image()[:, :, ::-1]

    def get_config(self) -> CfgNode:
        """
        Get the model configuration.

        :return:    The model configuration.
        """
        return self.__cfg

    def parse_raw_instances(self, raw_instances: Instances) -> List[Instance]:
        """
        Parse a set of raw instances output by Detectron2 to make them easier to work with.

        :param raw_instances:   The raw instances output by Detectron2.
        :return:                A parsed version of the raw instances.
        """
        instances: List[InstanceSegmenter.Instance] = []

        class_names: List[str] = self.__get_metadata().get("thing_classes", None)

        with torch.cuda.stream(self.__stream):
            for i in range(len(raw_instances)):
                fields: Dict[str, Any] = raw_instances[i].get_fields()
                box: Tuple[float, float, float, float] = tuple(*fields["pred_boxes"].tensor.cpu().detach().numpy())
                label: str = class_names[fields["pred_classes"].cpu().detach().numpy()[0]]
                mask: np.ndarray = fields["pred_masks"].cpu().detach().numpy().squeeze()
                mask = np.where(mask, 255, 0).astype(np.uint8)
                score: float = fields["scores"].cpu().detach().numpy()[0]
                instances.append(InstanceSegmenter.Instance(box, label, mask, score))

        return instances

    def segment(self, image: np.ndarray) -> List[Instance]:
        """
        Segment the specified image, returning a parsed version of the raw instances output by Detectron2.

        :param image:   The image to segment.
        :return:        A parsed version of the raw instances output by Detectron2.
        """
        return self.parse_raw_instances(self.segment_raw(image))

    def segment_raw(self, image: np.ndarray) -> Instances:
        """
        Segment the specified image, returning the raw instances output by Detectron2.

        :param image:   The image to segment.
        :return:        The raw instances output by Detectron2.
        """
        with torch.cuda.stream(self.__stream):
            raw_instances: Instances = self.__predictor(image)["instances"]

        return raw_instances

    # PRIVATE METHODS

    def __get_metadata(self) -> Metadata:
        """
        Get the metadata for the dataset.

        .. note::
            This can be used to e.g. get the labels corresponding to the class IDs.

        :return:    The metadata for the dataset.
        """
        return MetadataCatalog.get(self.__cfg.DATASETS.TRAIN[0])
