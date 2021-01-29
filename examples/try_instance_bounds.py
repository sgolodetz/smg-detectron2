import cv2
import numpy as np
import open3d as o3d
import sys

from detectron2.structures import Instances
from typing import List, Tuple

from smg.detectron2 import InstanceSegmenter, ObjectDetector3D
from smg.open3d import VisualisationUtil
from smg.openni import OpenNICamera


def main() -> None:
    np.set_printoptions(threshold=sys.maxsize)

    segmenter: InstanceSegmenter = InstanceSegmenter.make_mask_rcnn()
    detector: ObjectDetector3D = ObjectDetector3D(segmenter)

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
            colour_image, depth_image, intrinsics
        )
        to_visualise: List[o3d.geometry.Geometry] = [pcd]

        objects: List[ObjectDetector3D.Object] = detector.lift_instances_to_objects(
            segmenter.parse_raw_instances(raw_instances), depth_image, np.eye(4), intrinsics
        )
        for obj in objects:
            box: o3d.geometry.AxisAlignedBoundingBox = o3d.geometry.AxisAlignedBoundingBox(*obj.pred_box_3d)
            box.color = (1.0, 0.0, 1.0)
            to_visualise.append(box)

        VisualisationUtil.visualise_geometries(to_visualise)


if __name__ == "__main__":
    main()
