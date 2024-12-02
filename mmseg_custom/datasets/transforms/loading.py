from mmseg.registry import TRANSFORMS
from mmseg.datasets.transforms.loading import LoadAnnotations
import mmengine.fileio as fileio
import mmcv
import numpy as np

@TRANSFORMS.register_module()
class LoadAnnotationsCL(LoadAnnotations):
    """Load annotations for classification.
    Args:
        reduce_zero_label (bool, optional): Whether to reduce the pixel value
            of 'background' to 0. Default: True.
    """

    def __init__(self, class_range=None, anno_check=False, **kwargs):
        super().__init__(**kwargs)
        self.class_range = class_range
        self.anno_check = anno_check

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        if self.class_range is not None:
            gt_semantic_seg = gt_semantic_seg.astype(np.int16)
            if self.reduce_zero_label:
                # previous
                # class_range = [item - self.class_range[0] for item in self.class_range]
                # gt_semantic_seg = gt_semantic_seg - (self.class_range[0] - 1)
                # gt_semantic_seg[gt_semantic_seg < class_range[0]] = 255
                # gt_semantic_seg[gt_semantic_seg > class_range[1]] = 255
                
                # new
                class_range = [item - 1 for item in self.class_range]
                gt_semantic_seg[gt_semantic_seg < class_range[0]] = 255
                gt_semantic_seg[gt_semantic_seg > class_range[1]] = 255
            else:
                class_range = [item + 1 - self.class_range[0] for item in self.class_range]
                gt_semantic_seg = gt_semantic_seg - (self.class_range[0] - 1)
                gt_semantic_seg[gt_semantic_seg < class_range[0]] = 0
                gt_semantic_seg[gt_semantic_seg > class_range[1]] = 0

        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id

        # elimate the gt_semantic_seg which has no annotation
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')


    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation and keypoints annotations.
        """

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_seg:
            self._load_seg_map(results)
        if self.with_keypoints:
            self._load_kps(results)
        if self.anno_check:
            check_valid = np.unique(results['gt_seg_map'])
            if len(check_valid) == 1 and check_valid[0] == 255:
                return None
        return results