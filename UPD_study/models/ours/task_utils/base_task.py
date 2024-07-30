from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
from skimage import measure

from UPD_study.models.ours.task_utils.labelling import AnomalyLabeller
from UPD_study.models.ours.task_utils.task_shape import EitherDeformedHypershapePatchMaker
from UPD_study.models.ours.task_utils.utils import get_patch_slices, nsa_sample_dimension


class BaseTask(ABC):

    def __init__(self, sample_labeller: Optional[AnomalyLabeller] = None, **all_kwargs):
        self.sample_labeller = sample_labeller
        self.rng = np.random.default_rng()
        self.anomaly_shape_maker = EitherDeformedHypershapePatchMaker(nsa_sample_dimension)
        self.all_kwargs = all_kwargs

    def apply(self, sample: np.ndarray[float],
              sample_mask: Optional[np.ndarray[np.bool_]],
              target_area_slice: Optional[tuple[slice, ...]],
              *args, **kwargs)\
            -> tuple[np.ndarray[float], np.ndarray[float]]:
        """
        Apply the self-supervised task to the single data sample.
        :param sample: Normal sample to be augmented
        :param sample_mask: Object mask of sample.
        :return: sample with task applied and label map.
        """

        aug_sample = sample.copy()
        sample_shape = np.array(sample.shape[1:])
        anomaly_mask = np.zeros(sample_shape, dtype=bool)

        min_anom_prop = self.all_kwargs.get('min_anom_prop', 0.06)
        max_anom_prop = self.all_kwargs.get('max_anom_prop', 0.8)

        mask_regions = measure.regionprops(measure.label(sample_mask)) if sample_mask is not None else None
        if sample_mask is not None:
            # Find biggest region
            biggest_region = max(mask_regions, key=lambda r: r.area_bbox)
            sample_size = np.array([s.stop - s.start for s in biggest_region.slice])
        else:
            sample_size = sample_shape

        min_dim_lens = np.ceil(min_anom_prop * sample_size).astype(int)
        max_dim_lens = np.ceil(max_anom_prop * sample_size).astype(int)

        max_dim_lens = np.maximum(max_dim_lens, min_dim_lens + 1)

        dim_bounds = list(zip(min_dim_lens, max_dim_lens))

        spatial_dims = tuple(range(1, len(sample_shape)))
        channel_min_vals = np.min(sample, axis=spatial_dims, keepdims=True)
        channel_max_vals = np.max(sample, axis=spatial_dims, keepdims=True)

        # For random number of times
        for _ in range(4):

            # Compute anomaly mask
            curr_anomaly_mask, intersect_fn = self.anomaly_shape_maker.get_patch_mask_and_intersect_fn(dim_bounds,
                                                                                                       sample_shape)
            # Choose anomaly location
            anomaly_corner = self.find_valid_anomaly_location(curr_anomaly_mask, sample_mask, mask_regions,
                                                              sample_shape, target_area_slice)

            if sample_mask is not None and self.all_kwargs.get('fg_only', False):
                sample_mask_patch = sample_mask[get_patch_slices(anomaly_corner, curr_anomaly_mask.shape)]
                curr_anomaly_mask = np.logical_and(curr_anomaly_mask, sample_mask_patch)

            # Apply self-supervised task
            aug_sample = self.augment_sample(aug_sample, sample_mask, anomaly_corner, curr_anomaly_mask, intersect_fn)
            aug_sample = np.clip(aug_sample, channel_min_vals, channel_max_vals)

            anomaly_mask[get_patch_slices(anomaly_corner, curr_anomaly_mask.shape)] |= curr_anomaly_mask

            # Randomly brake at end of loop, ensuring we get at least 1 anomaly
            if self.rng.random() > 0.5:
                break

        if self.sample_labeller is not None:
            return aug_sample, self.sample_labeller(aug_sample, sample, anomaly_mask)
        else:
            # If no labeller is provided, we are probably in a calibration process
            return aug_sample, np.expand_dims(anomaly_mask, 0)

    def find_valid_anomaly_location(self, curr_anomaly_mask: np.ndarray[np.bool_],
                                    sample_mask: Optional[np.ndarray[np.bool_]],
                                    mask_regions: Optional[list],
                                    sample_shape: np.ndarray[np.int_],
                                    target_area_slice: tuple[slice, ...]):
        # mask_regions is a list of RegionProperties objects
        # type is not exposed :((

        curr_anomaly_shape = np.array(curr_anomaly_mask.shape)

        if target_area_slice is None:
            slice_start = np.zeros(len(sample_shape))
            slice_stop = sample_shape
        else:
            slice_start = np.array([s.start for s in target_area_slice])
            slice_stop = np.array([s.stop for s in target_area_slice])

            if sample_mask is not None:
                # Filter out any regions which are not in the target area
                mask_regions = [r for r in mask_regions
                                if all(t_s.start < r_s.stop and r_s.start < t_s.stop
                                       for t_s, r_s in zip(target_area_slice, r.slice))]

        min_corner = np.maximum(np.zeros(len(sample_shape)), slice_start - curr_anomaly_shape + 1)
        max_corner = np.minimum(sample_shape - curr_anomaly_shape, slice_stop - 1)
        # - Apply anomaly at location

        min_overlap_prop = self.all_kwargs.get('min_anom_prop', 0.06) / 3
        if sample_mask is None:
            anom_min_visible = np.prod((min_overlap_prop) * sample_shape)
        else:
            anom_min_visible = sample_mask.sum() * ((min_overlap_prop) ** len(sample_shape))

        curr_anomaly_area = curr_anomaly_mask.sum()
        assert curr_anomaly_area >= anom_min_visible, \
            f'Anomaly mask should be at least {anom_min_visible} voxels, but is {curr_anomaly_area}'

        if mask_regions is not None:
            bbox_area_sum = sum(r.area_bbox for r in mask_regions) if mask_regions is not None else None
            bbox_area_norm = [r.area_bbox / bbox_area_sum for r in mask_regions]
        else:
            bbox_area_norm = None
        i = 0
        target_mask_overlap_prop = 0.5
        while True:

            if i % 1000 == 0:
                if i == 5000:
                    raise RuntimeError(f'Could not find valid anomaly location, current target')
                target_mask_overlap_prop /= 2

            if mask_regions is not None:
                # Randomly choose a region, weighted by its area
                random_region = self.rng.choice(mask_regions, p=bbox_area_norm)
                # Choose a random corner within the region
                curr_min_corner = np.maximum(min_corner,
                                             np.array([s.start for s in random_region.slice]) - curr_anomaly_shape + 1) 
                curr_max_corner = np.minimum(max_corner,
                                             np.array([s.stop for s in random_region.slice]) - 1)
            else:
                curr_min_corner = min_corner
                curr_max_corner = max_corner

            anomaly_corner = self.rng.integers(curr_min_corner, curr_max_corner, endpoint=True)

            if target_area_slice is not None:
                target_area_slice_rel_anom = tuple([slice(max(s.start - a, 0), min(s.stop - a, m))
                                                    for s, a, m in zip(target_area_slice, anomaly_corner, curr_anomaly_shape)])
                assert all(s.start < s.stop for s in target_area_slice_rel_anom), \
                    'Anomaly should always have SOME overlap with target area:' +\
                    f'\n target_area_slice{target_area_slice},\n anom corner{anomaly_corner},\n' +\
                    f' anom shape{curr_anomaly_shape},\n target relativae slice{target_area_slice_rel_anom}\n' +\
                    f' min_corner{curr_min_corner}, \nmax_corner{curr_max_corner}'

                anom_overlap = np.sum(curr_anomaly_mask[target_area_slice_rel_anom])

                if anom_overlap < anom_min_visible:
                    # Visible anomaly must be at least 1/3 of the smallest possible anomaly
                    # Otherwise try again
                    continue

            # If the sample mask is None, any location within the bounds is valid
            if sample_mask is None:
                break

            # Otherwise, we need to check that the intersection of the anomaly mask and the sample mask is at least 50%
            target_patch_obj_mask = sample_mask[get_patch_slices(anomaly_corner, curr_anomaly_mask.shape)]
            overlap_size = np.sum(target_patch_obj_mask & curr_anomaly_mask)
            if (overlap_size / curr_anomaly_area) >= target_mask_overlap_prop:
                break

            i += 1
        return anomaly_corner

    def __call__(self, sample: np.ndarray[float],
                 sample_mask: Optional[np.ndarray[np.bool_]],
                 target_area_slice: Optional[tuple[slice, ...]],
                 *args, **kwargs)\
            -> tuple[np.ndarray[float], np.ndarray[float]]:
        """
        Apply the self-supervised task to the single data sample.
        :param sample: Normal sample to be augmented
        :param sample_mask: Object mask of sample.
        :param **kwargs:
            * *sample_path*: Path to source image
        :return: sample with task applied and label map.
        """
        return self.apply(sample, sample_mask, target_area_slice, *args, **kwargs)

    @abstractmethod
    def augment_sample(self,
                       sample: np.ndarray[float],
                       sample_mask: Optional[np.ndarray[np.bool_]],
                       anomaly_corner: np.ndarray[np.int_],
                       anomaly_mask: np.ndarray[np.bool_],
                       anomaly_intersect_fn: Callable[[np.ndarray[float], np.ndarray[float]], np.ndarray[float]]) \
            -> np.ndarray[float]:
        """
        Apply self-supervised task to region at anomaly_corner covered by anomaly_mask
        :param sample: Sample to be augmented.
        :param sample_mask: Object mask of sample.
        :param anomaly_corner: Index of anomaly corner.
        :param anomaly_mask: Mask
        :param anomaly_intersect_fn: Function which, given a line's origin and direction, finds its intersection with
        the edge of the anomaly mask
        :return:
        """
