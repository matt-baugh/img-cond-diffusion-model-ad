from abc import abstractmethod
from typing import Callable, Optional, Union

import numpy as np
from numpy.linalg import norm
from scipy import ndimage

from UPD_study.models.ours.task_utils.base_task import BaseTask
from UPD_study.models.ours.task_utils.labelling import AnomalyLabeller
from UPD_study.models.ours.task_utils.utils import get_patch_slices


class BaseDeformationTask(BaseTask):

    @abstractmethod
    def compute_mapping(self, sample: np.ndarray[float], sample_mask: Optional[np.ndarray[bool]],
                        anomaly_corner: np.ndarray[int], anomaly_mask: np.ndarray[bool],
                        anomaly_intersect_fn: Callable[[np.ndarray[float], np.ndarray[float]], np.ndarray[float]]) \
            -> np.ndarray[float]:
        """
        Returns array of size (*anomaly_mask.shape, len(anomaly_mask.shape)).
        Probably don't need entire sample, but including in for generality.
        :param sample:
        :param sample_mask:
        :param anomaly_corner:
        :param anomaly_mask:
        :param anomaly_intersect_fn:
        :return:
        """

    def augment_sample(self, sample: np.ndarray[float], sample_mask: Optional[np.ndarray[bool]],
                       anomaly_corner: np.ndarray[int], anomaly_mask: np.ndarray[bool],
                       anomaly_intersect_fn: Callable[[np.ndarray[float], np.ndarray[float]], np.ndarray[float]]) \
            -> np.ndarray[float]:
        num_channels = sample.shape[0]
        mapping = self.compute_mapping(sample, sample_mask, anomaly_corner, anomaly_mask, anomaly_intersect_fn)
        sample_slices = get_patch_slices(anomaly_corner, tuple(anomaly_mask.shape))

        for chan in range(num_channels):
            sample[chan][sample_slices] = ndimage.map_coordinates(sample[chan][sample_slices],
                                                                  mapping,
                                                                  mode='nearest')

        return sample


class RadialDeformationTask(BaseDeformationTask):

    def __init__(self, sample_labeller: Optional[AnomalyLabeller], deform_factor: Optional[float] = None,
                 deform_centre: Optional[np.ndarray] = None, **kwargs):
        super().__init__(sample_labeller, **kwargs)
        self.deform_factor = deform_factor
        self.deform_centre = deform_centre

    def get_deform_factor(self, def_centre: np.ndarray[int], anomaly_mask: np.ndarray[bool]):
        return self.deform_factor if self.deform_factor is not None else 2 ** self.rng.uniform(0.5, 2)

    @abstractmethod
    def compute_new_distance(self, curr_distance: float, max_distance: float, factor: float) -> float:
        """
        Compute new distance for point to be sampled from
        :param curr_distance:
        :param max_distance:
        :param factor:
        """

    def compute_mapping(self, sample: np.ndarray[float], sample_mask: Optional[np.ndarray[bool]],
                        anomaly_corner: np.ndarray[int], anomaly_mask: np.ndarray[bool],
                        anomaly_intersect_fn: Callable[[np.ndarray[float], np.ndarray[float]], np.ndarray[float]]) \
            -> np.ndarray[float]:
        # NOTE: This assumes that the shape is convex, will make discontinuities if it's not.

        anomaly_shape = np.array(anomaly_mask.shape)
        num_dims = len(anomaly_shape)

        # Expand so can later be broadcast with (D, N)
        mask_centre = (anomaly_shape - 1) / 2
        exp_mask_centre = np.reshape(mask_centre, (-1, 1))
        # Shape (D, N)
        poss_centre_coords = np.stack(np.nonzero(anomaly_mask))
        def_centre = self.deform_centre if self.deform_centre is not None else \
            poss_centre_coords[:, np.random.randint(poss_centre_coords.shape[1])]

        assert anomaly_mask[tuple(def_centre.round().astype(int))], f'Centre is not within anomaly: {def_centre}'

        # exp_ = expanded
        exp_def_centre = np.reshape(def_centre, (-1, 1))

        # (D, *anomaly_shape)
        mapping = np.stack(np.meshgrid(*[np.arange(s, dtype=float) for s in anomaly_shape], indexing='ij'), axis=0)

        # Ignore pixels on edge of bounding box
        mask_inner_slice = tuple([slice(1, -1)] * num_dims)
        map_inner_slice = tuple([slice(None)] + list(mask_inner_slice))
        # Get all coords and transpose so coord index is last dimension (D, N)
        anomaly_coords = mapping[map_inner_slice][(slice(None), anomaly_mask[mask_inner_slice])]

        all_coords_to_centre = anomaly_coords - exp_def_centre
        all_coords_distance = norm(all_coords_to_centre, axis=0)
        # Ignore zero divided by zero, as we correct it before mapping is returned
        with np.errstate(invalid='ignore'):
            all_coords_norm_dirs = all_coords_to_centre / all_coords_distance
        mask_edge_intersections = anomaly_intersect_fn(exp_def_centre - exp_mask_centre, all_coords_norm_dirs) + \
            exp_mask_centre
        mask_edge_distances = norm(mask_edge_intersections - exp_def_centre, axis=0)

        # Get factor once, so is same for all pixels
        def_factor = self.get_deform_factor(def_centre, anomaly_mask)
        new_coord_distances = self.compute_new_distance(all_coords_distance, mask_edge_distances, def_factor)
        # (D, N)
        new_coords = exp_def_centre + new_coord_distances * all_coords_norm_dirs

        mapping[map_inner_slice][(slice(None), anomaly_mask[mask_inner_slice])] = new_coords

        # Revert centre coordinate, as it will be nan due to the zero magnitude direction vector
        mapping[(slice(None), *def_centre)] = def_centre

        return mapping


class SourceDeformationTask(RadialDeformationTask):

    def compute_new_distance(self, curr_distance: Union[float, np.ndarray[float]],
                             max_distance: Union[float, np.ndarray[float]],
                             factor: Union[float, np.ndarray[float]]) -> Union[float, np.ndarray[float]]:
        # y = x^3 (between 0 and 1)
        # -> y = max_d * (curr / max) ^ factor
        # -> y = curr ^ factor / max_d ^ (factor - 1)   to avoid FP errors
        return curr_distance ** factor / max_distance ** (factor - 1)


class SinkDeformationTask(RadialDeformationTask):
    # y = 1 - (1 - x)^3 (between 0 and 1)
    # -> y = max_d (1 - (1 - curr / max_d) ^ factor)
    # -> y = max_d - (max_d - curr) ^ factor / max_d ^ (factor - 1)
    def compute_new_distance(self, curr_distance: Union[float, np.ndarray[float]],
                             max_distance: Union[float, np.ndarray[float]],
                             factor: Union[float, np.ndarray[float]]) -> Union[float, np.ndarray[float]]:
        return max_distance - (max_distance - curr_distance) ** factor / max_distance ** (factor - 1)


class FPISinkDeformationTask(RadialDeformationTask):
    def compute_new_distance(self, curr_distance: Union[float, np.ndarray[float]],
                             max_distance: Union[float, np.ndarray[float]],
                             factor: Union[float, np.ndarray[float]]) -> Union[float, np.ndarray[float]]:
        prop_distance = curr_distance / max_distance
        return max_distance * (prop_distance + (1 - prop_distance ** 2) * prop_distance)


class IdentityDeformationTask(RadialDeformationTask):
    def compute_new_distance(self, curr_distance: Union[float, np.ndarray[float]],
                             max_distance: Union[float, np.ndarray[float]],
                             factor: Union[float, np.ndarray[float]]) -> Union[float, np.ndarray[float]]:
        return curr_distance


def calc_perpendicular_deformation_dist(def_centre: np.ndarray[int], anomaly_mask: np.ndarray[bool],
                                        **kwargs) -> (float, float):
    inner_edge = np.logical_xor(anomaly_mask, ndimage.binary_erosion(anomaly_mask))
    inner_edge_coords = np.stack(np.nonzero(inner_edge))
    inner_edge_distances = norm(inner_edge_coords - np.reshape(def_centre, (-1, 1)), axis=0)
    avg_edge_dist = np.mean(inner_edge_distances)

    max_push_dist = np.sqrt(avg_edge_dist ** 2 / 2)
    curr_min_push_dist = max_push_dist * 0.1
    curr_max_push_dist = max_push_dist * 0.9

    if 'test_push_dist' in kwargs:
        push_dist = min(kwargs['test_push_dist'], curr_max_push_dist)
    else:

        glob_min_push_dist = kwargs['min_push_dist']
        glob_max_push_dist = kwargs['max_push_dist']

        if glob_min_push_dist >= curr_max_push_dist:
            push_dist = curr_max_push_dist
        elif glob_max_push_dist <= curr_min_push_dist:
            # This should be very unlikely, if a sensible global max push is chosen
            push_dist = curr_min_push_dist
        else:
            push_dist = np.random.uniform(max(curr_min_push_dist, glob_min_push_dist),
                                          min(curr_max_push_dist, glob_max_push_dist))

    return push_dist, avg_edge_dist


class BendSourceDeformationTask(SourceDeformationTask):
    def get_deform_factor(self, def_centre: np.ndarray[int], anomaly_mask: np.ndarray[bool]) -> float:

        push_dist, max_push_dist = calc_perpendicular_deformation_dist(def_centre, anomaly_mask, **self.all_kwargs)

        mid_x = max_push_dist / 2 + np.sqrt(push_dist ** 2 / 2)
        mid_y = max_push_dist / 2 - np.sqrt(push_dist ** 2 / 2)

        # Equation is y = max_push_dist * (x / max_push_dist) ^ factor
        # Rearrange to get factor
        return np.log(mid_y / max_push_dist) / np.log(mid_x / max_push_dist)


class BendSinkDeformationTask(SinkDeformationTask):
    def get_deform_factor(self, def_centre: np.ndarray[int], anomaly_mask: np.ndarray[bool]) -> float:
        push_dist, max_push_dist = calc_perpendicular_deformation_dist(def_centre, anomaly_mask, **self.all_kwargs)

        mid_x = max_push_dist / 2 - np.sqrt(push_dist ** 2 / 2)
        mid_y = max_push_dist / 2 + np.sqrt(push_dist ** 2 / 2)

        # Equation is y = max_push_dist (1 - (1 - x/max_push_dist) ^ factor)
        # Rearrange to get factor
        return np.log(1 - mid_y / max_push_dist) / np.log(1 - mid_x / max_push_dist)
