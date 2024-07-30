from typing import Callable, Optional

import numpy as np
from scipy.ndimage import distance_transform_edt


from UPD_study.models.ours.task_utils.base_task import BaseTask
from UPD_study.models.ours.task_utils.labelling import AnomalyLabeller
from UPD_study.models.ours.task_utils.noise_variants import generate_fractal_noise
from UPD_study.models.ours.task_utils.utils import get_patch_image_slices


class SmoothIntensityChangeTask(BaseTask):

    def __init__(self, sample_labeller: Optional[AnomalyLabeller], intensity_task_scale: float,
                 only_hyper: bool = False, only_hypo: bool = False, **all_kwargs):
        super().__init__(sample_labeller, **all_kwargs)

        self.intensity_task_scale = intensity_task_scale
        self.only_hyper = only_hyper
        self.only_hypo = only_hypo
        assert not (self.only_hyper and self.only_hypo), \
            "Cannot set both only_hyper and only_hypo to True"

    def get_intensity_map(self, sample: np.ndarray[float], sample_mask: Optional[np.ndarray[bool]],
                          anomaly_corner: np.ndarray[int], anomaly_mask: np.ndarray[bool]) \
            -> np.ndarray[float]:
        num_chans = sample.shape[0]
        sample_shape = sample.shape[1:]
        num_dims = len(sample_shape)

        if self.only_hyper:
            intensity_sign = 1
        elif self.only_hypo:
            intensity_sign = -1
        else:
            intensity_sign = np.random.choice([1, -1], size=num_chans)

        # Randomly negate, so some intensity changes are subtractions
        intensity_changes = (self.intensity_task_scale / 2 + np.random.gamma(3, self.intensity_task_scale)) \
            * intensity_sign

        return np.reshape(intensity_changes, [-1] + [1] * num_dims)

    def augment_sample(self, sample: np.ndarray[float], sample_mask: Optional[np.ndarray[bool]],
                       anomaly_corner: np.ndarray[int], anomaly_mask: np.ndarray[bool],
                       anomaly_intersect_fn: Callable[[np.ndarray[float], np.ndarray[float]], np.ndarray[float]]) \
            -> np.ndarray[float]:

        sample_shape = sample.shape[1:]

        dist_map = distance_transform_edt(anomaly_mask)
        min_shape_dim = np.min(sample_shape)

        smooth_dist = np.minimum(
            min_shape_dim * (0.02 + np.random.gamma(3, 0.01)), np.max(dist_map))
        smooth_dist_map = dist_map / smooth_dist
        smooth_dist_map[smooth_dist_map > 1] = 1

        anomaly_patch_slices = get_patch_image_slices(
            anomaly_corner, anomaly_mask.shape)
        # anomaly_pixel_stds = np.array([np.std(c[anomaly_mask]) for c in sample[anomaly_patch_slices]])

        intensity_change_map = smooth_dist_map * \
            self.get_intensity_map(sample, sample_mask,
                                   anomaly_corner, anomaly_mask)
        new_patch = sample[anomaly_patch_slices] + intensity_change_map
        spatial_axis = tuple(range(1, len(sample.shape)))
        sample[anomaly_patch_slices] = np.clip(new_patch,
                                               sample.min(
                                                   axis=spatial_axis, keepdims=True),
                                               sample.max(axis=spatial_axis, keepdims=True))

        return sample


class SmoothNoiseAdditionTask(SmoothIntensityChangeTask):

    def get_intensity_map(self, sample: np.ndarray[float], sample_mask: Optional[np.ndarray[bool]],
                          anomaly_corner: np.ndarray[int], anomaly_mask: np.ndarray[bool]) \
            -> np.ndarray[float]:
        # Instead of a uniform value, use a random noise map generated from a fractal noise generator
        # (one layer of fractal noise is equivalent to Perlin noise)

        num_chans = sample.shape[0]
        anom_shape = anomaly_mask.shape
        num_dims = len(anom_shape)

        # Randomly negate, so some intensity changes are subtractions
        # TODO: should we sample a separate intensity change for each channel?
        # If so, we should probably use multiple intensity scales, one for each channel
        if self.only_hyper:
            intensity_sign = 1
        elif self.only_hypo:
            intensity_sign = -1
        else:
            intensity_sign = np.random.choice([1, -1], size=num_chans)

        intensity_changes = (self.intensity_task_scale / 2 + np.random.gamma(3, self.intensity_task_scale)) \
            * intensity_sign

        mean_dim = np.mean(anom_shape)

        max_num_intervals = mean_dim / 3

        first_core_num_intervals = 1 + \
            self.rng.exponential((max_num_intervals - 1) / 4.5)

        max_num_layers = np.floor(
            np.log2(max_num_intervals / first_core_num_intervals)) + 1

        target_num_layers = min(max_num_layers, 1 + self.rng.poisson(1))

        all_core_num_intervals = [first_core_num_intervals]

        while len(all_core_num_intervals) < target_num_layers and all_core_num_intervals[-1] < max_num_intervals:

            # Could potentially widen the range of the normal distribution
            # But sticking with this for now as normally the number of intervals
            # Just increases by a factor of 2
            all_core_num_intervals.append(
                all_core_num_intervals[-1] * 2 * self.rng.normal(1, 0.1))

        # As this is a product of a product, there is a small chance that the later layers which should
        # have more intervals will have less intervals than the earlier layers
        # Seems pretty low tho, let's see what the images look like first.
        all_num_intervals = [tuple(self.rng.normal(1, 0.1, size=num_dims) * core_num_intervals)
                             for core_num_intervals in all_core_num_intervals]

        noise = generate_fractal_noise(anom_shape, all_num_intervals, amplitudes=[
                                       2 ** -i for i in range(len(all_num_intervals))])

        # Scale to range 0-1
        noise -= np.min(noise)
        noise /= np.max(noise)

        return np.reshape(intensity_changes, [-1] + [1] * num_dims) * noise
