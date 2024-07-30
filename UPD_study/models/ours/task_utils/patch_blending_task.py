import functools
import itertools
from typing import Callable, Tuple, Optional
import warnings

import numpy as np
from scipy.ndimage import affine_transform

from UPD_study.models.ours.task_utils.base_task import BaseTask
from UPD_study.models.ours.task_utils.blending_methods import cut_paste, patch_interpolation, poisson_image_editing
from UPD_study.models.ours.task_utils.labelling import AnomalyLabeller
from UPD_study.models.ours.task_utils.utils import accumulate_rotation, accumulate_scaling,  get_patch_image_slices


class BasePatchBlendingTask(BaseTask):

    def __init__(self, sample_labeller: Optional[AnomalyLabeller],
                 get_source_sample_and_mask: Callable[[], Tuple[np.ndarray[float], Optional[np.ndarray[bool]]]],
                 blend_images: Callable[[np.ndarray[float], np.ndarray[float], np.ndarray[int], np.ndarray[bool]],
                                        np.ndarray[float]],
                 **all_kwargs):
        super().__init__(sample_labeller, **all_kwargs)
        self.get_source_sample_and_mask = get_source_sample_and_mask
        self.blend_images = blend_images

    def augment_sample(self, sample: np.ndarray[float], sample_mask: Optional[np.ndarray[bool]],
                       anomaly_corner: np.ndarray[int], anomaly_mask: np.ndarray[bool],
                       anomaly_intersect_fn: Callable[[np.ndarray[float], np.ndarray[float]], np.ndarray[float]]) \
            -> np.ndarray[float]:

        num_channels = sample.shape[0]
        num_dims = len(sample.shape[1:])

        # Sample source to blend into current sample
        source_sample, source_sample_mask = self.get_source_sample_and_mask()
        source_sample_shape = np.array(source_sample.shape[1:])
        assert len(source_sample_shape) == num_dims, 'Source and target have different number of spatial dimensions: ' \
                                                     f's-{len(source_sample_shape)}, t-{num_dims}'
        if source_sample.shape[0] != num_channels:
            warnings.warn('Source and target have different number of channels: '
                         f's-{source_sample.shape[0]}, t-{num_channels}')
            if source_sample.shape[0] < num_channels:
                # Randomly sample channels to fill in missing channels
                source_sample = np.concatenate([source_sample,
                                                self.rng.choice(sample, size=num_channels - source_sample.shape[0],
                                                                axis=0, replace=True)],
                                               axis=0)
            else:
                # Randomly sample channels to remove
                source_sample = self.rng.choice(source_sample, size=num_channels, axis=0, replace=False)

        # Compute INVERSE transformation matrix for parameters (rotation, resizing)
        # This is the backwards operation (final source region -> initial source region).

        trans_matrix = functools.reduce(lambda m, ds: accumulate_rotation(m,
                                                                          self.rng.uniform(-np.pi / 4, np.pi / 4),
                                                                          ds),
                                        itertools.combinations(range(num_dims), 2),
                                        np.identity(num_dims))

        # Compute effect on corner coords
        target_anomaly_shape = np.array(anomaly_mask.shape)
        corner_coords = np.array(np.meshgrid(*np.stack([np.zeros(num_dims), target_anomaly_shape], axis=-1),
                                             indexing='ij')).reshape(num_dims, 2 ** num_dims)

        trans_corner_coords = trans_matrix @ corner_coords
        min_trans_coords = np.floor(np.min(trans_corner_coords, axis=1))
        max_trans_coords = np.ceil(np.max(trans_corner_coords, axis=1))
        init_grid_shape = max_trans_coords - min_trans_coords

        # Sample scale and clip so that source region isn't too big
        max_scale = np.min(0.8 * source_sample_shape / init_grid_shape)

        # Compute final transformation matrix
        scale_change = 1 + self.rng.exponential(scale=0.1)
        scale_raw = self.rng.choice([scale_change, 1 / scale_change])
        scale = np.minimum(scale_raw, max_scale)

        trans_matrix = accumulate_scaling(trans_matrix, scale)

        # Recompute effect on corner coord
        trans_corner_coords = trans_matrix @ corner_coords
        min_trans_coords = np.floor(np.min(trans_corner_coords, axis=1))
        max_trans_coords = np.ceil(np.max(trans_corner_coords, axis=1))
        final_init_grid_shape = max_trans_coords - min_trans_coords

        if np.any(final_init_grid_shape > source_sample_shape):
            print('Source shape: ', source_sample_shape)
            print('Source extracted shape without scale:, ', init_grid_shape)
            print('Resize factor: ', scale)
            print('Final extracted shape:, ', final_init_grid_shape)
            print()

        # Choose anomaly source location
        final_init_grid_shape = final_init_grid_shape.astype(int)
        min_corner = np.zeros(len(source_sample_shape))
        max_corner = source_sample_shape - final_init_grid_shape

        if source_sample_mask is not None:
            source_mask_coords = np.argwhere(source_sample_mask)

            final_grid_half = final_init_grid_shape // 2
            valid_middle_coords = [c for c in source_mask_coords
                                   if all(c - final_grid_half >= min_corner) and all(c - final_grid_half <= max_corner)]

            if valid_middle_coords == []:
                print('No valid source locations found: \n' +
                      f'Final init grid shape {final_init_grid_shape}\n' +
                      f'All mask coords {source_mask_coords}\n' +
                      f'Final grid half {final_grid_half}\n' +
                      f'Min corner {min_corner}\n' +
                      f'Max corner {max_corner}\n' +
                      f'Valid middle coords {valid_middle_coords}')
                print('Recursing to try a different source sample')

                # This is a super rare case, only observed ~1/40000 chance when using slices of brain MRI
                # Happens when the source sample mask is very small and off-center, and the attempted
                # anomaly size is too big, so it cannot remain inside the bounds of the source sample whilst also
                # being centered on the source mask.
                # In this case, we try again as although we could choose a nearby patch location, ut would mean the
                # anomaly would be largely background, so probably not that useful. 
                return self.augment_sample(sample, sample_mask, anomaly_corner, anomaly_mask, anomaly_intersect_fn)

            source_corner = self.rng.choice(valid_middle_coords) - final_grid_half

        else:
            # Choose source location
            source_corner = self.rng.integers(min_corner, max_corner, endpoint=True)

        # Extract source
        source_orig = source_sample[get_patch_image_slices(source_corner, tuple(final_init_grid_shape))]

        # Because we computed the backwards transformation we don't need to inverse the matrix
        source_to_blend = np.stack([affine_transform(chan, trans_matrix, offset=-min_trans_coords,
                                                     output_shape=tuple(target_anomaly_shape))
                                    for chan in source_orig])

        spatial_axis = tuple(range(1, len(source_sample.shape)))
        # Spline interpolation can make values fall outside domain, so clip to the original range
        source_to_blend = np.clip(source_to_blend,
                                  source_sample.min(axis=spatial_axis, keepdims=True),
                                  source_sample.max(axis=spatial_axis, keepdims=True))

        # As the blending can alter areas outside the mask, update the mask with any effected areas
        aug_sample = self.blend_images(sample, source_to_blend, anomaly_corner, anomaly_mask)
        sample_slices = get_patch_image_slices(anomaly_corner, tuple(anomaly_mask.shape))
        sample_diff = np.mean(np.abs(sample[sample_slices] - aug_sample[sample_slices]), axis=0)
        anomaly_mask[sample_diff > 0.001] = True

        # Return sample with source blended into it
        return aug_sample


class TestCutPastePatchBlender(BasePatchBlendingTask):

    def __init__(self, sample_labeller: Optional[AnomalyLabeller],
                 test_source_sample: np.ndarray[float],
                 test_source_sample_mask: Optional[np.ndarray[bool]], **kwargs):
        super().__init__(sample_labeller, lambda: (test_source_sample, test_source_sample_mask), cut_paste)


class TestPatchInterpolationBlender(BasePatchBlendingTask):
    def __init__(self, sample_labeller: Optional[AnomalyLabeller],
                 test_source_sample: np.ndarray[float],
                 test_source_sample_mask: Optional[np.ndarray[bool]], **all_kwargs):
        super().__init__(sample_labeller, lambda: (test_source_sample, test_source_sample_mask), patch_interpolation,
                         **all_kwargs)


class TestPoissonImageEditingMixedGradBlender(BasePatchBlendingTask):
    def __init__(self, sample_labeller: Optional[AnomalyLabeller],
                 test_source_sample: np.ndarray[float],
                 test_source_sample_mask: Optional[np.ndarray[bool]],
                 **all_kwargs):
        super().__init__(sample_labeller, lambda: (test_source_sample, test_source_sample_mask),
                         lambda *args: poisson_image_editing(*args, True),
                         **all_kwargs)


class TestPoissonImageEditingSourceGradBlender(BasePatchBlendingTask):
    def __init__(self, sample_labeller: Optional[AnomalyLabeller],
                 test_source_sample: np.ndarray[float],
                 test_source_sample_mask: Optional[np.ndarray[bool]], **all_kwargs):
        super().__init__(sample_labeller, lambda: (test_source_sample, test_source_sample_mask),
                         lambda *args: poisson_image_editing(*args, False),
                         **all_kwargs)


class PoissonImageEditingMixedGradBlender(BasePatchBlendingTask):
    def __init__(self, sample_labeller: Optional[AnomalyLabeller],
                 get_source_sample_and_mask: Callable[[], Tuple[np.ndarray[float], Optional[np.ndarray[bool]]]],
                 **all_kwargs):
        super().__init__(sample_labeller,
                         get_source_sample_and_mask,
                         lambda *args: poisson_image_editing(*args, True),
                         **all_kwargs)


class PoissonImageEditingSourceGradBlender(BasePatchBlendingTask):
    def __init__(self, sample_labeller: Optional[AnomalyLabeller],
                 get_source_sample_and_mask: Callable[[], Tuple[np.ndarray[float], Optional[np.ndarray[bool]]]],
                 **all_kwargs):
        super().__init__(sample_labeller,
                         get_source_sample_and_mask,
                         lambda *args: poisson_image_editing(*args, False),
                         **all_kwargs)
