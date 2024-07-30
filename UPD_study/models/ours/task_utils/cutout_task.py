from typing import Callable, Optional

import numpy as np

from UPD_study.models.ours.task_utils.base_task import BaseTask
from UPD_study.models.ours.task_utils.utils import get_patch_image_slices


class Cutout(BaseTask):
    def augment_sample(self, sample: np.ndarray[float], sample_mask: Optional[np.ndarray[bool]],
                       anomaly_corner: np.ndarray[int], anomaly_mask: np.ndarray[bool],
                       anomaly_intersect_fn: Callable[[np.ndarray[float], np.ndarray[float]], np.ndarray[float]]) \
            -> np.ndarray[float]:

        anomaly_patch_slices = get_patch_image_slices(anomaly_corner, anomaly_mask.shape)
        anomaly_image_shape = tuple([sample.shape[0]] + list(anomaly_mask.shape))

        sample[anomaly_patch_slices][np.broadcast_to(anomaly_mask, anomaly_image_shape)] = self.rng.random()

        return sample
