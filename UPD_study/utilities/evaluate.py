from argparse import Namespace
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from UPD_study.utilities.utils import metrics, log


def get_result_files(eval_dir: Path) -> tuple[Path, Path]:
    return eval_dir / 'anomaly_maps.pt', eval_dir / 'anomaly_scores.pt'


def evaluate(config: Namespace, test_loader: DataLoader, val_step: Callable) -> None:
    """
    Common evaluation method. Handles inference on evaluation set, metric calculation,
    logging and the speed benchmark.

    Args:
        config (Namespace): configuration object.
        test_loader (DataLoader): evaluation set dataloader
        val_step (Callable): validation step function
    """

    labels = []
    anomaly_scores = []
    anomaly_maps = []
    inputs = []
    segmentations = []

    # forward pass the testloader to extract anomaly maps, scores, masks, labels
    for input_imgs, mask in tqdm(test_loader, desc="Test set", disable=config.speed_benchmark):
        if not config.using_accelerate:
            input_imgs = input_imgs.to(config.device)

        output = val_step(input_imgs, test_samples=True)

        anomaly_map, anomaly_score = output[:2]

        if config.using_accelerate:
            input_imgs, mask, anomaly_map, anomaly_score = \
                config.accelerator.gather_for_metrics((input_imgs, mask, anomaly_map, anomaly_score))

        batch_input_imgs, batch_masks, batch_anomaly_maps, batch_anomaly_scores = input_imgs.cpu(), \
                mask.cpu(), anomaly_map.cpu(), anomaly_score.cpu()

        inputs.append(batch_input_imgs)

        if config.method == 'Cutpaste' and config.localization:
            anomaly_maps.append(batch_anomaly_maps)
            segmentations.append(batch_masks)

            label = torch.where(batch_masks.sum(dim=(1, 2, 3)) > 0, 1, 0)
            labels.append(label)
            anomaly_scores.append(torch.zeros_like(label))
        elif config.method == 'Cutpaste' and not config.localization:
            segmentations = None
            anomaly_scores.append(batch_anomaly_scores)
            label = torch.where(batch_masks.sum(dim=(1, 2, 3)) > 0, 1, 0)
            labels.append(label)
        else:
            anomaly_maps.append(batch_anomaly_maps)
            segmentations.append(batch_masks)
            anomaly_scores.append(batch_anomaly_scores)
            label = torch.where(batch_masks.sum(dim=(1, 2, 3)) > 0, 1, 0)
            labels.append(label)


    metric_prefix = ''
    if config.modality == 'MRI' and config.sequence == 't1':
        metric_prefix = ('brats' if config.brats_t1 else 'atlas') + '/'


    if not config.using_accelerate or config.accelerator.is_main_process:
        metrics(config, anomaly_maps, segmentations, anomaly_scores, labels, metric_prefix)

    if config.eval_dir is not None:
        config.eval_dir.mkdir(parents=True, exist_ok=True)
        anomaly_maps_file, anomaly_scores_file = get_result_files(config.eval_dir)
        torch.save(torch.cat(anomaly_maps), anomaly_maps_file)
        torch.save(torch.cat(anomaly_scores), anomaly_scores_file)
        # For debugging purposes, save the labels to check ordering is the same
        torch.save(torch.cat(labels), anomaly_scores_file.with_stem('labels'))

    # do a single forward pass to extract images to log
    # the batch size is num_images_log for test_loaders, so only a single forward pass necessary
    input_imgs, mask = next(iter(test_loader))

    if not config.using_accelerate:
        input_imgs = input_imgs.to(config.device)

    output = val_step(input_imgs, test_samples=True)
    if config.using_accelerate:
        input_imgs, mask, output = config.accelerator.gather_for_metrics((input_imgs, mask, output))

    if not config.using_accelerate or config.accelerator.is_main_process:

        anomaly_maps = output[0]

        log({f'anom_val/{metric_prefix}input images': input_imgs.cpu(),
            f'anom_val/{metric_prefix}targets': mask.cpu(),
            f'anom_val/{metric_prefix}anomaly maps': anomaly_maps.cpu()}, config)

        # if recon based method, len(x)==3 because val_step returns reconstructions.
        # if thats the case log the reconstructions
        if len(output) == 3:
            log({f'anom_val/{metric_prefix}reconstructions': output[2].cpu()}, config)
