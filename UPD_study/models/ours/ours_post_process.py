from pathlib import Path
from time import time

import torch
from torch.utils.data import DataLoader

from UPD_study.utilities.evaluate import evaluate, get_result_files
from UPD_study.models.ours.ours_trainer import DSET_DICT, get_test_dset_name, init_setup, logger, get_checkpoint_step


class ResultsLoader:

    def __init__(self, results_folder: list[Path], upd_config):
        self.results_folder = results_folder
        self.next_index = 0
        self.upd_config = upd_config

        anomaly_maps_file, anomaly_scores_file = get_result_files(self.results_folder)

        self.anomaly_maps = torch.load(anomaly_maps_file)
        self.anomaly_scores = torch.load(anomaly_scores_file)

    def anom_inference(self, input_imgs, test_samples: bool = False):
        # Assume input_imgs are being input in same order as saved maps
        num_samples = input_imgs.shape[0]
        curr_anomaly_maps = self.anomaly_maps[self.next_index: self.next_index + num_samples].to(input_imgs.device)
        curr_anomaly_scores = self.anomaly_scores[self.next_index: self.next_index + num_samples].to(input_imgs.device)

        if self.upd_config.zero_bg_pred and self.upd_config.modality == 'MRI':
            # Zero out the background
            curr_anomaly_maps = curr_anomaly_maps * (input_imgs > input_imgs.amin(dim=list(range(input_imgs.ndim - 1)), keepdim=True))

        self.next_index = (self.next_index + num_samples) % len(self.anomaly_maps)
        return curr_anomaly_maps, curr_anomaly_scores

    def __len__(self):
        return len(self.anomaly_maps)

def main():

    upd_config, h_config = init_setup()
    assert upd_config.eval, "This script is only for evaluation"

    model_folder = Path(h_config.output_dir)

    logger.info(f"Model folder: {model_folder}")

    # For each, find where results are stored
    test_dset_name = get_test_dset_name(upd_config)
    eval_prefix = f'eval_{test_dset_name}'

    model_subfolders = [f for f in model_folder.iterdir(
    ) if f.is_dir() and f.name.startswith(eval_prefix)]
    # Get the most recent one
    fold_results_folders = sorted(
        model_subfolders, key=lambda x: x.stat().st_ctime, reverse=True)

    if len(fold_results_folders) == 0:
        logger.warning(f"Eval prefix: {eval_prefix}")
        logger.warning(f"Model subfolders: {list(model_folder.iterdir())}")
        raise ValueError(f"No results folder found for {model_folder}")

    results_folder = fold_results_folders[0]

    if upd_config.zero_bg_pred and upd_config.modality == 'MRI':
        eval_prefix += '_zero_bg'
    upd_config.eval_dir = model_folder.with_name(f"{eval_prefix}_{model_folder.stem}")

    upd_config.step = get_checkpoint_step(results_folder)

    assert results_folder.exists()
    print('Found results folder:', results_folder)

    # Load the saved results
    start = time()
    ensemble = ResultsLoader(results_folder, upd_config)
    logger.info(f"Generating ensemble took {time() - start:.2f} seconds")

    # Load test data
    test_dset, _ = DSET_DICT[upd_config.modality](upd_config, train=False)
    assert len(test_dset) == len(ensemble), f"Test dset has {len(test_dset)} samples, but ensemble has {len(ensemble)}"

    test_dataloader = DataLoader(
        test_dset,
        shuffle=False,
        batch_size=h_config.validation_batch_size,
        num_workers=h_config.dataloader_num_workers,
    )

    test_dataloader = upd_config.accelerator.prepare(test_dataloader)
    evaluate(upd_config, test_dataloader, ensemble.anom_inference)

    # Closes trackers, so put at very end of everything
    upd_config.accelerator.end_training()

if __name__ == "__main__":
    main()
