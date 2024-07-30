from pathlib import Path
from time import time

import torch
from torch.utils.data import DataLoader

from UPD_study.utilities.evaluate import evaluate, get_result_files
from UPD_study.utilities.utils import log
from UPD_study.models.ours.ours_trainer import DSET_DICT, get_test_dset_name, init_setup, logger


class ResultsEnsemble:

    def __init__(self, results_folders: list[Path], upd_config):
        self.results_folders = results_folders
        self.upd_config = upd_config
        self.next_index = 0

        self.ensemble_anomaly_maps = None
        self.ensemble_anomaly_scores = None

        num_models = len(results_folders)
        self.anomaly_labels = None
        self.anomaly_labels_path = None
        for r_f in results_folders:
            anomaly_maps_file, anomaly_scores_file = get_result_files(r_f)
            curr_anomaly_maps = torch.load(anomaly_maps_file) / num_models
            curr_anomaly_scores = torch.load(anomaly_scores_file) / num_models

            if self.ensemble_anomaly_maps is None:
                self.ensemble_anomaly_maps = curr_anomaly_maps
                self.ensemble_anomaly_scores = curr_anomaly_scores
            else:
                self.ensemble_anomaly_maps += curr_anomaly_maps
                self.ensemble_anomaly_scores += curr_anomaly_scores

            curr_anomaly_labels_path = anomaly_scores_file.with_stem('labels')
            if self.anomaly_labels is None:
                self.anomaly_labels_path = curr_anomaly_labels_path
                self.anomaly_labels = torch.load(self.anomaly_labels_path)
            else:
                curr_anom_labels = torch.load(curr_anomaly_labels_path)
                assert torch.equal(self.anomaly_labels, curr_anom_labels), "Anomaly labels do not match"


    def ensemble_anom_inference(self, input_imgs, test_samples: bool = False):
        # Assume input_imgs are being input in same order as saved maps
        num_samples = input_imgs.shape[0]
        anomaly_maps = self.ensemble_anomaly_maps[self.next_index: self.next_index + num_samples].to(input_imgs.device)
        anomaly_scores = self.ensemble_anomaly_scores[self.next_index: self.next_index + num_samples].to(input_imgs.device)

        if self.upd_config.zero_bg_pred and self.upd_config.modality == 'MRI':
            # Zero out the background
            anomaly_maps = anomaly_maps * (input_imgs > input_imgs.amin(dim=list(range(input_imgs.ndim - 1)), keepdim=True))

            anomaly_scores = anomaly_maps.mean(dim=(1, 2, 3))

        self.next_index = (self.next_index + num_samples) % len(self.ensemble_anomaly_scores)
        return anomaly_maps, anomaly_scores

    def __len__(self):
        return len(self.ensemble_anomaly_scores)

def main():

    upd_config, h_config = init_setup()
    assert upd_config.fold == 'ensemble', "This script is only for ensemble models"
    assert upd_config.eval, "This script is only for evaluation"
    upd_config.step = 0 # Needed for logging

    output_dir_path = Path(h_config.output_dir)

    # Find model folders
    model_folder_prefix = upd_config.modality
    if upd_config.modality == 'MRI':
        model_folder_prefix += f'_{upd_config.sequence}'

    model_folders = [f for f in output_dir_path.parent.iterdir() if f.name.startswith(model_folder_prefix) and
                     f.is_dir() and 'ensemble' not in f.name]

    logger.info(f"Model folders: {model_folders}")

    # For each, find where results are stored
    test_dset_name = get_test_dset_name(upd_config)
    if upd_config.ssim_eval:
        eval_prefix = f"eval_{test_dset_name}"
    else:
        eval_prefix = f"eval_mae_{test_dset_name}"
    result_folders = []
    for model_folder in model_folders:
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
        result_folders.append(results_folder)

        assert results_folder.exists()
        logger.info(f'Found results folder: {results_folder}')

    if upd_config.figure_images:
        logger.info('Logging ensemble examples')
        fold_examples = [torch.load(rf / 'test_imgs.pt') for rf in result_folders]

        image_key = [k for k in fold_examples[0].keys() if 'input images' in k][0]
        label_key = [k for k in fold_examples[0].keys() if 'targets' in k][0]
        anom_maps_key = [k for k in fold_examples[0].keys() if 'anomaly maps' in k][0]
        restored_images_key = [k for k in fold_examples[0].keys() if 'restored images' in k][0]

        for i, fold_ex in enumerate(fold_examples):
            assert image_key in fold_ex, f"Image key {image_key} not found in {fold_ex.keys()}"
            assert torch.equal(fold_ex[image_key], fold_examples[0][image_key]), \
                f"Image key {image_key} not equal in fold {result_folders[i]} (compared with {result_folders[0]})"

            assert label_key in fold_ex, f"Label key {label_key} not found in {fold_ex.keys()}"
            assert torch.equal(fold_ex[label_key], fold_examples[0][label_key]), \
                f"Label key {label_key} not equal in fold {result_folders[i]} (compared with {result_folders[0]})"

            assert anom_maps_key in fold_ex, f"Anom maps key {anom_maps_key} not found in {fold_ex.keys()}"
            assert restored_images_key in fold_ex, f"Restored images key {restored_images_key} not found in {fold_ex.keys()}"

        ensemble_results = {}

        ensemble_results[image_key] = fold_examples[0][image_key]
        ensemble_results[label_key] = fold_examples[0][label_key]

        all_anom_maps = torch.stack([fe[anom_maps_key] for fe in fold_examples])
        avg_anom_maps = all_anom_maps.mean(dim=0)
        ensemble_results[anom_maps_key] = torch.clamp(avg_anom_maps, 0, 1)

        closest_to_avg = torch.argmin((all_anom_maps - avg_anom_maps).abs().mean(dim=tuple(range(1, all_anom_maps.ndim))), dim=0).item()
        logger.info(f"Closest to avg: {closest_to_avg} - {result_folders[closest_to_avg]}")
        ensemble_results[anom_maps_key + '_closest'] = torch.clamp(fold_examples[closest_to_avg][anom_maps_key], 0, 1)
        ensemble_results[restored_images_key + '_closest'] = fold_examples[closest_to_avg][restored_images_key]

        ensemble_results[restored_images_key + '_avg'] = torch.stack([fe[restored_images_key] for fe in fold_examples]).mean(dim=0)

        log(ensemble_results, upd_config)

    else:

        # Load the saved results
        start = time()
        ensemble = ResultsEnsemble(result_folders, upd_config)
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
        evaluate(upd_config, test_dataloader, ensemble.ensemble_anom_inference)

    # Closes trackers, so put at very end of everything
    upd_config.accelerator.end_training()

if __name__ == "__main__":
    main()
