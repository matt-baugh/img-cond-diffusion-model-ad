from argparse import ArgumentParser, Namespace
from copy import deepcopy
import logging
import math
import os
from pathlib import Path
import random
import shutil
import time

from einops import rearrange
from omegaconf import OmegaConf, DictConfig
from tqdm.auto import tqdm
from packaging import version

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from UPD_study.utilities.common_data import BaseDataset
from UPD_study.data.dataloaders.MRI import get_datasets_mri
from UPD_study.data.dataloaders.CXR import get_datasets_cxr
from UPD_study.data.dataloaders.RF import get_datasets_rf
from UPD_study.models.ours.early_stopping import MovingAverageEarlyStopper
from UPD_study.models.ours.restoration_task import get_task_split, RestorationTask
from UPD_study.utilities.common_config import common_config
from UPD_study.utilities.utils import misc_settings, ssim_map, str_to_bool, log
from UPD_study.utilities.evaluate import evaluate


if is_wandb_available():
    import wandb
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DSET_DICT = {
    'MRI': get_datasets_mri,
    'CXR': get_datasets_cxr,
    'RF': get_datasets_rf
    # TODO: add imagenet
}


def get_checkpoint_step(checkpoint_folder):
    return int(checkpoint_folder.name.split("-")[1])


def get_checkpoint_paths(checkpoint_folder):
    return sorted([d for d in Path(checkpoint_folder).iterdir() if d.name.startswith("checkpoint")],
                    key=get_checkpoint_step)


def get_best_checkpoint_path(checkpoints_folder, return_all=False):
    all_checkpoints = get_checkpoint_paths(checkpoints_folder)
    if len(all_checkpoints) == 0:
        return None
    best_checkpoints = [d for d in all_checkpoints if "best" in d.name]
    if len(best_checkpoints) == 0:
        return None

    if return_all:
        return best_checkpoints

    # get_checkpoint_paths returns sorted by step, so we can just take the last one
    return best_checkpoints[-1]


def anom_inference(unet, scheduler, h_config, upd_config, accelerator, weight_dtype, input_imgs):

    scheduler.set_timesteps(h_config.validation_timesteps)
    timesteps = scheduler.timesteps

    unet.eval()
    noise = torch.randn_like(
        input_imgs, device=accelerator.device, dtype=weight_dtype)  # B x C x H x W

    # TODO: if we do it in latent space, apply encoder here

    with torch.no_grad(), torch.autocast("cuda"):
        for t in timesteps:
            if h_config.validation_guidance > 0:
                noise_input = torch.cat([noise] * 2)
                conditional_input = torch.cat(
                    [torch.randn_like(input_imgs), input_imgs])  # uncond + cond
            else:
                noise_input = noise
                conditional_input = input_imgs

            noise_model_input = scheduler.scale_model_input(
                noise_input, timestep=t)

            # logger.info(f"noise_model_input {noise_model_input.device}, conditional_input {conditional_input.device}")
            model_input = torch.cat(
                [noise_model_input, conditional_input], dim=1)  # concat on channels

            noise_pred = unet(model_input, t).sample

            if h_config.validation_guidance > 0:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + h_config.validation_guidance * \
                    (noise_pred_cond - noise_pred_uncond)

            noise = scheduler.step(noise_pred, t, noise).prev_sample

    # TODO: if we do it in latent space, apply decoder here

    # Noise is now a batch of fully sampled images, matching normalisation/format of input
    diffusion_output = noise

    # Anomaly map
    if upd_config.ssim_eval:
        anomaly_map = ssim_map(diffusion_output, input_imgs)
    else:
        anomaly_map = (
            (diffusion_output - input_imgs).abs().mean(1, keepdim=True)
        )

    if upd_config.zero_bg_pred and upd_config.modality == 'MRI':
        # Zero out the background
        anomaly_map = anomaly_map * (input_imgs > input_imgs.amin(dim=list(range(input_imgs.ndim - 1)), keepdim=True))

    # TODO: is this a good aggregation
    anomaly_score = anomaly_map.mean(dim=(1, 2, 3))

    # Uncenter images for visualisation if necessary
    if upd_config.center:
        diffusion_output = diffusion_output / 2 + 0.5
    restored_imgs = diffusion_output.clamp(0, 1)

    return anomaly_map, anomaly_score, restored_imgs


def gen_anomaly_inference_function(unet, scheduler, h_config, upd_config, accelerator, weight_dtype):
    def configured_anom_inference(input_imgs, test_samples: bool = False):
        return anom_inference(unet, scheduler, h_config, upd_config, accelerator, weight_dtype, input_imgs)
    return configured_anom_inference


def log_validation(unet, scheduler, h_config, upd_config, accelerator, weight_dtype, val_dataset: BaseDataset):
    unet = accelerator.unwrap_model(unet)

    if h_config.enable_xformers_memory_efficient_attention:
        unet.enable_xformers_memory_efficient_attention()

    if h_config.seed is None:
        generator = None
    else:
        generator = torch.Generator(
            device=accelerator.device).manual_seed(h_config.seed)

    indices = torch.randint(0, len(val_dataset), (h_config.validation_samples,),
                            generator=generator, device=accelerator.device, dtype=torch.long)

    img_pairs = [val_dataset[i] for i in indices]
    conditionals = [p[0] for p in img_pairs]
    targets = [p[1] for p in img_pairs]
    if h_config.get("uncond_p", 0.1) > 0:
        #  add a blank image to check the unconditional generation capability of the model
        conditionals.append(torch.zeros_like(conditionals[0]))
        targets.append(torch.zeros_like(targets[0]))

    conditionals = torch.stack(conditionals, dim=0).to(
        accelerator.device)  # B x C x H x W
    targets = torch.stack(targets, dim=0).to(
        accelerator.device)  # B x C x H x W

    aug_diff = (targets - conditionals).pow(2).mean(dim=1, keepdim=True)

    start = time.time()
    # B x C x H x W, intensity range should match that of conditional
    anom_maps, _, restored_imgs = anom_inference(
        unet, scheduler, h_config, upd_config, accelerator, weight_dtype, conditionals)
    logger.info(
        f"Generating validation examples took {time.time() - start:.2f} seconds")

    if val_dataset.center:
        conditionals = conditionals / 2 + 0.5
        # Restorted images are already uncentered
        targets = targets / 2 + 0.5

    # Scale anomaly maps for visualisation
    anom_maps /= anom_maps.max()

    if conditionals.shape[1] != anom_maps.shape[1]:
        assert anom_maps.shape[1] == 1
        assert aug_diff.shape[1] == 1
        anom_maps = anom_maps.repeat(1, conditionals.shape[1], 1, 1)
        aug_diff = aug_diff.repeat(1, conditionals.shape[1], 1, 1)

    # concat with conditionals
    images = torch.cat([conditionals, restored_imgs, targets,
                       anom_maps, aug_diff], dim=2)  #  vertical concat

    images = images.clamp(0, 1)
    images = (images * 255).to(torch.uint8)

    images = images.cpu().numpy()
    # horizontal concat for display
    images = rearrange(images, "b c h w -> h (b w) c")

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log({"validation": wandb.Image(
                images, caption="Top to bottom: Input, Restoration, Original, Anomaly Map, Input - Original")})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    torch.cuda.empty_cache()

    return images


def log_test_examples(unet, scheduler, h_config, upd_config, accelerator, weight_dtype, val_dataset: BaseDataset, indices=None):
    unet = accelerator.unwrap_model(unet)
    assert upd_config.eval_dir is not None

    if h_config.enable_xformers_memory_efficient_attention:
        unet.enable_xformers_memory_efficient_attention()

    if h_config.seed is None:
        generator = None
    else:
        generator = torch.Generator(
            device=accelerator.device).manual_seed(h_config.seed)

    if indices is None:
        indices = torch.randint(0, len(val_dataset), (h_config.validation_samples,),
                                generator=generator, device=accelerator.device, dtype=torch.long)
    else:
        indices = torch.tensor(indices, device=accelerator.device, dtype=torch.long)

    img_pairs = [val_dataset[i] for i in indices]
    test_imgs = [p[0] for p in img_pairs]
    test_segs = [p[1] for p in img_pairs]

    test_imgs = torch.stack(test_imgs, dim=0).to(
        accelerator.device)  # B x C x H x W

    start = time.time()
    # B x C x H x W, intensity range should match that of conditional
    anom_maps, _, restored_imgs = anom_inference(
        unet, scheduler, h_config, upd_config, accelerator, weight_dtype, test_imgs)
    logger.info(
        f"Generating validation examples took {time.time() - start:.2f} seconds")

    if val_dataset.center:
        test_imgs = test_imgs / 2 + 0.5
        # Restored images are already uncentered

    if upd_config.zero_bg_pred and upd_config.modality == 'MRI':
        # Zero out the background
        anom_maps = anom_maps * (test_imgs > test_imgs.amin(dim=list(range(test_imgs.ndim - 1)), keepdim=True))

    metric_prefix = ''
    if upd_config.modality == 'MRI' and upd_config.sequence == 't1':
        metric_prefix = ('brats' if upd_config.brats_t1 else 'atlas') + '/'

    example_results = {
        f'anom_test/{metric_prefix}input images': test_imgs.cpu(),
        f'anom_test/{metric_prefix}targets': torch.stack(test_segs, dim=0).cpu(),
        f'anom_test/{metric_prefix}anomaly maps': torch.clamp(anom_maps.cpu(), 0, 1),
        f'anom_test/{metric_prefix}restored images': restored_imgs.cpu()
    }

    log(example_results, upd_config)

    upd_config.eval_dir.mkdir(parents=True, exist_ok=True)
    torch.save(example_results, upd_config.eval_dir / 'test_imgs.pt')
    torch.cuda.empty_cache()


def parse_args():
    parser = ArgumentParser()
    parser = common_config(parser)

    parser.add_argument(
        "--h_config", type=str, default=Path(__file__).parent / "omega_config_big_split.yaml", help="Path to the config file.")
    parser.add_argument("--fold", type=str, default="0", help="Fold number or 'ensemble'.")
    parser.add_argument("--p_no_aug", type=float, default=0.2,
                        help="Probability of no augmentation.")
    parser.add_argument("--zero_bg_pred", type=str_to_bool, default=True,
                        help='Apply post-processing to zero prediction in background')
    parser.add_argument("--figure_images", type=str_to_bool, default=False)
    # parser.add_argument()
    upd_config = parser.parse_args()

    upd_config.method = 'ours'
    upd_config.model_dir_path = Path(__file__).parents[0]
    upd_config.ignore_wandb = True  # Disable benchmark wandb because use our own
    misc_settings(upd_config)
    upd_config.using_accelerate = True


    h_config = OmegaConf.load(upd_config.h_config)
    # Set unet parameters to match data
    h_config.unet.sample_size = upd_config.image_size
    h_config.unet.in_channels = 2 * upd_config.img_channels
    h_config.unet.out_channels = upd_config.img_channels
    upd_config.batch_size = h_config.train_batch_size

    experiment_name = upd_config.modality

    if upd_config.modality == 'MRI':
        experiment_name += f'_{upd_config.sequence}'

        # if upd_config.eval and upd_config.sequence == 't1':
        #     experiment_name += '_brats' if upd_config.brats_t1 else '_atlas'

    experiment_name += f'_fold{upd_config.fold}'

    h_config.output_dir = os.path.join(h_config.output_dir, experiment_name)
    return upd_config, h_config, experiment_name


def init_setup() -> tuple[Namespace, DictConfig]:
    upd_config, h_config, experiment_name = parse_args()

    # Setup accelerator
    logging_dir = os.path.join(h_config.output_dir, h_config.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=h_config.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=h_config.gradient_accumulation_steps,
        mixed_precision=h_config.mixed_precision,
        log_with=h_config.report_to,
        project_config=accelerator_project_config,
    )
    upd_config.device = accelerator.device
    upd_config.logger = accelerator
    upd_config.accelerator = accelerator

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = OmegaConf.to_container(h_config, resolve=True)
        # tracker_config = dict(vars(tracker_config))
        accelerator.init_trackers(
            h_config.tracker_project_name,
            tracker_config,
            init_kwargs={
                "wandb": {
                    "group": experiment_name
                },
            },
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        # datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        # datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if h_config.seed is not None:
        set_seed(h_config.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if h_config.output_dir is not None:
            os.makedirs(h_config.output_dir, exist_ok=True)

    return upd_config, h_config


def main():
    upd_config, h_config = init_setup()
    assert upd_config.fold != 'ensemble', "This script is not for individual folds (training and evaluation)"

    accelerator = upd_config.accelerator

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler()
    # Get the target for loss depending on the prediction type
    if h_config.prediction_type is not None:
        # set prediction_type of scheduler if defined
        noise_scheduler.register_to_config(
            prediction_type=h_config.prediction_type)
    kwargs = OmegaConf.to_container(h_config.unet, resolve=True)
    unet = UNet2DModel(**kwargs)

    # Set unet to trainable
    unet.train()

    # Create EMA for the unet.
    if h_config.use_ema:
        kwargs = OmegaConf.to_container(h_config.unet, resolve=True)
        ema_unet = UNet2DModel(**kwargs)
        ema_unet = EMAModel(ema_unet.parameters(),
                            model_cls=UNet2DModel, model_config=ema_unet.config)

    if h_config.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during "
                    "training, please update xFormers to at least 0.0.17. See "
                    "https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    early_stopper = MovingAverageEarlyStopper(
        patience=h_config.validation_patience, ma_alpha=h_config.validation_loss_ma_alpha)
    accelerator.register_for_checkpointing(early_stopper)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if h_config.use_ema:
                    ema_unet.save_pretrained(
                        os.path.join(output_dir, "unet_ema"))

                for model in models:
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if h_config.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNet2DModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DModel.from_pretrained(
                    input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if h_config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    unet = accelerator.prepare(unet)

    if h_config.use_ema:
        ema_unet.to(accelerator.device)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if h_config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    global_step = 0

    # Potentially load in the weights and states from a previous save
    if h_config.resume_from_checkpoint:
        if h_config.resume_from_checkpoint == "latest":
            # Get the most recent checkpoint
            dirs = get_checkpoint_paths(h_config.output_dir)
            path = dirs[-1] if len(dirs) > 0 else None
        elif h_config.resume_from_checkpoint == "best":
            path = get_best_checkpoint_path(h_config.output_dir)
        else:
            path = Path(h_config.output_dir) / h_config.resume_from_checkpoint

        if path is None:
            accelerator.print(
                f"Checkpoint '{h_config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            h_config.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(path)
            global_step = get_checkpoint_step(path)

            initial_global_step = global_step
    else:
        initial_global_step = 0

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to
    # half-precision as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        h_config.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        h_config.mixed_precision = accelerator.mixed_precision

    if not upd_config.eval:
        # Set up optimizer and learning rate scheduler
        if h_config.scale_lr:
            h_config.learning_rate = (
                h_config.learning_rate * h_config.gradient_accumulation_steps *
                h_config.train_batch_size * accelerator.num_processes
            )

        # Initialize the optimizer
        if h_config.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError as exc:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                ) from exc

            optimizer_cls = bnb.optim.AdamW8bit
            print("Using 8-bit Adam")
        else:
            optimizer_cls = torch.optim.AdamW

        optimizer = optimizer_cls(
            unet.parameters(),
            lr=h_config.learning_rate,
            betas=(h_config.adam_beta1, h_config.adam_beta2),
            weight_decay=h_config.adam_weight_decay,
            eps=h_config.adam_epsilon,
        )

        upd_blending_config = deepcopy(upd_config)
        upd_blending_config.percentage = min(30, upd_config.percentage)
        upd_blending_config.normal_split = 1

        assert upd_config.modality in DSET_DICT, f'Unknown modality {upd_config.modality}'
        ext_blending_dsets = []
        for dset_name in tqdm([k for k in DSET_DICT if k != upd_config.modality],
                              desc='Loading external blending datasets',
                              disable=not accelerator.is_main_process):
            ext_blending_dsets.append(DSET_DICT[dset_name](
                upd_blending_config, train=True)[0])

        # TODO: change this to 2 or 3 once training working
        train_tasks, val_tasks = get_task_split(int(upd_config.fold), 3)
        upd_config.aug_fn = RestorationTask(train_tasks, has_bg=upd_config.modality == 'MRI', center=upd_config.center,
                                            p_no_aug=upd_config.p_no_aug, task_kwargs=h_config.task_kwargs,
                                            external_blending_dsets=ext_blending_dsets)
        val_task = RestorationTask(val_tasks, has_bg=upd_config.modality == 'MRI', center=upd_config.center,
                                   p_no_aug=upd_config.p_no_aug, task_kwargs=h_config.task_kwargs,
                                   external_blending_dsets=ext_blending_dsets)

        logger.info('Loading main dataset...')
        train_dataset, val_dataset = DSET_DICT[upd_config.modality](
            upd_config, train=True)
        val_dataset.aug_fn = val_task

        logger.info('Main dataset loaded.')

        # DataLoaders creation:
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=h_config.train_batch_size,
            num_workers=h_config.dataloader_num_workers,
        )
        val_dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=h_config.validation_batch_size,
            num_workers=h_config.dataloader_num_workers,
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / h_config.gradient_accumulation_steps)
        if h_config.max_train_steps is None:
            h_config.max_train_steps = h_config.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            h_config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=h_config.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=h_config.max_train_steps * accelerator.num_processes,
        )

        # Prepare everything with our `accelerator`.
        optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            optimizer, train_dataloader, val_dataloader, lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / h_config.gradient_accumulation_steps)
        if overrode_max_train_steps:
            h_config.max_train_steps = h_config.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        h_config.num_train_epochs = math.ceil(
            h_config.max_train_steps / num_update_steps_per_epoch)

        # Train!
        total_batch_size = h_config.train_batch_size * \
            accelerator.num_processes * h_config.gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {h_config.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {h_config.train_batch_size}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(
            f"  Gradient Accumulation steps = {h_config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {h_config.max_train_steps}")
        logger.info(
            f"  Starting from step {global_step}, which is epoch {first_epoch}.")
        logger.info(f"  Early stopping state: {early_stopper.state_dict()}")

        progress_bar = tqdm(
            range(0, h_config.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            # disable=not accelerator.is_local_main_process,
            disable=not accelerator.is_main_process,
        )

        uncond_p = h_config.get("uncond_p", 0.1)

        def save_checkpoint(suffix=""):
            if accelerator.is_main_process:
                checkpoint_name = f"checkpoint-{global_step}{suffix}"
                save_path = os.path.join(h_config.output_dir, checkpoint_name)
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")

        for _ in range(first_epoch, h_config.num_train_epochs):
            train_loss = 0.0
            for val_batch in train_dataloader:
                with accelerator.accumulate(unet):
                    unet.train()
                    # Convert images to latent space
                    loss = compute_loss(
                        h_config, noise_scheduler, unet, uncond_p, val_batch)

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss).mean()
                    train_loss += avg_loss.item() / h_config.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            unet.parameters(), h_config.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    if h_config.use_ema:
                        ema_unet.step(unet.parameters())
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log(
                        {"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0

                    if global_step % h_config.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if h_config.checkpoints_total_limit is not None:
                                # Don't count the best checkpoint in the total count
                                checkpoints = [c for c in get_checkpoint_paths(h_config.output_dir) if "best" not in c.name]

                                # before we save the new checkpoint, we need to have at _most_
                                # `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= h_config.checkpoints_total_limit:
                                    num_to_remove = len(
                                        checkpoints) - h_config.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, "
                                        f"removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(
                                        f"removing checkpoints: {removing_checkpoints}")

                                    for removing_checkpoint in removing_checkpoints:
                                        shutil.rmtree(removing_checkpoint)

                            save_checkpoint()

                logs = {"step_loss": loss.detach().item(
                ), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                log_val_img = global_step % h_config.validation_img_steps == 0
                log_val_loss = global_step % h_config.validation_loss_steps == 0

                if global_step >= h_config.validation_monitoring_start and (log_val_img or log_val_loss):
                    logger.info("Running validation... ")

                    if h_config.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())

                    unet.eval()

                    if accelerator.is_main_process and log_val_img:
                        logger.info("Logging validation images...")
                        log_validation(unet, noise_scheduler, h_config, upd_config,
                                       accelerator, weight_dtype, val_dataset)

                    if log_val_loss:
                        logger.info("Computing validation loss...")
                        total_val_loss = 0.0
                        for val_batch in val_dataloader:
                            # Convert images to latent space
                            with torch.no_grad():
                                unet.eval()
                                # Set unconditional probability to 0 when prediction on validation images.
                                val_loss = compute_loss(
                                    h_config, noise_scheduler, unet, 0., val_batch)

                            # Gather the losses across all processes for logging (if we use distributed training).
                            total_val_loss += accelerator.gather_for_metrics(
                                val_loss).mean().item()

                        total_val_loss /= len(val_dataloader)

                        val_loss_ma, new_best, early_stop = early_stopper(total_val_loss, global_step)

                        logger.info(f"Validation loss: {total_val_loss:.5f}, val loss ma: {val_loss_ma:.5f}")
                        accelerator.log(
                            {"val_loss": total_val_loss, "val_loss_ma": val_loss_ma},
                            step=global_step)

                        if new_best:
                            if accelerator.is_main_process:
                                logger.info(
                                    f"New best validation loss MA: {val_loss_ma:.5f}")
                                # last_best_checkpoint = get_best_checkpoint_path()
                                # if last_best_checkpoint is not None:
                                #     shutil.rmtree(last_best_checkpoint)
                                save_checkpoint("-best")
                        if early_stop:
                            logger.info(
                                f"Early stopping triggered at step {global_step}")
                            break

                    unet.train()
                    if h_config.use_ema:
                        # Switch back to the original UNet parameters.
                        ema_unet.restore(unet.parameters())

                if global_step >= h_config.max_train_steps:
                    break

    else:

        assert h_config.resume_from_checkpoint, "Must specify a checkpoint to evaluate"

        test_dset_name = get_test_dset_name(upd_config)

        if upd_config.ssim_eval:
            eval_prefix = f"eval_{test_dset_name}"
        else:
            eval_prefix = f"eval_mae_{test_dset_name}"

        upd_config.eval_dir = path.with_name(f"{eval_prefix}_{path.stem}")
        logger.info(f"Saving evaluation results to {upd_config.eval_dir}")

        if h_config.use_ema:
            # Load EMA parameters for inference.
            ema_unet.store(unet.parameters())
            ema_unet.copy_to(unet.parameters())

        unet.eval()
        upd_config.step = global_step

        # Second return value is smaller test set which other methods use to monitor performance
        test_dset, _ = DSET_DICT[upd_config.modality](upd_config, train=False)

        if upd_config.figure_images:

            # Other modalities visible nicely on WANDB
            assert upd_config.modality == 'MRI', "Figure images only implemented for MRI"

            # Test indices chosen for slices which contain most anomalous pixels in each sample to show the models
            # ability to restore anomalies
            # BRATS INDICES:
            # - T2: 41023, 38184, 44245, 45921, 24469, 41003, 45901, 45210, 36555, 19347
            # - T1: 41055, 38217, 44277, 45951, 24485, 41035, 45931, 45241, 36573, 19346
            # ATLAS INDICES:
            # - 70352, 66406, 70372, 69965, 9374, 44880, 13352, 15812, 69945, 70920
            if upd_config.sequence == 't2':
                indices = [41023, 38184, 44245, 45921, 24469, 41003, 45901, 45210, 36555, 19347]
            elif upd_config.sequence == 't1':
                if upd_config.brats_t1:
                    indices = [41055, 38217, 44277, 45951, 24485, 41035, 45931, 45241, 36573, 19346]
                else:
                    indices = [70352, 66406, 70372, 69965, 9374, 44880, 13352, 15812, 69945, 70920]    
            else:
                raise ValueError(f"Unknown sequence {upd_config.sequence}")

            log_test_examples(unet, noise_scheduler, h_config, upd_config,
                           accelerator, weight_dtype, test_dset, indices)
        else:
            test_dataloader = DataLoader(
                test_dset,
                shuffle=False,
                batch_size=h_config.validation_batch_size,
                num_workers=h_config.dataloader_num_workers,
            )

            test_dataloader = accelerator.prepare(test_dataloader)
            evaluate(upd_config, test_dataloader,
                    gen_anomaly_inference_function(unet, noise_scheduler, h_config, upd_config, accelerator, weight_dtype))

    # Closes trackers, so put at very end of everything
    accelerator.end_training()

def get_test_dset_name(upd_config):
    test_dset_name = upd_config.modality
    if test_dset_name == 'MRI':
        test_dset_name += f'_{upd_config.sequence}'
        if upd_config.sequence == 't1' and not upd_config.brats_t1:
            test_dset_name += '_atlas'
        else:
            test_dset_name += '_brats'
    return test_dset_name


def compute_loss(h_config, noise_scheduler, unet, uncond_p, batch):
    corrupted, target = batch  #  device is already set by accelerator
    batch_size = target.shape[0]

    corrupted = torch.stack([corrupted[i] * 0. if random.random() < uncond_p else corrupted[i]
                             for i in range(batch_size)],
                            dim=0)

    noise = torch.randn_like(target)

    if h_config.noise_offset > 0:
        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
        noise += h_config.noise_offset * torch.randn(
            (target.shape[0], target.shape[1], 1, 1), device=target.device
        )

    if h_config.input_perturbation:
        new_noise = noise + h_config.input_perturbation * \
            torch.randn_like(noise)

        # Sample a random timestep for each image
    timesteps = torch.randint(
        0, int(noise_scheduler.config.num_train_timesteps),
        (batch_size,), device=target.device)
    timesteps = timesteps.long()

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    if h_config.input_perturbation:
        noisy_target = noise_scheduler.add_noise(
            target, new_noise, timesteps)
    else:
        noisy_target = noise_scheduler.add_noise(
            target, noise, timesteps)

    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(
            target, noise, timesteps)
    else:
        raise ValueError(
            f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        # Predict the noise residual and compute loss
        # input is noisy target + corrupted image
    model_input = torch.cat(
        [noisy_target, corrupted], dim=1)  # concat on channels

    # takes B x 2C x H x W and outputs B x C x H x W
    model_pred = unet(model_input, timesteps).sample

    if h_config.snr_gamma is None:
        loss = F.mse_loss(model_pred.float(),
                          target.float(), reduction="mean")
    else:
        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
        # This is discussed in Section 4.2 of the same paper.
        snr = compute_snr(noise_scheduler, timesteps)
        if noise_scheduler.config.prediction_type == "v_prediction":
            # Velocity objective requires that we add one to SNR values before we divide by them.
            snr = snr + 1
        mse_loss_weights = (
            torch.stack(
                [snr, h_config.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
        )

        loss = F.mse_loss(model_pred.float(),
                          target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))
                         ) * mse_loss_weights
        loss = loss.mean()

    return loss


if __name__ == "__main__":
    main()
