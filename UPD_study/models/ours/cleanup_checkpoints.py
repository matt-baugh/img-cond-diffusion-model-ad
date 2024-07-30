from pathlib import Path
import shutil

from UPD_study.models.ours.ours_trainer import parse_args, get_best_checkpoint_path, get_checkpoint_step


MIN_CHECKPOINT_INTERVAL = 5000


def main():
    _, h_config, _ = parse_args()

    root_output_dir = Path(h_config.output_dir).parent

    for output_folder in root_output_dir.iterdir():
        if not output_folder.is_dir():
            continue

        # We want to process folders which end with pattern `fold\d`
        if not output_folder.name[-1].isdigit() or output_folder.name[-5:-1] != 'fold':
            continue

        print('Processing', output_folder)

        evaluated_steps = [get_checkpoint_step(f) for f in output_folder.iterdir(
        ) if f.is_dir() and f.name.startswith('eval') and '-' in f.name]

        all_best_checkpoint_folders = get_best_checkpoint_path(output_folder, return_all=True)
        print('Best checkpoint folders', all_best_checkpoint_folders)

        if not isinstance(all_best_checkpoint_folders, list):
            continue

        # Don't delete the first and last checkpoint folders
        folders_to_prune = all_best_checkpoint_folders[1:-1]

        last_saved_step = get_checkpoint_step(all_best_checkpoint_folders[0])

        # Delete folders which are too close to the last saved checkpoint
        for f in folders_to_prune:

            new_checkpoint_step = get_checkpoint_step(f)

            # Don't delete anything we have recorded evaluations for
            if new_checkpoint_step in evaluated_steps:
                last_saved_step = new_checkpoint_step
                continue

            step_change = new_checkpoint_step - last_saved_step
            if step_change < MIN_CHECKPOINT_INTERVAL:
                # REMOVE DIR
                print(f"Removing {f}")
                shutil.rmtree(f)

            else:
                print(f"Keeping {f}")
                last_saved_step = new_checkpoint_step
        print()

if __name__ == "__main__":
    main()
