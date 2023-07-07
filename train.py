import argparse
import json
import os
from typing import Optional

from bort.fairseq_ext.cli_wrapper import fairseq_train
from bort.fairseq_ext.plugins.part_b_task import TASK_TARGET_PREDICTION
from bort.resources.logs import logger
import shutil


RANDOM_SEED = 8675309

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("lang_dir", type=str)
    parser.add_argument("save_dir", type=str)
    parser.add_argument("scratch_dir", type=str)
    parser.add_argument("--arch", default = "bart_base")
    parser.add_argument("--restore-file", default="bart.base")
    parser.add_argument("--batch-size", type=int, default=8000)
    parser.add_argument("--batch-size-valid", type=int, default=None)
    parser.add_argument("--beam", type=int, default=25)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--fp16", type=json.loads, default=True)
    parser.add_argument("--sample-break-mode", default="eos")  #  Default is 'complete_doc' which is tooooooo long
    parser.add_argument("--max-tokens", default=80000)
    parser.add_argument("--shorten-method", default="random_crop")
    parser.add_argument("--save-interval-updates", default=500)
    parser.add_argument("--no-epoch-checkpoints", default=True)
    # parser.add_argument("--keep-interval-updates", default=1)
    # parser.add_argument("--keep-best-checkpoints", default=1)
    parser.add_argument("--patience", default=20)
    parser.add_argument("--num-workers", default=1)
    #parser.add_argument("--skip-invalid-size-inputs-valid-test", default = False)
    parser.add_argument("--extra-name", type=str, default="")
    parser.add_argument("--fold-id", type=int, default=0)

    return parser.parse_args()

def main(
        *,
        lang_dir: str,
        save_dir: str,
        scratch_dir: str,
        batch_size: int,
        batch_size_valid: Optional[int],
        max_tokens: int,
        shorten_method: str,
        beam: int,
        arch: str,
        restore_file: str,
        sample_break_mode: str,
        save_interval_updates: int,
        # keep_interval_updates: int,
        # keep_best_checkpoints: int,
        no_epoch_checkpoints: bool,
        patience: int,
        num_workers: int,
        lr: float = 1e-3,
        fp16: bool = True,
        seed: int = RANDOM_SEED,
        #skip_invalid_size_inputs_valid_test: bool = True,
        extra_name: str = "",
        fold_id: int
):

    lang_dir = f"{lang_dir}/folds/fold_{fold_id}"
    save_dir = f"{save_dir}/folds/fold_{fold_id}"
    scratch_dir = f"{scratch_dir}/folds/fold_{fold_id}"
    if extra_name!= "":
        configuration = f'{restore_file.split("/")[-1]}-{extra_name}-fine-tuned'
    else:
        configuration = f'{restore_file.split("/")[-1]}-fine-tuned'
    batch_size_valid = batch_size_valid or batch_size
    checkpoint_dir = os.path.join(scratch_dir, "_working", configuration)
    tensorboard_logdir = os.path.join(save_dir, "tensorboard", configuration)
    final_model_dir = os.path.join(save_dir, configuration)

    if os.path.exists(f"{final_model_dir}/checkpoint_best.pt"):
        logger.warning(f"MODEL EXISTS! Skipping {final_model_dir}")
        return

    fairseq_train(
        lang_dir,
        seed=seed,
        arch=arch,
        bpe="gpt2",
        task=TASK_TARGET_PREDICTION,
        save_dir=checkpoint_dir,
        tensorboard_logdir=tensorboard_logdir,
        batch_size=batch_size,
        lr=lr,
        fp16=fp16,
        optimizer="adam",
        sample_break_mode=sample_break_mode,
        save_interval_updates=save_interval_updates,
        batch_size_valid=batch_size_valid,
        max_tokens=max_tokens,
        shorten_method=shorten_method,
        reset_optimizer=True,
        reset_dataloader=True,
        reset_meters=True,
        restore_file=restore_file,
        patience=patience,
        # keep_best_checkpoints=keep_best_checkpoints,
        # keep_interval_updates=keep_interval_updates,
        no_epoch_checkpoints=no_epoch_checkpoints,
        num_workers=num_workers,
        #skip_invalid_size_inputs_valid_test = skip_invalid_size_inputs_valid_test

    )

    move_model(checkpoint_dir, final_model_dir)

def move_model(checkpoint_dir, final_model_dir):
    os.makedirs(final_model_dir, exist_ok=True)
    for file in os.listdir(checkpoint_dir):
        checkpoint_file = os.path.join(checkpoint_dir, file)
        if file == "checkpoint_best.pt":
            final_model_file = os.path.join(final_model_dir, file)
            if os.path.isfile(final_model_file):
                os.rename(final_model_file, f"{final_model_file}.bak")
            shutil.move(checkpoint_file, final_model_file)
        else:
            os.remove(checkpoint_file)
    os.removedirs(checkpoint_dir)

if __name__ == '__main__':
    args = get_args()
    main(**vars(args))