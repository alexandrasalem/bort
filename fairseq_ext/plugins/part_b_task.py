import logging
import random
from argparse import ArgumentError
from contextlib import contextmanager
from typing import Optional

import fairseq
import fairseq.models.bart
from fairseq.tasks import register_task
from fairseq.tasks.denoising import DenoisingTask
from tqdm import tqdm

from bort.fairseq_ext.plugins.models import load_model_dictionary
from bort.fairseq_ext.plugins.transforms import TransformCounter
from bort.fairseq_ext.plugins.models import BORTModel

logger = logging.getLogger(__name__)

TASK_TARGET_PREDICTION = "phoneme_aware_target_prediction"


@register_task(TASK_TARGET_PREDICTION)
class PhonemeAwareTargetPrediction(DenoisingTask):
    def __init__(
            self,
            args,
            dictionary,
            running_stats: Optional[TransformCounter] = None
    ):
        super().__init__(args, dictionary)
        self.running_stats = running_stats or TransformCounter.empty()

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        loss, sample_size, logging_output = super().train_step(sample, model, criterion, optimizer, update_num, ignore_grad)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        return super().valid_step(sample, model, criterion)


    def build_model(self, cfg, from_checkpoint=True):
        assert from_checkpoint
        model_name_or_path = cfg.restore_file
        #going to need to change for providing variants
        model = BORTModel.from_pretrained(model_name_or_path)
        #model = load_model(model_name_or_path)
        assert model.cfg.model.max_source_positions == cfg.max_source_positions, f"Expected --max-source-positions={model.cfg.max_source_positions}"
        assert model.cfg.model.max_target_positions == cfg.max_target_positions, f"Expected --max-target-positions={model.cfg.max_target_positions}"
        return model

    @classmethod
    def add_args(cls, parser):
        with _fix_what_might_be_a_bug_with_fairseq_add_args(parser):
            super().add_args(parser)


    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task."""
        random.seed(args.seed)
        paths = fairseq.utils.split_paths(args.data)
        assert len(paths) > 0
        #model = BORTModel.from_pretrained(args.restore_file)
        dictionary = load_model_dictionary(paths[0], args.restore_file)
        if not hasattr(args, "shuffle_instance"):
            args.shuffle_instance = False
        return cls(args, dictionary=dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        from bort.fairseq_ext.plugins.part_b_dataset import BracketsSpacesPhonemesDataset

        self.datasets[split] = BracketsSpacesPhonemesDataset.build(
            self.args,
            split=split,
            dictionary=self.dictionary,
            epoch=epoch,
            combine=combine,
            **kwargs
        )
        verifying = (self.datasets[split][i] for i in range(len(self.datasets[split])))
        list(tqdm(verifying, total=len(self.datasets[split]), desc=f"Verifying {split} dataset"))


@contextmanager
def _fix_what_might_be_a_bug_with_fairseq_add_args(parser):
    # Just a fix to what might be a bug in fairseq_ext.
    orig_add_argument = parser.add_argument

    def add_argument_shim(*args, **kwargs):
        try:
            return orig_add_argument(*args, **kwargs)
        except ArgumentError as e:
            if e.argument_name in ("--max-source-positions", "--max-target-positions",):
                return
            raise

    parser.add_argument = add_argument_shim
    yield
    parser.add_argument = orig_add_argument
