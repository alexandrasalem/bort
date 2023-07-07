import sys
from dataclasses import dataclass, field
from functools import lru_cache

import regex

from bort.resources import consts
from bort.resources.consts import LettersStrategy


BULLET = "\xb7"
SUPERFLUOUS_BULLET_PATTERN = regex.compile(rf"{BULLET}(?![a-z])|(?<![a-z\s]){BULLET}(?=[a-z])", flags=regex.IGNORECASE)


@dataclass
class Configuration:
    letters_strategy: LettersStrategy = field(default_factory=lambda: _cfg_getter("letters_strategy"))
    vocab_size: int = field(default_factory=lambda: _get_vocab_size())


@dataclass
class Dependencies:
    config: Configuration = field(default_factory=lambda: Configuration)
    fairseq_bpe: "fairseq.hub_utils.BPEHubInterface" = field(default_factory=lambda: _load_bpe_from_hub())


@lru_cache()
def _load_bpe_from_hub() -> "fairseq.hub_utils.BPEHubInterface":
    def add_unk_token(bpe: "fairseq.data.encoders.gpt2_bpe_utils.Encoder"):
        new_pattern = rf" ?{regex.escape(consts.UNK_TOKEN)}|{bpe.pat.pattern}"
        bpe.pat = regex.compile(new_pattern, flags=bpe.pat.flags)
        mask_id = bpe.encoder.get(consts.UNK_TOKEN, len(bpe.encoder))
        bpe.encoder[consts.UNK_TOKEN] = mask_id
        bpe.decoder[mask_id] = f"Ġ{consts.UNK_TOKEN}"
        bpe.cache[consts.UNK_TOKEN] = consts.UNK_TOKEN  # HACK: Disallows split of mask token
        bpe.cache[f"Ġ{consts.UNK_TOKEN}"] = consts.UNK_TOKEN  # HACK: Disallows split of mask token
        return bpe

    import torch
    bpe_from_hub = torch.hub.load('pytorch/fairseq', 'bpe', 'gpt2')
    bpe_from_hub.bpe.bpe.pat = _gpt2_word_tokenization_pattern() # OVERRIDE WITH OUR WORD TOKENIZATION PATTERN
    return add_unk_token(bpe_from_hub.bpe.bpe)


@lru_cache()
def _gpt2_word_tokenization_pattern():
    FAIRSEQ_GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    apostrophes_hyphens_underscores_within_letters = r" ?\p{L}+(?:['\-_]\p{L}+)+"
    special_bulleted_separators = rf" ?\p{{L}}*{BULLET}\p{{L}}(?:{BULLET}?\p{{L}})*"
    pattern = regex.compile("|".join([
        apostrophes_hyphens_underscores_within_letters,
        special_bulleted_separators,
        FAIRSEQ_GPT2_PATTERN,
    ]), flags=regex.IGNORECASE)
    return pattern

@lru_cache()
def _cfg(input_args=None):
    def _modify_parser(parser):
        parser.set_defaults(arch="bart_base", data="unk")
        return parser

    import fairseq.options
    from fairseq.dataclass.utils import convert_namespace_to_omegaconf
    from phatp.fairseq_ext.plugins.part_a_task import TASK_DENOISING_P2G
    parser = fairseq.options.get_training_parser(TASK_DENOISING_P2G)
    input_args = input_args or sys.argv[1:] or ["UNKNOWN_DATA_SOURCE!"]
    args = fairseq.options.parse_args_and_arch(parser, input_args=input_args, modify_parser=_modify_parser)
    cfg = convert_namespace_to_omegaconf(args)
    return cfg.task


def _cfg_getter(attr):
    def inner():
        cfg = _cfg()
        if not hasattr(cfg, attr):
            raise AttributeError(attr)
        return getattr(cfg, attr)
    return inner


def _get_vocab_size():
    return len(_load_bpe_from_hub().encoder)
