import gzip
import inspect
import logging
import os
import re
import sys
from typing import Iterable

import torch


def fairseq_multiprocessing_bpe_encoder(
        inputs,
        outputs,
        workers=8,
):
    from fairseq import file_utils

    encoder_json = file_utils.cached_path('https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json')
    vocab_bpe = file_utils.cached_path('https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe')

    from fairseq.examples.roberta import multiprocessing_bpe_encoder
    args = ["--keep-empty"]
    orig_stdin = sys.stdin
    orig_stdout = sys.stdout
    try:
        with gzip.open(inputs, "rt") as input_f, gzip.open(outputs, "wt") as output_f:
            sys.stdin = input_f
            sys.stdout = output_f
            _run_fairseq(
                multiprocessing_bpe_encoder.main,
                *args,
                encoder_json=encoder_json,
                vocab_bpe=vocab_bpe,
                inputs="-",
                outputs="-",
                workers=workers,
            )
    finally:
        sys.stdin = orig_stdin
        sys.stdout = orig_stdout


def fairseq_preprocess_dir(
        out_dir,
        *,
        source_files,
        target_files=None,
        train="train",
        valid="dev",
        test="test",
        task="translation",
        only_source: bool = False,
        bpe: str = None,
        srcdict: str = None
):
    txt_dir, source, target = _parse_lang_files(
        source_files,
        target_files,
        sep_pattern=rf"\/(?:{train}|{valid}|{test})(\.|$)"
    )
    fairseq_preprocess(
        source_lang=source,
        target_lang=target,
        padding_factor=1,  # I don't like the look of `madeupword001` etc in the dictionary
        trainpref=os.path.join(txt_dir, train),
        validpref=os.path.join(txt_dir, train),
        testpref=os.path.join(txt_dir, train),
        tokenizer="space",
        destdir=out_dir,
        task=task,
        only_source=only_source,
        bpe=bpe,
        srcdict=srcdict,
    )


def fairseq_preprocess(
        source_lang=None,
        target_lang=None,
        *,
        destdir=None,
        trainpref=None,
        validpref=None,
        testpref=None,
        task="translation",
        tokenizer="space",
        only_source: bool = False,
        bpe: str = None,
        srcdict: str = None,
        padding_factor=1,
        workers=1,
        **kwargs,
):
    import fairseq_cli.train

    _run_fairseq(
        fairseq_cli.preprocess.cli_main,
        padding_factor=padding_factor,    # I don't like the look of `madeupword001` etc in the dictionary
        source_lang=source_lang,
        target_lang=target_lang,
        trainpref=trainpref,
        validpref=validpref,
        testpref=testpref,
        tokenizer=tokenizer,
        destdir=destdir,
        task=task,
        only_source=only_source,
        bpe=bpe,
        srcdict=srcdict,
        workers=workers,
        **kwargs,
    )


def fairseq_train_translate(
        source_dict,
        target_dict,
        *,
        save_dir,
        tensorboard_logdir,
        seed: int,
        train_weight: str = None,
        encoder_embed_path: str = None,
        encoder_embed_dim: int = None,
        encoder_hidden_size: int = 512,
        encoder_ffn_embed_dim: int = 2048,
        encoder_attention_heads: int = 8,
        decoder_embed_path: str = None,
        decoder_embed_dim: int = None,
        decoder_hidden_size: int = 512,
        decoder_out_embed_dim: int = None,
        encoder_layers=None,
        decoder_layers=None,
        decoder_attention_heads: int = 8,
        lr=1e-4,
        arch="lstm",
        dropout=0.1,
        fp16=True,
        optimizer="adam",
        # adam_betas=(0.9, 0.999),
        batch_size="1024",
        max_epoch=None,
        patience=None,
        no_epoch_checkpoints=True,
        encoder_bidirectional=True,
        share_decoder_input_output_embed=True,
        clip_norm=0.0,
):
    # strip the .txt to parse source/target
    source_dict = source_dict[:-4]
    target_dict = target_dict[:-4]
    lang_dir, source, target = _parse_lang_files([source_dict], [target_dict], sep_pattern=rf"\/dict\.")

    more_args = []
    more_kwargs = {}

    if no_epoch_checkpoints:
        more_args.append("--no-epoch-checkpoints")
    if train_weight:
        more_kwargs["train_weight"] = train_weight

    if arch in ("lstm",):
        if encoder_bidirectional:
            more_args.append("--encoder-bidirectional")
        if share_decoder_input_output_embed:
            more_args.append("--share-decoder-input-output-embed")
        more_kwargs["encoder_hidden_size"] = int(encoder_hidden_size)
        more_kwargs["decoder_hidden_size"] = int(decoder_hidden_size)
        more_kwargs["encoder_layers"] = int(encoder_layers or 1)
        more_kwargs["decoder_layers"] = int(decoder_layers or 1)
        more_kwargs["decoder_out_embed_dim"] = int(decoder_out_embed_dim)

    if arch in ("fconv",):
        default_conv = "[(128, 3)] + [(64, 3)]"
        more_kwargs["encoder_layers"] = encoder_layers or default_conv
        more_kwargs["decoder_layers"] = decoder_layers or default_conv
        more_kwargs["decoder_out_embed_dim"] = int(decoder_out_embed_dim)

    if arch in ("transformer",):
        more_kwargs["encoder_ffn_embed_dim"] = encoder_ffn_embed_dim
        more_kwargs["encoder_attention_heads"] = encoder_attention_heads
        more_kwargs["decoder_attention_heads"] = decoder_attention_heads

    if encoder_embed_path is not None:
        more_kwargs["encoder_embed_path"] = encoder_embed_path
    if decoder_embed_path is not None:
        more_kwargs["decoder_embed_path"] = decoder_embed_path

    fairseq_train(
        lang_dir,
        *more_args,
        seed=seed,
        source_lang=source,
        target_lang=target,
        save_dir=save_dir,
        tensorboard_logdir=tensorboard_logdir,
        lr=lr,
        dropout=float(dropout),
        arch=arch,
        optimizer=optimizer,
        task="translation_weighted",
        fp16=fp16,
        # adam_betas=f"'{adam_betas}'",
        encoder_embed_dim=int(encoder_embed_dim),
        decoder_embed_dim=int(decoder_embed_dim),
        batch_size=int(batch_size),
        max_epoch=int(max_epoch),
        patience=int(patience),
        clip_norm=float(clip_norm),
        # user_dir=user_dir,
        **more_kwargs
    )

def fairseq_train(
        *args,
        save_dir,
        tensorboard_logdir,
        seed: int,
        task: str,
        fp16=True,
        lr: float,
        **kwargs
):
    import fairseq_cli.train

    if fp16 and torch.cuda.is_available():
        args = ("--fp16", *args)
    _run_fairseq(
        fairseq_cli.train.cli_main,
        *args,
        save_dir=save_dir,
        tensorboard_logdir=tensorboard_logdir,
        lr=float(lr),
        seed=int(seed),
        task=str(task),
        **kwargs
    )


def _run_fairseq(main_func, *args, **kwargs):
    # !/Users/galer/.pyenv/versions/3.9.9/envs/phoneme-to-grapheme/bin/python3.9
    # -*- coding: utf-8 -*-
    import re
    import sys
    from fairseq_cli.train import cli_main
    if __name__ == '__main__':
        sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
        sys.exit(cli_main())
    old_argv = sys.argv
    try:
        cmd_args = [str(a) for a in args]
        for arg, value in kwargs.items():
            if value in (None, ""):
                continue
            if isinstance(value, bool):
                if value:
                    value = []
                else:
                    continue
            if isinstance(value, str) or not isinstance(value, Iterable):
                value = [value]
            cmd_args.extend([f"--{arg.replace('_', '-')}", *(str(v) for v in value)])
        sys.argv = [inspect.getmodule(main_func).__file__, *cmd_args]
        logging.info(f"Running command:\n\n{_print_command(sys.argv)}\n")
        return main_func()
        # try:
        #
        #     return subprocess.run([*cmd, *cmd_args], check=True)
        # except subprocess.CalledProcessError as e:
        #     raise RuntimeError(print_cmd) from e
    finally:
        sys.argv = old_argv


def _print_command(args):
    print_cmd = " \\\n    ".join(args)
    print_cmd = re.sub(r"^(\s*--.*?) \\\n\s{4}(?!--\w)", r"\1 ", print_cmd, flags=re.MULTILINE)
    return f"python {print_cmd}"


def _parse_lang_files(source_files, target_files, *, sep_pattern):
    def parse_filenames(files):
        prefix = ""
        for chars in zip(*files):
            if len(set(chars)) != 1:
                break
            prefix = f"{prefix}{chars[0]}"

        suffix = ""
        for chars in zip(*(reversed(f) for f in files)):
            if len(set(chars)) != 1:
                break
            suffix = f"{chars[0]}{suffix}"

        assert len(files)
        return prefix, suffix

    source_dir, source_lang = parse_filenames(source_files)
    target_lang = None
    if target_files:
        target_dir, target_lang = parse_filenames(target_files)
        assert source_dir == target_dir, f"Expect source and target files to be in same directory"
    return source_dir, source_lang, target_lang
