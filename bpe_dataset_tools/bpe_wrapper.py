from dataclasses import dataclass, field
from functools import lru_cache, cache
from itertools import chain
from typing import Union, Iterable, Tuple

import torch

from bort.resources.consts import LettersStrategy, UNK_TOKEN
from bort.resources.dependencies import _load_bpe_from_hub, Configuration, BULLET, SUPERFLUOUS_BULLET_PATTERN


class BpeToken(int):
    """
    A subword token encoded as BPE
    """
    def __str__(self):
        return bpe_to_text(self)


class BpeSequence(Tuple[BpeToken, ...]):
    """
    A sequence of BpeTokens, i.e. some encoded text
    """

    def to_words(self):
        return bpe_to_words(self)

    def to_model_encoding(self, dictionary: "fairseq.data.Dictionary"):
        return bpe_to_model_encoding(self, dictionary)

    def __str__(self):
        return bpe_to_text(self)

    @property
    def is_word(self):
        return not len(text_to_bpe_words(str(self))) == 1

    @classmethod
    def concatenate(cls, items: Iterable["BpeSequence"]):
        return BpeSequence(chain(*items))

    @classmethod
    def from_text(cls, text: str):
        return text_to_bpe(text)

    @classmethod
    def from_bpe_encoding(cls, ids: Iterable[int]):
        return cls(BpeToken(idx) for idx in ids)

    @classmethod
    def from_model_encoding(cls, ids: Iterable[int], model_dictionary: "fairseq.data.Dictionary"):
        return cls.from_bpe_encoding(model_dictionary[id] for id in ids[1:-1])


class BpeSequenceCollection(Tuple[BpeSequence, ...]):
    """
    Data structure for when a BpeSequence has been split into words
    """
    def __str__(self):
        return "".join(str(word) for word in self)


def bpe_raw_encode(text: str) -> BpeSequence:
    def encode(text):
        tokenizer = _load_bpe_from_hub()
        for match in tokenizer.pat.finditer(text):
            yield from _bpe_raw_encode_single_word(match[0])

    return BpeSequence(encode(text))


@lru_cache(maxsize=999999)
def _bpe_raw_encode_single_word(word_string: str):
    tokenizer = _load_bpe_from_hub()
    token = "".join(tokenizer.byte_encoder[b] for b in word_string.encode("utf-8"))
    return BpeSequence(
        BpeToken(tokenizer.encoder[bpe_token])
        for bpe_token in tokenizer.bpe(token).split(" ")
    )


def tokens_would_merge(a: Union[str, BpeSequence], b: Union[str, BpeSequence]) -> bool:
    if None in (a, b):
        return False
    if isinstance(a, str):
        a = bpe_raw_encode(a)
    if isinstance(b, str):
        b = bpe_raw_encode(b)
    compare = tuple(BpeToken(idx) for idx in bpe_raw_encode(f"{a}{b}"))
    return compare != (*a, *b)


def bpe_to_words(sequence: BpeSequence) -> BpeSequenceCollection:
    return text_to_bpe_words(str(sequence))


def text_to_bpe(text: str, letters_strategy: LettersStrategy = LettersStrategy.COMPACT_BULLETS) -> BpeSequence:
    encoded = text_to_bpe_words(text)
    return _apply_strategy(encoded, letters_strategy)


@lru_cache()
def text_to_bpe_words(text: str):
    tokenizer =_load_bpe_from_hub()
    words = (_bpe_raw_encode_single_word(m[0]) for m in tokenizer.pat.finditer(text))
    return BpeSequenceCollection(words)


def _apply_strategy(words: BpeSequenceCollection, strategy: LettersStrategy):
    result = []
    for word in words:
        if BULLET not in str(word):
            result.extend(word)
            continue

        if strategy == LettersStrategy.BULLETS:
            result.extend(word)
        elif strategy == LettersStrategy.COMPACT_BULLETS:
            compact_text = SUPERFLUOUS_BULLET_PATTERN.sub("", str(word))
            result.extend(bpe_raw_encode(compact_text))
        else:
            raise NotImplementedError()
    return BpeSequence(result)


def bpe_to_model_encoding(sequence: "BpeSequence", dictionary: "fairseq.data.Dictionary") -> torch.tensor:
    rewrites = {_unk_id(): dictionary.unk_index}
    bpe_ids = (rewrites.get(bpe, str(int(bpe))) for bpe in sequence)
    model_encoded = [dictionary.bos_index, *(dictionary.indices[bpe] for bpe in bpe_ids), dictionary.eos_index]
    return torch.tensor(model_encoded)


@cache
def _unk_id():
    return text_to_bpe(UNK_TOKEN)


def bpe_to_text(encoded: Union[Iterable[int], int, str]) -> str:
    if isinstance(encoded, str):
        encoded = [int(id) for id in encoded.split()]
    if isinstance(encoded, int):
        encoded = [encoded]
    return _load_bpe_from_hub().decode(int(id) for id in encoded)


@lru_cache(maxsize=999999999)
def _bpe_to_bytes_cached(ints) -> bytearray:
    if len(ints) <= 3:
        return _load_bpe_from_hub().decode(ints).encode("utf8")
    else:
        return _bpe_to_bytes_cached(ints[:3]) + _bpe_to_bytes_cached(ints[3:])


@dataclass
class BpeWrapper:
    """
    An interface to Fairseq's BPE tokenizer/encoder/decoder
    """

    args: Configuration = field(default_factory=Configuration)

    @property
    def vocab_size(self):
        return self.args.vocab_size

    def encode(self, text: str) -> BpeSequence:
        return text_to_bpe(text, letters_strategy=self.args.letters_strategy)

    def decode(self, encoded: Union[Iterable[int], int, str]):
        return bpe_to_text(encoded)
