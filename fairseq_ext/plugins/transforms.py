import dataclasses
import os
import random
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable

import regex
import torch
import yaml
from typing_extensions import override

from bort.bpe_dataset_tools.bpe_wrapper import BpeSequenceCollection, text_to_bpe
from bort.resources.dependencies import BULLET
from bort.resources.pronounce import PronunciationDictionary, tokenize_ipa
from bort.resources.yaml_serialization import yaml_dumper_with_our_dataclasses


# _ESCAPE_MATCH_PATTERN = regex.compile(r"((?<=^|[A-Z])(?=[A-Z]))", flags=re.IGNORECASE)


@dataclass
class WordNoiser:
    """
    Randomly swaps, drops, and inserts letters/symbols in a word.
    """

    p_swap: float
    p_drop: float
    p_insert: float
    alphabet: List[str]

    def __call__(self, tokens: Iterable[str]) -> Tuple[str]:
        out_tokens = []
        if random.random() < self.p_insert:
            out_tokens.append(random.choice(self.alphabet))

        for token in tokens:
            if random.random() < self.p_swap:
                token = random.choice(self.alphabet)

            if random.random() > self.p_drop:
                out_tokens.append(token)

            if random.random() < self.p_insert:
                out_tokens.append(random.choice(self.alphabet))

        return tuple(out_tokens)

    @classmethod
    def build(cls, args, use_ipa: bool):
        ipa_letters = 'aɪ aʊ b d e f h i j k l m n o p s t u v w z æ ð ŋ ɑ ɔ ɔɪ ə ɛ ɝ ɡ ɪ ɹ ɾ ʃ ʊ ʌ ʒ ʔ ʤ ʧ θ'
        alphabet = tokenize_ipa(ipa_letters) if use_ipa else list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        return cls(args.p_swap, args.p_drop, args.p_insert, alphabet)

    @classmethod
    def noop(self):
        return WordNoiser(0.0, 0.0, 0.0, [])


class _StringTransform:
    def transform(self, word_text_orig: str) -> str:
        raise NotImplementedError()

    # def _escape_so_it_tokenizes_letters(self, s):
    #     # To avoid BPE merging IPA characters that happen to be Latin letters
    #     return regex.sub(_ESCAPE_MATCH_PATTERN, rf"{_ESCAPE_CHARACTER}\1", s)


@dataclass
class WordPronouncer(_StringTransform):

    dictionary: PronunciationDictionary
    noiser: WordNoiser = WordNoiser.noop()
    phoneme_separator: str = BULLET

    @override
    def transform(self, word_text_orig: str) -> str:
        pronunciations = self.dictionary.get(word_text_orig)
        if not len(pronunciations):
            raise KeyError(f"Could not pronounce word `{word_text_orig}`")
        random_pronuncation = random.choice(list(pronunciations))
        phonemes = tokenize_ipa(random_pronuncation)
        noisy_phonemes = self.noiser(phonemes)
        source_text_transformed = self.phoneme_separator.join(("", *noisy_phonemes))
        # source_text_transformed = self._escape_so_it_tokenizes_letters(source_text_transformed)
        return source_text_transformed


@dataclass
class WordSpeller(_StringTransform):
    """
    Transforms a word into its (uppercase) spelling, e.g. "cereal" -> "C E R E A L"
    """

    noiser: WordNoiser = WordNoiser.noop()
    letter_separator: str = BULLET

    WORD_PATTERN = regex.compile(r"^\s*\p{L}+$")

    @override
    def transform(self, word_text_orig: str) -> str:
        letters = word_text_orig.strip().upper()
        noisy_letters = self.noiser(letters)
        source_text_transformed = BULLET + BULLET.join(noisy_letters)
        return source_text_transformed


@dataclass
class _BpeSentenceTransformer:
    """
    Takes a whole sequence and randomly applies a transform according to parameters
    """

    string_transform: _StringTransform
    use_mask_token: bool = False

    def __call__(
            self,
            source_words: BpeSequenceCollection,
            target_words: BpeSequenceCollection,
            mask: List[bool],
            num_to_transform: int,
            must_idx: Optional[int]
    ) -> Tuple[BpeSequenceCollection, BpeSequenceCollection, List[bool]]:
        assert len(source_words) == len(target_words)
        transform_candidates = [i for i, m in enumerate(mask) if m and i != must_idx]

        random.shuffle(transform_candidates)
        new_source_words = list(source_words)
        new_target_words = list(target_words)
        were_transformed = []

        if must_idx:
            transform_candidates.insert(0, must_idx)

        for idx in transform_candidates:
            if len(were_transformed) >= num_to_transform:
                break  # We're done.

            source_word = source_words[idx]
            target_word = target_words[idx]

            # strip preceding space
            source_str = str(source_word).lstrip()
            maybe_replace_space = " {}" if str(source_word).startswith(" ") else "{}"

            new_source_text = self.string_transform.transform(source_str)

            new_source_word = text_to_bpe(maybe_replace_space.format(new_source_text))
            new_target_word = target_word  # We might need formatting?

            new_source_words[idx] = new_source_word
            new_target_words[idx] = new_target_word
            were_transformed.append(idx)

        tr_mask = torch.zeros(len(source_words), dtype=torch.bool).scatter_(0, torch.tensor(were_transformed), 1)
        return BpeSequenceCollection(new_source_words), BpeSequenceCollection(new_target_words), tr_mask


class PronounceWordsBpeTransform(_BpeSentenceTransformer):
    @classmethod
    def build(cls, args, pronunciations):
        noiser = WordNoiser.build(args, use_ipa=True)
        pronouncer = WordPronouncer(dictionary=pronunciations, noiser=noiser)
        return cls(
            string_transform=pronouncer,
            use_mask_token=args.use_mask_token
        )


class SpellWordsBpeTransform(_BpeSentenceTransformer):
    @classmethod
    def build(cls, args):
        noiser = WordNoiser.build(args, use_ipa=False)
        speller = WordSpeller(noiser=noiser)
        return cls(
            string_transform=speller,
            use_mask_token=args.use_mask_token
        )


@dataclass(unsafe_hash=True)
class TransformCounter:
    name: str
    changed: Counter
    unchanged: Counter
    n_combined: int = 0

    def to_yaml(self, filename):
        dumper = yaml_dumper_with_our_dataclasses(data_class_types=[TransformCounter], mapping_types=[Counter])
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        with open(filename, "w", encoding="utf8") as f:
            f.write(yaml.dump(self, Dumper=dumper, sort_keys=False, allow_unicode=True))

    @classmethod
    def build(cls, name, source_words, transformed_source):
        assert len(source_words) == len(transformed_source), "Number of words changed during transform!"
        unchanged = Counter()
        changed = Counter()
        for before_word, after_word in zip(source_words, transformed_source):
            before_text = cls._normalize_word(before_word)
            if before_word == after_word:
                unchanged.update([before_text])
            else:
                changed.update([before_text])
        return cls(name, changed=changed, unchanged=unchanged)

    @classmethod
    def combine(cls, reports: Iterable["TransformCounter"]) -> "TransformCounter":
        if not reports:
            return TransformCounter.empty()
        changed = sum(r.changed for r in reports)
        unchanged = sum(r.unchanged for r in reports)
        n_combined = sum(r.n_combined for r in reports)
        result = dataclasses.replace(reports[-1], changed=changed, unchanged=unchanged, n_combined=n_combined)
        return result

    @classmethod
    def empty(cls):
        return TransformCounter(name="", changed=Counter(), unchanged=Counter())

    @staticmethod
    def _normalize_word(word: str):
        return str(word).strip().lower()
