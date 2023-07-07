import os
import random
import sys
from functools import cache
from functools import lru_cache
from tempfile import TemporaryDirectory
from typing import Dict, Tuple, List
from typing import Iterable

import datasets
import torch.utils.data
import torch.utils.data
from datasets import Features

from bort.resources.consts import PROD_DELIM_LEFT, PROD_DELIM_RIGHT
from bort.bpe_dataset_tools.bpe_wrapper import BpeSequence, text_to_bpe
from bort.bpe_dataset_tools.bpe_wrapper import BpeSequenceCollection, text_to_bpe_words
from bort.bpe_dataset_tools.datasets_mapping_helper import DatasetMapper
from bort.resources.dependencies import Configuration
from bort.fairseq_ext.plugins.transforms import WordSpeller
from bort.resources.logs import logger

DATASETS_WIKIPEDIA_PATH = "wikipedia"
DATASETS_WIKIPEDIA_VERSION = "20220301.en"


class BpeDataset(torch.utils.data.Dataset):
    """
    Dataset of BPE-encoded texts
    """

    def __init__(self, inner_dataset, *, trim_to_length=None):
        super().__init__()
        self.dataset = inner_dataset
        self.sizes = [min(trim_to_length or sys.maxsize, size) for size in inner_dataset["bpe_size"]]
        self.trim_to_length = trim_to_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> dict:
        try:
            row = self.dataset[int(index)]
            sequence = BpeSequenceCollection(BpeSequence.from_bpe_encoding(w) for w in row["bpe_words"])
            row["bpe_words"] = BpeSequenceCollection(sequence)
            row["must_idx"] = get_median_occurrence_of_word(row["bpe_words"], row["must_pronounce"])  # None if None.
            return row
        except Exception as e:
            print(f"{type(e).__name__}: {e}")
            raise


class PartBBpeDataset(torch.utils.data.Dataset):
    """
    Dataset of BPE-encoded texts
    """

    def __init__(self, inner_dataset, *, trim_to_length=None):
        super().__init__()
        self.dataset = inner_dataset
        self.sizes = [min(trim_to_length or sys.maxsize, size) for size in inner_dataset["bpe_size"]]
        self.trim_to_length = trim_to_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> dict:
        try:
            row = self.dataset[int(index)]
            bpe = BpeSequence.from_bpe_encoding(row["bpe"])
            if self.trim_to_length:
                bpe = self.trim_custom(bpe, self.trim_to_length)
            return bpe
        except Exception as e:
            print(f"{type(e).__name__}: {e}")
            raise

    @staticmethod
    @cache
    def _left_delim():
        return text_to_bpe(PROD_DELIM_LEFT)[0]

    @staticmethod
    @cache
    def _right_delim():
        return text_to_bpe(PROD_DELIM_RIGHT)[0]

    @classmethod
    def trim_custom(cls, sequence: BpeSequence, max_length) -> BpeSequence:
        max_length = len(sequence) if max_length is None else max_length
        if len(sequence) <= max_length:
            return sequence
        try:
            left_idx = sequence.index(cls._left_delim())
            right_idx = sequence.index(cls._right_delim(), left_idx)
        except ValueError:
            raise ValueError(f'Could not find either left {[str(d) for d in cls._LEFT_DELIM]} '
                             f'or right {[str(d) for d in cls._LEFT_DELIM]} delimiters in sequence')

        if len(sequence) > max_length:
            new_sequence = sequence[left_idx:right_idx + 1]
            while len(new_sequence) < max_length:
                if left_idx > 0:
                    left_idx -= 1
                    new_sequence = (sequence[left_idx],) + new_sequence
                    if len(new_sequence) == max_length:
                        break
                if right_idx < len(sequence) - 1:
                    right_idx += 1
                    new_sequence = new_sequence + (sequence[right_idx],)
        result = BpeSequence(new_sequence)
        assert len(result) == max_length
        return result

    @classmethod
    def from_texts(cls, texts: Iterable[str], *args, **kwargs):
        import datasets
        bpes = (text_to_bpe(t) for t in texts)
        texts_dataloader = datasets.Dataset.from_list([
            {"bpe": bpe, "bpe_size": len(bpe)}
            for bpe in bpes
        ])
        return cls(texts_dataloader, *args, **kwargs)


def wikipedia_data_as_bpe(limit=None, num_proc=32):
    source_dataset = datasets.load_dataset(DATASETS_WIKIPEDIA_PATH, DATASETS_WIKIPEDIA_VERSION, split="train")
    if limit:
        source_dataset = datasets.Dataset(source_dataset.data[:limit])
    config = Configuration()
    dtype = uint_dtype(config.vocab_size)
    return source_dataset.map(
        _map_to_bpe,
        batched=False,
        num_proc=num_proc,
        remove_columns=["text", "url"],
        features=Features({
            'id': datasets.Value("string"),
            'title': datasets.Value("string"),
            'bpe_words': datasets.Sequence(datasets.Sequence(datasets.Value(dtype=dtype))),
            'bpe_size': datasets.Value("uint32"),
        })
    )


def _map_to_bpe(row):
    try:
        bpe_words = text_to_bpe_words(row["text"])
        return {
            "bpe_words": bpe_words,
            "bpe_size": sum(len(w) for w in bpe_words)
        }
    except Exception as e:
        os.makedirs("./fail", exist_ok=True)
        error_file = f"./fail/{row['id']}.txt"
        with open(error_file, "w") as f:
            f.write(row["text"])
        raise ValueError(f"Couldn't map to BPE: \"{row['title']}\". Wrote text to {error_file}") from e


def prepare_experiment_dataset(bpe_dataset: datasets.Dataset, *, random_seed, limit, num_proc, out_dir):
    from phatp.datasets.wikisplits import split_dataset, get_aphasiabank_holdout_words, build_cmudict_splits

    built_dataset_dir = os.path.join(out_dir, f"wikipedia_splits_bpe-{limit}" if limit else f"wikipedia_splits_bpe")
    if os.path.exists(built_dataset_dir):
        return datasets.load_from_disk(built_dataset_dir)

    with TemporaryDirectory() as temp_dir:
        splits = build_cmudict_splits(out_dir, random_seed, n_workers=num_proc, limit=limit)
        split_datasets = split_dataset(bpe_dataset, splits)
        for split_name, split_data in split_datasets.items():
            temp_filename = f"{temp_dir}/{split_name}"
            split_data.save_to_disk(temp_filename, max_shard_size="1GB")
            split_datasets[split_name] = datasets.load_from_disk(temp_filename)
            assert split_datasets[split_name]["id"] == split_data["id"]

        mapped_datasets = {}
        for split_name, split_data in split_datasets.items():
            logger.info(f"Preparing BPE/Pronunciation dataset: {split_name}")
            lookup = splits[split_name].must_pronounce
            vocab = list(splits[split_name].pronunciations)

            forbidden_words = set() if split_name == "test" else set(get_aphasiabank_holdout_words())
            for sn, sd in splits.items():
                if sn != split_name:
                    forbidden_words.update(set(sd.word_to_article_id))

            map_to_bpe = BpeMapper(num_proc=num_proc, vocab_words=vocab, must_pronounce=lookup, forbidden_words=forbidden_words)

            cache_dir = os.path.join(temp_dir, split_name)
            os.makedirs(cache_dir, exist_ok=True)
            mapped_datasets[split_name] = split_data.map(
                map_to_bpe,
                batched=False,
                num_proc=num_proc,
                cache_file_name=os.path.join(cache_dir, f"prepared_data.arrow"),
                features=Features({
                    **bpe_dataset.features,
                    'spellable': datasets.Sequence(datasets.Value("bool")),
                    'pronounceable': datasets.Sequence(datasets.Value("bool")),
                    'must_pronounce': datasets.Sequence(datasets.Value("string"))
                })
            )
            assert mapped_datasets[split_name]["id"] == split_data["id"]

        mapped_datasets = datasets.DatasetDict(mapped_datasets)

        staging_dir = f"{built_dataset_dir}.temp"
        mapped_datasets.save_to_disk(staging_dir, max_shard_size="1GB")
        for split_name, split_data in mapped_datasets.items():
            split_data.cleanup_cache_files()
        splits[split_name].pronunciations.to_tsv(staging_dir)
        os.rename(staging_dir, built_dataset_dir)

    return datasets.load_from_disk(built_dataset_dir)


class BpeMapper(DatasetMapper):
    def __init__(
            self,
            *args,
            vocab_words: Iterable[str],
            must_pronounce: Dict[str, str],
            forbidden_words: Dict[str, str],
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.vocab_words = set(vocab_words)
        self.must_pronounce = must_pronounce
        self.forbidden_words = set(forbidden_words)

    @property
    def config(self):
        return Configuration()

    def _can_spell(self, w):
        return True if (WordSpeller.WORD_PATTERN.match(w) and w not in self.forbidden_words) else False

    def map_sample(self, row: Dict[str, any]):
        word_sequence = [_bpe_word_to_string_cached(tuple(w)) for w in row["bpe_words"]]
        spellable = [self._can_spell(w) for i, w in enumerate(word_sequence)]
        pronounceable = [w in self.vocab_words for i, w in enumerate(word_sequence)]
        must_pronounce = self.must_pronounce.get(row["id"], [])
        return {
            "spellable": spellable,
            "pronounceable": pronounceable,
            "must_pronounce": must_pronounce
        }


@lru_cache(999999)
def _bpe_word_to_string_cached(w: Tuple[int]):
    return str(BpeSequence(w)).lower().strip()


def uint_dtype(max_value):
    # Auto-select a data type so the dataset is less yuge
    dtypes = (("uint8", 2**8), ("uint16", 2**16), ("uint32", 2**32), ("uint64", 2**64))
    return next(dtype for dtype, capacity in dtypes if capacity >= max_value)


def pronunciation_file_for_dataset(d: datasets.Dataset):
    d = os.path.dirname(d.cache_files[0]["filename"])
    return os.path.join(d, "pronunciations.tsv")


def get_median_occurrence_of_word(words: BpeSequenceCollection, must_pronounce: List[str]):
    # We only *must* pronounce one word per sample. Doesn't affect valid/test.
    must_pronounce = random.choice(must_pronounce) if len(must_pronounce) else None

    if must_pronounce is None:
        return None

    assert isinstance(must_pronounce, str)
    word_text = must_pronounce.lower().strip()
    idxs = [i for i, w in enumerate(words) if str(w).lower().strip(" “") == word_text]
    if not len(idxs):
        raise ValueError(f"Couldn't find word `{word_text}`")
    return idxs[(len(idxs) - 1) // 2]  # Median, guaranteeing the item is in the set.
