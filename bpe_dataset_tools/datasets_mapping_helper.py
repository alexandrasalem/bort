import multiprocessing
from functools import lru_cache
from typing import Union, Dict, Iterator, List

from datasets.formatting.formatting import LazyBatch


class _P2GBatch(dict):
    """A wrapper type that clarifies that we're looking at a batch rather than an individual sample"""
    pass


@lru_cache()
def _pool(num_proc: int):
    return multiprocessing.Pool(num_proc) if num_proc else multiprocessing.pool.ThreadPool(1)


class DatasetMapper:
    """
    Helps handle batch/non-batch mapping for HF bpe_dataset_tools in a more predictable way
    """

    def __init__(self, *args, num_proc=None, **kwargs):
        self.num_proc = num_proc

    def __call__(self, data_to_map: Union[LazyBatch, "_P2GBatch", dict]):
        # fork the overloaded
        if isinstance(data_to_map, (LazyBatch, _P2GBatch)):
            return self.map_batch(data_to_map)
        else:
            return self.map_sample(data_to_map)

    def map_sample(self, sample: Dict[str, any]):
        raise NotImplementedError()

    def map_batch(self, batch: LazyBatch):
        out_batch = _pool(self.num_proc).imap(self.map_sample, self.iterate_batch(batch))
        out_batch = list(out_batch)
        result = self.restack_batch(out_batch)
        return result

    def iterate_batch(self, batch: LazyBatch) -> Iterator[Dict[str, any]]:
        keys = list(batch.keys())
        all_values = zip(*batch.values())
        yield from (dict(zip(keys, values)) for values in all_values)

    def restack_batch(self, samples: List[Dict[str, any]]):
        keys = samples[0].keys() if samples else []
        values = zip(*(s.values() for s in samples)) if samples else []
        pydict = {k: list(v) for k, v in zip(keys, values)}
        return _P2GBatch(pydict)