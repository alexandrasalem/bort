import fairseq.data
import numpy as np

from bort.bpe_dataset_tools.datasets_bpe import PartBBpeDataset

class BracketsSpacesPhonemesDataset(fairseq.data.DenoisingDataset):
    def __init__(
            self,
            source_dataset: PartBBpeDataset,
            target_dataset: PartBBpeDataset,
            sizes,
            vocab,
            # pronouncer: PronounceWordsBpeTransform,
            shuffle,
            seed,
            args,
    ):
        # We've overridden all but the most basic functionality from fairseq.data.DenoisingDataset (e.g. __len__, collater, etc)
        args.replace_length = 1 # check on this?
        super().__init__(source_dataset, np.array(sizes), vocab, None, None, shuffle, seed, args)
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.max_source_positions = args.max_source_positions
        self.max_target_positions = args.max_target_positions

    def __getitem__(self, index):
        source_tensor = self.source_dataset[index].to_model_encoding(self.vocab)
        target_tensor = self.target_dataset[index].to_model_encoding(self.vocab)

        self._assert_length(source_tensor, self.max_source_positions, "Source is too long!")
        self._assert_length(target_tensor, self.max_target_positions, "Target is too long!")

        return {
            "id": index,
            "source": source_tensor,
            "target": target_tensor
        }

    def collater(self, samples, pad_to_length=None):
        collated = super().collater(samples, pad_to_length)
        return collated

    def _assert_length(self, values, max_length, msg):
        if len(values) > max_length:
            # fix this
            text = " "#" ".join(w.byte_value for w in self.from_fairseq_dictionary_symbols(values))
            raise ValueError(f"Text is too long! ({len(values)} > {max_length})\n{text}")

    @classmethod
    def build(cls, args, *, split, dictionary, epoch=1, combine=False, **kwargs):
        trim_to_length = args.max_target_positions - 2  # the two: (<s>, </s>)
        trim_method = "custom_angle_bracket"
        filename = f"{args.data}/splits/{split}/source.txt"
        with open(filename) as f:
            sources = f.read().splitlines()
        filename = f"{args.data}/splits/{split}/target.txt"
        with open(filename) as f:
            targets = f.read().splitlines()
        source_data = PartBBpeDataset.from_texts(sources, trim_to_length=trim_to_length)
        target_data = PartBBpeDataset.from_texts(targets, trim_to_length=trim_to_length)  #the trimming shouldn't actually be needed here
        return cls(
            source_data,
            target_data,
            source_data.sizes,
            dictionary,
            shuffle=args.shuffle_instance,
            seed=args.seed,
            args=args,
        )
