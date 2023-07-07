import logging
import os
from fairseq.models import register_model
from fairseq.models.bart import BARTModel

from bort.resources.consts import UNK_TOKEN

logger = logging.getLogger(__name__)


@register_model("bort")
class BORTModel(BARTModel):
    """
    Fairseq wrapper for BORT models uploaded to Huggingface
    """

    @classmethod
    def from_pretrained(cls, model_name_or_path, checkpoint_file="model.pt", *args, variant=None, **kwargs):
        if model_name_or_path == "palat/bort":
            model_name_or_path, checkpoint_file = cls._get_huggingface_path(variant)
        model = super().from_pretrained(model_name_or_path, checkpoint_file, *args, **kwargs)
        return model

    @classmethod
    def _get_huggingface_path(cls, variant):
        try:
            import transformers

            transformers.AutoModel.from_pretrained("palat/bort", variant=variant)
            filename = transformers.WEIGHTS_NAME
            if variant is not None:
                filename = filename.replace(".bin", f".{variant}.bin")
            cached_file = transformers.utils.cached_file("palat/bort", filename)
            return os.path.split(cached_file)
        except ImportError as e:
            raise ImportError("Try: `pip install transformers~=4.30.2`")

def load_model(model_name_or_path, checkpoint_file="model.pt", data_name_or_path=".", bpe="gpt2"):
    import fairseq.models.bart
    import fairseq.hub_utils
    import fairseq.data

    from_hub_utils = fairseq.hub_utils.from_pretrained(
        model_name_or_path,
        checkpoint_file,
        data_name_or_path,
        bpe=bpe,
        load_checkpoint_heads=True,
        archive_map=fairseq.models.bart.BARTModel.hub_models()
    )
    return from_hub_utils["models"][0]


def load_model_dictionary(data_dir, model_name_or_path):
    import fairseq.data

    dict_path = os.path.join(data_dir, f"dict.{model_name_or_path.replace('/', '_')}.txt")
    if not os.path.exists(dict_path):
        model = BORTModel.from_pretrained(model_name_or_path) #load_model(model_name_or_path)
        model.encoder.dictionary.save(dict_path)
    dictionary = fairseq.data.Dictionary.load(dict_path)
    dictionary.add_symbol(UNK_TOKEN)
    logger.info("dictionary: {} types".format(len(dictionary)))
    return dictionary