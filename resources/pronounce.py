import csv
import multiprocessing
import os
import re
from functools import cache
from io import StringIO
from itertools import product, chain
from typing import Set, Mapping, Iterator, Callable, Collection, Dict, Tuple


def maybe_apply_mapping_maybe_parallel(iterable, func=None, n_parallel=1):
    func = func if func is not None else (lambda item: item)
    if n_parallel > 1:
        yield from multiprocessing.Pool(n_parallel).imap(func, iterable)
    else:
        yield from (func(item) for item in iterable)


class PronunciationDictionary(Mapping[str, Set[str]]):
    def __init__(
            self,
            normalize_key: Callable[[str], str] = None,
            normalize_value: Callable[[str], str] = None,
    ):
        self.normalize_key = normalize_key or _default_normalize_word
        self.normalize_value = normalize_value or str
        self._inner_dict = {}

    def __getitem__(self, key: str) -> Set[str]:
        if len(key) and (key[0], key[-1]) == ("/", "/"):
            # Already a pronunciation
            return {key.strip("/")}

        if self.normalize_key:
            key = self.normalize_key(key)

        if " " in key.strip():
            # Multi-word handling. Brings up questions around how these are handled.
            first, rest = key.split(maxsplit=1)
            return { f"{pron_a}{pron_b}" for pron_a, pron_b in product(self[first], self[rest]) }

        pronunciations = self._inner_dict.get(key, set())
        return set(pronunciations)

    def __len__(self) -> int:
        return len(self._inner_dict)

    def __iter__(self) -> Iterator[str]:
        return iter(self._inner_dict)

    def __contains__(self, item):
        return len(self.get(item, None)) > 0

    def num_pronunciations(self):
        return sum(len(p) for p in self.values())

    def num_words(self):
        return len(self)

    def update(self, key, values):
        key = self.normalize_key(key)
        if isinstance(values, str):
            return self.update(key, [values])
        if key not in self._inner_dict:
            self._inner_dict[key] = set()
        if self.normalize_value:
            values = (self.normalize_value(v) for v in values)
        self._inner_dict[key].update(values)

    @classmethod
    def from_string(
            cls,
            file_data,
            normalize_key: Callable[[str], str] = None,
            normalize_value: Callable[[str], str] = None
    ):
        io = StringIO(file_data)
        line_pattern = re.compile(r"^\w.*?\s.*$")
        lines = (
            line.strip().split(maxsplit=1)
            for line in io
            if line_pattern.match(line)
        )
        result = cls(normalize_key=normalize_key, normalize_value=normalize_value)
        for word, pronunciation in lines:
            word, *_ = word.split("(")
            result.update(word, pronunciation)
        return result

    def to_tsv(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            writer = csv.writer(f, dialect=csv.excel_tab)
            writer.writerow(("Word", "Pronunciation"))
            for word, pronunciations in sorted(self.items(), key=lambda kvp: kvp[0]):
                for pronunciation in sorted(pronunciations):
                    writer.writerow((word, pronunciation))

    @classmethod
    def from_dict(cls, d: Dict[str, Collection[str]], **kwargs):
        result = cls(**kwargs)
        for word, pronunciations in d.items():
            result.update(word, pronunciations)
        return result

    @classmethod
    def from_tsv(cls, filename, **kwargs):
        result = cls(**kwargs)
        with open(filename) as f:
            reader = csv.reader(f, dialect=csv.excel_tab)
            headers = next(reader)
            for values in reader:
                obj = dict(zip(headers, values))
                result.update(obj["Word"], obj["Pronunciation"])
        return result


def _default_normalize_word(word):
    return str(word).strip().lower()


def arpa_to_ipa(arpa, remove_stress=True):
    io = StringIO()

    # Quick hack to get rid of stress symbols. AH0 is a special case (schwa)
    for match in _arpa_token_pattern().finditer(arpa):
        arpa, not_arpa = match.groups()

        if not_arpa:
            raise ValueError(f"Couldn't handle non-ARPAbet symbol {not_arpa}")

        if arpa in {"0", "1", "2"} and remove_stress:
            continue

        ipa = _ARPABET_TO_IPA.get(match[1])[0]
        io.write(ipa)

    return io.getvalue()


def tokenize_ipa(ipa) -> Tuple[str, ...]:
    tokens = []

    for match in _ipa_token_pattern().finditer(ipa):
        ipa_token, not_ipa = match.groups()

        if ipa_token in _HANDLE_TALKBANK_SITUATIONS:
            replace_sequence = _HANDLE_TALKBANK_SITUATIONS[ipa_token]
            tokens.extend(replace_sequence)
            continue

        if not_ipa:
            raise ValueError(f"Couldn't handle non-IPA symbol {not_ipa} in {ipa}")

        tokens.append(ipa_token)

    return tuple(tokens)


_HANDLE_TALKBANK_SITUATIONS = {
    ":": [""],
    "ː": [""],
    "^": [""],

    # Below are used to normalize IPA to CMUDict's phoneme inventory.
    # Many of these may not even occur.
    "ɾ": ["t"],  # DX is not in CMUDict
    "a": ["ɑ"],
    "a͡ʊ": ["aʊ"],
    "a͡ɪ": ["aɪ"],
    "tʃ": ["ʧ"],
    "t͡ʃ": ["ʧ"],
    "ɜ˞": ["ɝ"],
    "ə˞": ["ɝ"],
    "ɚ": ["ɝ"],
    "eɪ": ["e"],  # Merging this diphthong, preferring shortest symbol
    "e͡ɪ": ["e"],
    "g": ["ɡ"],
    "dʒ": ["ʤ"],
    "d͡ʒ": ["ʤ"],
    "oʊ": ["o"],  # Merging this diphthong, preferring shortest symbol
    "o͡ʊ": ["o"],
    "ɔ͡ɪ": ["ɔɪ"],
    "r": ["ɹ"],
}

@cache
def _ipa_token_pattern():
    ipa_symbols = list(chain(*_ARPABET_TO_IPA.values()))
    ipa_symbols.extend(_HANDLE_TALKBANK_SITUATIONS)
    ipa_symbols_long_to_short = sorted(ipa_symbols, key=lambda symbol: len(symbol.encode("utf8")), reverse=True)
    ipa_symbols_long_to_short = "|".join(re.escape(sym) for sym in ipa_symbols_long_to_short)
    pattern_str = rf"\s*({ipa_symbols_long_to_short})|(.)"
    return re.compile(pattern_str)


@cache
def _arpa_token_pattern():
    arpa_symbols_long_to_short = sorted(_ARPABET_TO_IPA, key=lambda symbol: len(symbol.encode("utf8")), reverse=True)
    arpa_symbols_long_to_short = "|".join(re.escape(sym) for sym in arpa_symbols_long_to_short)
    pattern_str = rf"\s*({arpa_symbols_long_to_short})|(.)"
    return re.compile(pattern_str)


_ARPABET_TO_IPA = {
    "0": ("", ),
    "1": ("ˈ", ),
    "2": ("ˌ", ),
    "AA": ("ɑ", "a",),
    "AE": ("æ", ),
    "AH": ("ʌ",),
    "AH0": ("ə",),
    "AO": ("ɔ",),
    "AW": ("aʊ", "a͡ʊ",),
    "AX": ("ə",),
    "AY": ("aɪ", "a͡ɪ",),
    "B": ("b",),
    "CH": ("ʧ", "tʃ", "t͡ʃ",),
    "D": ("d",),
    "DH": ("ð",),
    "DX": ("ɾ",),
    "EH": ("ɛ",),
    "ER": ("ɝ", "ɜ˞", "ə˞", "ɚ",),
    "EY": ("e", "eɪ", "e͡ɪ",),
    "F": ("f",),
    "G": ("ɡ", "g",),
    "HH": ("h",),
    "IH": ("ɪ",),
    "IY": ("i",),
    "JH": ("ʤ", "dʒ", "d͡ʒ",),
    "K": ("k",),
    "L": ("l",),
    "M": ("m",),
    "N": ("n",),
    "NG": ("ŋ",),
    "OW": ("o", "oʊ", "o͡ʊ",),
    "OY": ("ɔɪ", "ɔ͡ɪ",),
    "P": ("p",),
    "Q": ("ʔ",),
    "R": ("ɹ", "r",),
    "S": ("s",),
    "SH": ("ʃ",),
    "T": ("t",),
    "TH": ("θ",),
    "UH": ("ʊ",),
    "UW": ("u",),
    "V": ("v",),
    "W": ("w",),
    "Y": ("j",),
    "Z": ("z",),
    "ZH": ("ʒ",),
}
