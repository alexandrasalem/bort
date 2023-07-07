import multiprocessing
import dataclasses
import os
from datetime import date
from datetime import datetime
from functools import lru_cache
from multiprocessing.pool import ThreadPool
from pathlib import PosixPath
from tempfile import TemporaryDirectory
from typing import Tuple, List, Iterable

#import chatfile
import yaml
#from chatfile.stringify_nodes import AnnotatedGroup, Error
#from talkbankclient import TalkBank
#from talkbankclient.types import FullPath
from tqdm import tqdm

from bort.resources.consts import UNK_TOKEN
from bort.resources.logs import logger
from bort.resources.yaml_serialization import yaml_loader_with_our_dataclasses, yaml_dumper_with_our_dataclasses


TALKBANK_UNK_TOKEN = "xxx"

# CONFIG_CHAT = chatfile.TranscriptConfiguration(
#     line_speaker_ids=False,
#     errors_imply_replacement=True
# )
#
# CONFIG_ERROR_CHAT = dataclasses.replace(
#     CONFIG_CHAT,
#     repetition_markers=False,
#     retracing_and_other_markers=False,
# )

# def _wrapped_chatfile_parse(filename):
#     try:
#         return chatfile.parsing.parse_chat_file(filename)
#     except Exception as e:
#         raise RuntimeError(f"Couldn't load {filename}: {e}") from e

# def iterate_chat_files(d, match_substring=None, parallel=8, progress_bar=True) -> Iterable[chatfile.Chat]:
#     chat_filenames = list(sorted(iterate_chat_filenames(d, match_substring)))
#     if not len(chat_filenames):
#         raise FileNotFoundError(f"No chat files found in {d}")
#
#     if parallel > 1:
#         jobs = multiprocessing.Pool(8).imap(_wrapped_chatfile_parse, chat_filenames)
#     else:
#         jobs = (_wrapped_chatfile_parse(f) for f in chat_filenames)
#
#     if progress_bar:
#         for chat in tqdm(jobs, total=len(chat_filenames)):
#             yield chat
#     else:
#         yield from jobs


# def iterate_chat_filenames(d, match_substring=None):
#     if not os.path.exists(d):
#         raise NotADirectoryError(d)
#
#     for root, dirs, files in sorted(os.walk(d), key=lambda rdf: rdf[0].lower()):
#         for filename in sorted(files, key=lambda fn: fn.lower()):
#             if match_substring and match_substring.lower() not in filename.lower():
#                 continue
#             if filename.lower().endswith(".cha"):
#                 yield os.path.join(root, filename)


# def download_chat_files(download_dir, remote_root=("aphasia", "aphasia/English/Aphasia"), n_threads=32):
#     if os.path.exists(download_dir):
#         raise FileExistsError(f"{os.path.abspath(download_dir)}")
#
#     remote_path = FullPath(*remote_root)
#     logger.info(f"Cloning {remote_path} => {download_dir}")
#
#     with TemporaryDirectory() as temp_dir, TalkBank(cache_dir=temp_dir) as tb:
#         PosixPath(temp_dir).joinpath(f"retrieved-{datetime.now()}").touch()
#         relevant_chat_files = tb.corpora.get(remote_path)
#         logger.info(f"Downloading TalkBank's {remote_path} to {download_dir}")
#         jobs = ThreadPool(n_threads).imap(tb.chats.get_text, relevant_chat_files)
#         list(tqdm(jobs, total=len(relevant_chat_files)))
#         os.rename(temp_dir, download_dir)


@dataclasses.dataclass
class Error:
    chat: str
    text: str
    production: str
    target: str
    codes: Tuple[str, ...]

    # @classmethod
    # def from_annotated_group(cls, annotated_group: AnnotatedGroup, text_config: chatfile.TranscriptConfiguration):
    #     config_error_content = dataclasses.replace(
    #         text_config,
    #         replaced_content=True,
    #         replacements=False,
    #         error_markings=False,
    #         repetition_markers=False,
    #         retracing_and_other_markers=False,
    #         errors_imply_replacement=True,
    #     )
    #     assert annotated_group.is_error_group()
    #     production = _cached_stringify(annotated_group.content, config_error_content)
    #     production = KNOWN_FIXES.get(production, production)
    #     unk_token = config_error_content.unk_token or TALKBANK_UNK_TOKEN
    #     target = _cached_stringify(annotated_group.replacement.words, config_error_content) if annotated_group.replacement else unk_token
    #     target = KNOWN_FIXES.get(target, target)
    #     chat = _cached_stringify(annotated_group, CONFIG_ERROR_CHAT)
    #     text = _cached_stringify(annotated_group, text_config)
    #
    #     chat = chat.replace("[: x@n]", f"[: {UNK_TOKEN}]")  # HACK
    #     text = text.replace("[: x]", f"[: {UNK_TOKEN}]")    # HACK
    #
    #     codes = tuple(_cached_stringify(n, CONFIG_CHAT) for n in annotated_group.content if isinstance(n, chatfile.stringify_nodes.Error))
    #     return cls(
    #         chat=chat,
    #         text=text,
    #         production=production,
    #         target=target,
    #         codes=codes,
    #     )


@dataclasses.dataclass
class Utterance:
    id: str
    speaker: str
    gem: str
    begin_ms: int
    end_ms: int
    chat: str
    text: str
    errors: Tuple[Error, ...]

    # @classmethod
    # def from_utterance(cls, utterance: chatfile.Utterance, gem: str, text_config: chatfile.TranscriptConfiguration):
    #     errors = tuple(
    #         Error.from_annotated_group(node, text_config=text_config)
    #         for node in chatfile.iterate_nodes(utterance, filter_to_type=AnnotatedGroup)
    #         if node.is_error_group()
    #     )
    #     begin_ms = utterance.media.start if utterance.media.start is None else (int(utterance.media.start * 1000))
    #     end_ms = utterance.media.end if utterance.media.end is None else (int(utterance.media.end * 1000))
    #
    #     chat = _cached_stringify(utterance, CONFIG_CHAT)
    #     text = _cached_stringify(utterance, text_config)
    #
    #     chat = chat.replace("[: x@n]", f"[: {UNK_TOKEN}]")  # HACK
    #     text = text.replace("[: x]", f"[: {UNK_TOKEN}]")    # HACK
    #
    #     return cls(
    #         id=str(utterance.u_id),
    #         speaker=str(utterance.who),
    #         gem=gem,
    #         begin_ms=begin_ms,
    #         end_ms=end_ms,
    #         chat=chat,
    #         text=text,
    #         errors=errors
    #     )


@dataclasses.dataclass
class Session:
    name: str
    date: date
    utterances: Tuple[Utterance, ...]

    # @classmethod
    # def from_chat(cls, session: chatfile.Chat, text_config: chatfile.TranscriptConfiguration):
    #     utterances = []
    #     for gem, gem_utterances in session.gems.items():
    #         parsed = tuple(
    #             Utterance.from_utterance(u, gem=str(gem), text_config=text_config)
    #             for u in gem_utterances
    #         )
    #         utterances.extend(parsed)
    #     session_date = date(*session.date[:3]) if session.date else None
    #     return cls(name=str(session.media), date=session_date, utterances=tuple(utterances))


class ErrorSessionCollection(List[Session]):
    def to_yaml(self, filename: str):
        dumper = yaml_dumper_with_our_dataclasses(
            data_class_types=[Session, Utterance, Error],
            collection_types=[ErrorSessionCollection]
        )
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        with open(filename, "w", encoding="utf8") as f:
            f.write(yaml.dump(self, Dumper=dumper, sort_keys=False, allow_unicode=True))
        logger.info(f"Wrote session file to {filename}")

    # @classmethod
    # def from_chat(cls, sessions: Iterable[chatfile.Chat], text_config: chatfile.TranscriptConfiguration):
    #     return cls((Session.from_chat(c, text_config) for c in sessions))

    @classmethod
    def from_yaml(cls, filename) -> "ErrorSessionCollection":
        try:
            loader = yaml_loader_with_our_dataclasses(
                data_class_types=[Session, Utterance, Error],
                collection_types=[ErrorSessionCollection]
            )
            return yaml.load(open(filename, "rb"), Loader=loader)
        except Exception as e:
            raise RuntimeError(f"Couldn't load YAML file: {filename}") from e


# @lru_cache(maxsize=99999)
# def _cached_stringify(*args, **kwargs):
#     return chatfile.stringify(*args, **kwargs)


KNOWN_FIXES = {
    "x": UNK_TOKEN,
    "/x/": UNK_TOKEN,

    # Missing @u
    'tʃi': '/tʃi/',
    'rɑ': '/rɑ/',
    'wɪz': '/wɪz/',
    'tuspeɪsts': '/tuspeɪsts/',
    'kæfɚ': '/kæfɚ/',
    'rɪkɪn': '/rɪkɪn/',
    'hɔː': '/hɔː/',
    'θu': '/θu/',
    'ʌt': '/ʌt/',
    'ɡə': '/ɡə/',
    'ɹɛsju': '/ɹɛsju/',
    'tɪkəɹɛlə': '/tɪkəɹɛlə/',
    'mɪdnɑɪk': '/mɪdnɑɪk/',
    'ɑɪdɑɪŋgəlɪ': '/ɑɪdɑɪŋgəlɪ/',
    'twɑtʃ': '/twɑtʃ/',
    'blekɪŋ': '/blekɪŋ/',
    'sɛr': '/sɛr/',
    'slɪdz': '/slɪdz/',
    'fɔk': '/fɔk/',
    'lɛnts': '/lɛnts/',
    'mejɪt': '/mejɪt/',
    'rɑɪəlz': '/rɑɪəlz/',
    'sɛnsɪsɪz': '/sɛnsɪsɪz/',
    'kemɪn': '/kemɪn/',
    'də': '/də/',
    'kræt': '/kræt/',
    'hækɪt': '/hækɪt/',
    'kampɪteɪʃənz': '/kampɪteɪʃənz/',
    'dɪsɑɪbɚz': '/dɪsɑɪbɚz/',
    'slɪplɚ': '/slɪplɚ/',
    'tritʃɪn': '/tritʃɪn/',
    'frʌn': '/frʌn/',
    'neɪləf': '/neɪləf/',
    'æstjɚ': '/æstjɚ/',
    'traɪvz': '/traɪvz/',
    'fɪŋ': '/fɪŋ/',
    'hɑʊt': '/hɑʊt/',
    'bɝli': '/bɝli/',
    'æpjulɛsɪn': '/æpjulɛsɪn/',
    'vɔkɪŋ': '/vɔkɪŋ/',
    'sɔndɚɛlə': '/sɔndɚɛlə/',
    'sɛksɪstɛk': '/sɛksɪstɛk/',
    'blɪkɪŋ': '/blɪkɪŋ/',
    'mɑt': '/mɑt/',
    'vædlɑɪk': '/vædlɑɪk/',
    'sɪnjəlrɛrlə': '/sɪnjəlrɛrlə/',
    'ɑbɪks': '/ɑbɪks/',
    'fɚɪns': '/fɚɪns/',
    'seɪmd': '/seɪmd/',
    'fɪdio': '/fɪdio/',
    'dɔks': '/dɔks/',
    'spæntwɪs': '/spæntwɪs/',
    'sɪksɚz': '/sɪksɚz/',
    'iʔi': '/iʔi/',
    'bæm': '/bæm/',
    'feɪʒə': '/feɪʒə/',
    'hɑspəl': '/hɑspəl/',
    'twɛnwə': '/twɛnwə/',
    'rispɑnsə': '/rispɑnsə/',
    'sɪldərɛlə': '/sɪldərɛlə/',
    'ændʒərɛlə': '/ændʒərɛlə/',
    'sɑn': '/sɑn/',
    'dʌ': '/dʌ/',
    'dʒu_P_S': '/dʒu_P_S/',

    # Mixed orthography and IPA
    "/stætʃlɪ of Liberty/": "/stætʃlɪ əv lɪbɝti/",  # BU05a
    "/Myrtle bis/": "/mɝtəl bis/",  # fridriksson01a
    "/all of sʌðɪŋ/": "/ɔl ʌv sʌðɪŋ/",  # kurland29d
    "/all of a sʌdɪŋ/": "/ɔl ʌv ə sʌdɪŋ/",  # kurland29c
    "/all ʌz ə fʌzɪn/": "/ɔl ʌz ə fʌzɪn/",  # kurland29d
    "/all of a tʃʌn/": "/ɔl ʌv ə tʃʌn/",  # kurland29e
    "/Prince trɑʊrɪ/": "/pɹɪns trɑʊrɪ/",  # whiteside15a
    "/Arlington hɑtsɪbəl/": "/ɑɹlɪŋtən hɑtsɪbəl/",  # williamson01a

    # "c": ["k"],  # cædəmɪnt@u "cabinet" - fridriksson01a
    # # crɪmbrɛlə@u "crɪmbrɛlə" - kurland19d
    # # creɪ@u "crazy" - kurland19d
    # "q": ["ɑ"],  # krɑɪnəlqdʒɪkəl@u "chronological" - kurland06a
    # "I": ["ɪ"],  # deɪI@u, dʌdI@u, ðæsIŋ@u, wʌsIŋ@u, tʌsIŋ@u, dIl@u, ɔfIt@u, dIndʌn@u - UNH07a
    # "y": ["i"],  # moʊnly@u "mainly" - MSU08b
    # "ɜ": ["ɛ"],  # ɛmə@u ɜstər@u for "ever after" - ACWT05a

    # New typos
    "/Pibɚ/": "/pibɚ/",  # NEURAL/48-1
    "/sɪndəwɜlə/": "/sɪndəwɛlə/",  # kurland07b
    "/yə˞ælə/": "/jə˞ælə/",  # BU08a

    # These seem like typos. Should be fixed in AphasiaBank already.
    "/cædəmɪnt/": "/kædəmɪnt/",  # fridriksson01a
    "/crɪmbrɛlə/": "/krɪmbrɛlə/",  # kurland19d
    "/creɪ/": "/kreɪ/",  # kurland19d
    "/krɑɪnəlqdʒɪkəl/": "/krɑɪnəlɑdʒɪkəl/",  # kurland06a
    "/deɪI/": "/deɪɪ/",  # UNH07a
    "/dʌdI/": "/dʌdɪ/",  # UNH07a
    "/ðæsIŋ/": "/ðæsɪŋ/",  # UNH07a
    "/wʌsIŋ/": "/wʌsɪŋ/",  # UNH07a
    "/tʌsIŋ/": "/tʌsɪŋ/",  # UNH07a
    "/dIl/": "/dɪl/",  # UNH07a
    "/ɔfIt/": "/ɔfɪt/",  # UNH07a
    "/dIndʌn/": "/dɪndʌn/",  # UNH07a
    "/moʊnly/": "/moʊnli/",  # MSU08b
    "/ɜstər/": "/ɛstər/",    # ACWT05a

    ############
    # These are narrower than our inventory

    # Remove unvoiced diacritic
    "/m̥lʌv/": "/mlʌv/",  # kurland29e

    # Remove rhotic hook
    "/bɛn˞/": "/bɛn/",  # scale14a
    "/zomɪŋwʊd˞/": "/zomɪŋwʊd/",  # scale14a

    # [ħ] to /h/
    "/ħɑʊs/": "/hɑʊs/",  # thompson06a

    # [d͡z] to /dz/
    "/ɔfɪd͡z/": "/ɔfɪdz/",  # UNH02b

    # [x] to /h/
    "/xɑʊs/": "/hɑʊs/",  # adler16a
    "/poxən/": "/pohən/",  # CMU02a
    "/hoxən/": "/hohən/",  # CMU02a
    "/nʌxɪn/": "/nʌhɪn/",  # CMU02b
    "/bʊxəbɪ/": "/bʊhəbɪ/",  # kansas14a

    # /l̩/ to /əl/
    "/lɪhl̩/": "/lɪhəl/",  # cmu02a
    "/lɜ˞tl̩/": "/lɜ˞təl/",  # tap07a
    "/stʌnl̩/": "/stʌnəl/",  # wright203a
    "/dɪɾl̩/": "/dɪɾəl/",  # wright204a

    # /æ̃/ to /æ/
    "/pæ̃/": "/pæ/",  # williamson23a

    #  /ɒ/ to /ɔ/
    "/oʊkwɒt/": "/oʊkwɔt/",  # ACWT02a
    "/pɒpkɪn/": "/pɔpkɪn/",  # ACWT12a
    "/hɒtləs/": "/hɔtləs/",  # fridriksson06b
    "/θɒkə/": "/θɔkə/",  # fridriksson06b
    "/ʌpɒd/": "/ʌpɔd/",  # scale18a

    #  /ɸ/ to /f/
    "/ɸrɪnts/": "/frɪnts/",  # fridriksson09b
    "/ɸrɪnɪts/": "/frɪnɪts/",  # ACWT02a
}
