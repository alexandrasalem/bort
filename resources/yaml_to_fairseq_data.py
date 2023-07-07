import random
import re
import os
from functools import cache
from itertools import groupby


#import chatfile

from bort.resources.consts import UNK_TOKEN, PROD_DELIM_RIGHT, PROD_DELIM_LEFT
from bort.resources.dependencies import SUPERFLUOUS_BULLET_PATTERN
from bort.fairseq_ext.plugins.transforms import WordPronouncer
from bort.resources.logs import logger
from bort.resources.cmu import retrieve_cmudict
from bort.resources.datasets_talkbank import ErrorSessionCollection, Utterance, Session#, download_chat_files, iterate_chat_files
from bort.resources.pronounce import tokenize_ipa

gem_names_used = ["Cinderella", "Stroke", "Cat", "CAT", "Umbrella", "Important_Event", "Window", "Sandwich", "Sandwich_Picture", "Speech", "Flood"]


RANDOM_SEED = 8675309
random.seed(RANDOM_SEED)

#used originally
# def do_download(out_dir):
#     logger.info(f"Preparing CHAT files from TalkBank...")
#     if os.path.exists(out_dir):
#         retrieved = next(f for f in os.listdir(out_dir) if f.startswith("retrieved-"))
#         logger.info(f"Exists: {os.path.abspath(out_dir)} ({retrieved}), skipping download...")
#     else:
#         download_chat_files(out_dir)

#used originally
# def prepare_yaml_file(out_dir, yaml_file, parallel):
#     logger.info(f"Preparing YAML file...")
#     if os.path.exists(yaml_file):
#         logger.info(f"Exists: {os.path.abspath(yaml_file)}, skipping YAML prep...")
#     else:
#         limit_to_session = None  # Put e.g. a session name here to only process a subset
#         chat_files = iterate_chat_files(out_dir, match_substring=limit_to_session, parallel=parallel)
#         session_collection = ErrorSessionCollection.from_chat(chat_files, text_config=CONFIG_TEXT)
#         session_collection.to_yaml(yaml_file)

#used originally
def fix_msu03_quirk(out_dir):
    # Fix known issue, kinda inconsequentianl since it doesn't even have target words
    msu03b = os.path.join(out_dir, "CHAT/aphasia/English/Aphasia/MSU/MSU03b/MSU03b.cha")
    with open(msu03b, "r") as f:
        contents = "".join(f.readlines())
    with open(msu03b, "w") as f:
        f.write(contents.replace("<the trails [* s:uk]> [* n:uk]", "<the trails> [* s:uk] [* n:uk]"))


def should_exclude(session: Session, utterance: Utterance):
    return "INV" in utterance.speaker


def all_source_target_texts(sessions: ErrorSessionCollection, *, use_target_pron=False):
    participant_ids = []
    session_ids = []
    gem_names = []
    source_texts = []
    target_texts = []
    prods = []
    error_types = []
    missing_pronunciations = []
    for session in sessions:
        try:
            for gem_name, gem_utterances in groupby(session.utterances, key=lambda u: u.gem):
                if gem_name not in gem_names_used:
                    continue
                gem_sources, gem_targets, gem_missing, gem_prods, gem_error_types = gem_target_texts(
                    session=session,
                    gem_utterances=gem_utterances,
                    use_target_pron=use_target_pron
                )
                participant_ids.extend([session.name[:-1]]*len(gem_targets))
                session_ids.extend([session.name]*len(gem_targets))
                gem_names.extend([gem_name]*len(gem_targets))
                source_texts.extend(gem_sources)
                target_texts.extend(gem_targets)
                missing_pronunciations.extend(gem_missing)
                prods.extend(gem_prods)
                error_types.extend(gem_error_types)
        except Exception as e:
            raise RuntimeError(f"Failed to process {session.name}: {e}") from e
    return participant_ids, session_ids, gem_names, source_texts, target_texts, sorted(missing_pronunciations), prods, error_types


def gem_target_texts(session, gem_utterances, use_target_pron=False):
    gem_source_texts = []
    gem_target_texts = []
    gem_prods = []
    gem_error_types = []
    missing_pronunciations = []
    for utterance in gem_utterances:
        if should_exclude(session, utterance):
            continue
        if use_target_pron:
            not_found = [
                "\t".join(str(item) for item in (session.name, utterance.begin_ms or 0, e.target))
                for e in utterance.errors
                if not pronounceable(e.target) and e.target != UNK_TOKEN
            ]
        else:
            not_found = [
                "\t".join(str(item) for item in (session.name, utterance.begin_ms or 0, e.production, e.target))
                for e in utterance.errors
                if not pronounceable(e.production) and e.target != UNK_TOKEN
            ]
        missing_pronunciations.extend(not_found)
        source_texts, target_texts, prods, error_types = utterance_source_target_texts(
            session=session,
            utterance=utterance,
            use_target_pron=use_target_pron
        )
        gem_source_texts.append(source_texts)
        gem_target_texts.append(target_texts)
        gem_prods.append(prods)
        gem_error_types.append(error_types)

    gem_source_texts_expanded = []
    gem_target_texts_expanded = []
    gem_error_types_expanded = []
    gem_prods_expanded = []
    for i in range(len(gem_source_texts)):
        if len(gem_target_texts[i]) == 1:
            continue
        for j in range(len(gem_target_texts[i][:-1])):
            target = gem_target_texts[i][j]
            prod = gem_prods[i][j]
            error_type = gem_error_types[i][j]
            # here is where I skip unknown targets
            if target == UNK_TOKEN:
                continue
            source = ""
            for k in range(len(gem_source_texts)):
                if k == i:
                    source += gem_source_texts[k][j] + " "
                else:
                    source += gem_source_texts[k][-1] + " "
            gem_source_texts_expanded.append(normalize_text(source))
            gem_target_texts_expanded.append(target)
            gem_prods_expanded.append(prod)
            gem_error_types_expanded.append(error_type)

    return gem_source_texts_expanded, gem_target_texts_expanded, missing_pronunciations, gem_prods_expanded, gem_error_types_expanded


def replace_error(error, text, source_template, target_template=None):
    if UNK_TOKEN in (error.production, error.target):
        source_text_replace = UNK_TOKEN
        if target_template:
            target_text = UNK_TOKEN
    elif not pronounceable(error.production):
        pronounced = error.production
        source_text_replace = source_template.format(production=pronounced)
        if target_template:
            target_text = UNK_TOKEN
    else:
        pronounced = pronounce_word(error.production, position_id="TODO")
        source_text_replace = source_template.format(production=pronounced)
        if target_template:
            target_text = target_template.format(target=error.target)

    updated_text = text.replace(error.text, source_text_replace, 1)
    if target_template:
        return updated_text, target_text
    return updated_text

def replace_error_target_pron(error, text, source_template, target_template=None):
    if UNK_TOKEN in (error.production, error.target):
        source_text_replace = UNK_TOKEN
        if target_template:
            target_text = UNK_TOKEN
    elif not pronounceable(error.target):
        pronounced = error.target
        source_text_replace = source_template.format(production=pronounced)
        if target_template:
            target_text = UNK_TOKEN
    else:
        pronounced = pronounce_word(error.target, position_id="TODO")
        source_text_replace = source_template.format(production=pronounced)
        if target_template:
            target_text = target_template.format(target=error.target)

    updated_text = text.replace(error.text, source_text_replace, 1)
    if target_template:
        return updated_text, target_text
    return updated_text

def pronounceable(word: str):
    if word not in cached_dictionary() or word == UNK_TOKEN:
        return False

    try:
        pronunciation, *_ = cached_dictionary()[word]
        return len(tokenize_ipa(pronunciation)) > 0
    except ValueError as e:
        logger.warning(e)
        return False

@cache
def pronounce_word(word, position_id):
    """
    When there are multiple possible pronunciations, the pronouncer returns a random one. Caching this ensures that
    the pronunciation won't change for different samples of the same utterance.



    :param word:
    :param position_id: TODO: A per-error position_id (e.g. "williamson22a-u42-e01") could be provided to maintain
                        consistency while still allowing random pronunciations for different instances.
    :return:
    """
    pronunciation = cached_pronouncer().transform(word)

    # Since tokens are limited, we only keep the separators which force tokenization with BPE.
    compact_pronunciation = SUPERFLUOUS_BULLET_PATTERN.sub("", pronunciation)

    return compact_pronunciation

@cache
def cached_dictionary():
    return retrieve_cmudict()

@cache
def cached_pronouncer():
    return WordPronouncer(cached_dictionary())

def utterance_source_target_texts(session, utterance: Utterance, *, use_target_pron=False):
    if len(utterance.errors) == 0:
        source_texts = [str(utterance.text)]
        target_texts = ["NA"]
        prods = ["NA"]
        error_types = ["NA"]
        return source_texts, target_texts, prods, error_types
    source_error_template = PROD_DELIM_LEFT + "{production}" + PROD_DELIM_RIGHT
    source_other_error_template = "{production}"
    target_error_template = "{target}"
    source_text = str(utterance.text)
    target_text = str(utterance.text)
    source_texts = []
    target_texts = []
    prods = []
    error_types = []

    #versions with the error marked
    i = 0
    while i < len(utterance.errors):
        error = utterance.errors[i]
        prods.append(error.production)
        error_type = error.codes[0] if len(error.codes) == 1 else "multiple"
        error_types.append(error_type)
        for t in (source_text, target_text):
            assert error.text in t, f"`{error.text}` not in `{t}`!"
        if error.production[0] != "/" and re.search(r"[^ \'_\-a-zA-Z]", error.production):
            logger.warning(f"('{session.name}', '{utterance.id}', '{error.production}'): '{error.production}@u',")
        if use_target_pron:
            source_text_updated, target_text_updated = replace_error_target_pron(error, source_text,
                                                                     source_error_template, target_error_template)
        else:
            source_text_updated, target_text_updated = replace_error(error, source_text, source_error_template, target_error_template)

        j = 0
        while j < len(utterance.errors):
            if j == i:
                j+=1
                continue
            other_error = utterance.errors[j]
            if use_target_pron:
                source_text_updated = replace_error_target_pron(other_error, source_text_updated, source_other_error_template)
            else:
                source_text_updated = replace_error(other_error, source_text_updated, source_other_error_template)
            j+=1
        source_texts.append(source_text_updated)
        target_texts.append(target_text_updated)
        i+=1

    #versions with all errors unmarked
    source_text = str(utterance.text)
    target_text = str(utterance.text)
    i = 0
    while i < len(utterance.errors):
        error = utterance.errors[i]
        for t in (source_text, target_text):
            assert error.text in t, f"`{error.text}` not in `{t}`!"
        if use_target_pron:
            source_text = replace_error_target_pron(error, source_text, source_other_error_template)
        else:
            source_text = replace_error(error, source_text, source_other_error_template)
        i+=1
    source_texts.append(source_text)
    target_texts.append("NA")
    prods.append("NA")
    error_types.append("NA")
    return source_texts, target_texts, prods, error_types

def normalize_text(s: str):
    s = _remove_unwanted_terminators(s)
    s = _transform_remaining_pronunciations(s)  # translate remaining corrections that aren't errors
    s = re.sub(r"( [.!?])+(?=[ $])", ".", s)    # No space before terminators, and condense empty sentences.
    s = re.sub(r"[\s]+", " ", s)                # condense consecutive spaces
    s = re.sub(r"\<unk\>", r"unk", s)
    s = s.lstrip(" .")
    return s


def _transform_remaining_pronunciations(s: str):
    s = re.sub(r"\[.*?\]", "", s) #remove remaining corrections that aren't marked as errors
    prons = re.findall(r"\/.*?\/", s)
    for pron in prons:
        s = s.replace(pron, pronounce_word(pron, position_id="TODO"))
    return s


def _remove_unwanted_terminators(s: str):
    unwanted_terminators = (
        re.escape("+."),    # TERMINATOR_TYPE.BROKEN_FOR_CODING
        re.escape("+..."),  # TERMINATOR_TYPE.TRAIL_OFF
        re.escape("+..?"),  # TERMINATOR_TYPE.TRAIL_OFF_QUESTION
        re.escape("+!?"),   # TERMINATOR_TYPE.QUESTION_EXCLAMATION
        re.escape("+/."),   # TERMINATOR_TYPE.INTERRUPTION
        re.escape("+/?"),   # TERMINATOR_TYPE.INTERRUPTION_QUESTION
        re.escape("+//."),  # TERMINATOR_TYPE.SELF_INTERRUPTION
        re.escape("+//?"),  # TERMINATOR_TYPE.SELF_INTERRUPTION_QUESTION
        re.escape('+"/.'),  # TERMINATOR_TYPE.QUOTATION_NEXT_LINE
        re.escape('+".'),   # TERMINATOR_TYPE.QUOTATION_PRECEDES
        re.escape("≋"),     # TERMINATOR_TYPE.TECHNICAL_BREAK_TCU_CONTINUATION
        re.escape("≈"),     # TERMINATOR_TYPE.NO_BREAK_TCU_CONTINUATION
    )
    unwanted_terminators = sorted(unwanted_terminators, key=len, reverse=True)
    pattern = re.compile("|".join(unwanted_terminators))
    s = re.sub(pattern, "", s)
    return s


# CONFIG_TEXT = chatfile.TranscriptConfiguration(
#     unk_token=UNK_TOKEN,
#     # error_markings=False,
#
#     ga_alternatives=False,       # we want <one or two> [=? one too]
#     ga_comments=False,           # I really wish you wouldn't [% said with strong raising of eyebrows] do that.
#     ga_explanation=False,        # don't look in there [= closet]!
#     ga_paralinguistics=False,    # that's mine [=! cries].
#
#     pos=False,                   # goodbyes$n
#     prosody_drawl=False,         # baby want bana:nas?
#     prosody_pause=False,         # is that a rhi^noceros
#     prosody_blocking=False,      # ^
#     postcodes=False,             # not this one. [+ neg] [+ req] [+ inc]
#     happenings=False,            # &=points:nose
#     line_speaker_ids=False,      # *PAR:   ...
#     ca_delimiters=False,         # CHAT-CA Transcription
#
#     quotations=True,
#     linkers=False,
#     terminators=True,
#     separators=False,
#
#     tagmarkers=False,         # Mommy ‡ I want some
#     errors_imply_replacement=True,
#     overlap_markers=False,
#
#     replacements=True,
#     error_markings=True,
#     media=False,                 # The timestamps at the end
#     group_markers=False,
#
#     fragments=False,  # "&"
#     fillers=False,  # "&-"
#     filler_markers=False,        # &-
#     incompletes=False,  # "&+"
#     omissions=True,  # "0"
#     omission_markers=False,  # Treat omissions like the word was there
#
#     retracing_and_other_markers=False,
#     replaced_content=True,
#
#     pauses=False,
#
#     underscores_to_spaces=True,
#     unknown_replacements=True,
#
#     clitics_within_word=False,
#     compound_markers_within_word=False,
#     babbling=False,
#     repetition_markers=False,
#     repetitions_expand=False,
#     segment_repetition_markers_to_hyphens=True,
#     special_markers=False,
#     unibet_to_unk=False,
#     phonemic_slashes=True,
#     shortening_markers=False,
#     untranscribed_markers=True,
#     happenings_inaudible=False,
#     lang_markers=False,
# )
