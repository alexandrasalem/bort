from bort.resources.pronounce import PronunciationDictionary, arpa_to_ipa
from bort.resources import github


def retrieve_cmudict(
        *,
        file="cmudict-0.7b",
        user="Alexir",
        repo="CMUdict",
        revision="7a37de7",
        lowercase_words=True,
        ipa=True,
):
    normalize_key = str.lower if lowercase_words else str.upper
    normalize_value = arpa_to_ipa if ipa else None

    response_text = github.retrieve(user=user, repo=repo, path=file, revision=revision)
    cmudict = PronunciationDictionary.from_string(
        response_text,
        normalize_key=normalize_key,
        normalize_value=normalize_value
    )
    return cmudict


