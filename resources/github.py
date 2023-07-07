import requests

BASE_URL = "https://raw.githubusercontent.com"


def retrieve(
        *,
        user: str,
        repo: str,
        path: str,
        revision: str,
        base_url=BASE_URL,
        text=True
):
    url = f"{base_url.strip('/')}/{user}/{repo}/{revision}/{path.strip('/')}"
    response = _requests_get_cacheable(url)
    if text:
        return response.text
    else:
        return response.content


def _requests_get_cacheable(url):
    get = requests.get
    response = get(url)
    response.raise_for_status()
    return response