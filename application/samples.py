import base64
import urllib
from typing import Dict, List

import streamlit as st
from github import Github
from github.ContentFile import ContentFile

GIT_REPO = "PlanTL-GOB-ES/SPACCC_MEDDOCAN"
FOLDER_PATH = "corpus/test/brat"


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def get_samples(
    repository=GIT_REPO,
    path: str = FOLDER_PATH,
    git_token: str = None,
    branch: str = "master",
) -> Dict[str, ContentFile]:
    files = github_repository_file_names(
        repository=repository,
        path=path,
        git_token=git_token,
        branch=branch,
    )
    return files


def get_text_from_name(files, select_report: str, suffix: str = "txt") -> str:
    name = ".".join([select_report, suffix])
    content_file = files[name]
    return text_from_content(content_file)


def github_file_to_bytes(repository, path, git_token, branch="main"):
    g = Github(git_token)
    repo = g.get_repo(repository)
    content_encoded = repo.get_contents(
        urllib.parse.quote(path), ref=branch
    ).content
    content = base64.b64decode(content_encoded)
    return content


def github_repository_list(
    repository: str, path: str, git_token: str = None, branch: str = "master"
) -> List[ContentFile]:
    g = Github(git_token)
    repo = g.get_repo(repository)
    content = repo.get_contents(urllib.parse.quote(path), ref=branch)
    if not isinstance(content, list):
        content = [content]
    return content


def github_repository_file_names(
    repository: str, path: str, git_token: str = None, branch: str = "master"
) -> Dict[str, ContentFile]:
    content_files = github_repository_list(
        repository, path, git_token, branch=branch
    )
    return {content_file.name: content_file for content_file in content_files}


def text_from_content(content_file: ContentFile, encoding="utf-8") -> str:
    content_encoded = content_file.content
    if content_encoded is not None:
        content = base64.b64decode(content_encoded)
        text = content.decode(encoding)
        return text
    else:
        raise ValueError(
            f"The content_file {content_file.name} as a None content!"
        )
