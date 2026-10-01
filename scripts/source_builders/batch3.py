"""Context adapters for the 20-dataset high-K batch.

The frozen index supplies labels, candidate texts, score bins and source IDs;
this module only reconstructs source-derived context strings. Rebuild code
matches contexts by SHA-256, so schema or normalization drift fails loudly.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Iterable, Iterator

from bs4 import BeautifulSoup
from datasets import get_dataset_config_names, load_dataset


REPOS = {
    "metacritic_games": ("Wada1/Metacritic_Games_Reviews_Dataset", "9109e48069fe4509421b71a98c7b6ef104253913"),
    "pitchfork_albums": ("Kebez/Pitchfork-Album-Reviews", "e6fb9718a9f20ac6464942051b02cbb379558b19"),
    "wine_quality": ("spawn99/wine-reviews", "e6b10f4db3091a6fed8c5b294c0cc885e7f6e99d"),
    "civil_comments_toxicity": ("google/civil_comments", "f2970eb3a55777454c94069077cc8d9b5866312d"),
    "measuring_hate_speech": ("ucberkeley-dlab/measuring-hate-speech", "5468f6e118396646b02a2f691e771f6b6d9502ea"),
    "stsbenchmark_similarity": ("mteb/stsbenchmark-sts", "96943a16ea6a35129e253c659081cb59daf81b30"),
    "wmt20_translation_quality": ("wmt/wmt20_mlqe_task1", "0783ed2bd75f44835df4ea664f9ccb85812c8563"),
    "word_concreteness": ("StephanAkkerman/concreteness-ratings", "10f460a4d535800d89c579db7cd2664dbaf6b1d6"),
    "openreview_recommendation": ("smallari/openreview-iclr-peer-reviews", "8ba8536f32b3f001627a2534cdb741649b4d1933"),
    "shp_community_approval": ("stanfordnlp/SHP", "e94b5f32602712d78ed494fe79105b1959396686"),
    "stackexchange_answer_score": ("HuggingFaceH4/stack-exchange-preferences", "c7bda74048748f55749cd663c3d8d1025a841fd9"),
    "realtoxicity_continuation": ("allenai/real-toxicity-prompts", "f21629712ffd6a3d13a54fd2807ccd521c55ef74"),
    "webtext_quality": ("lightblue/text_ratings", "953c12db0c7d21bcc9326bacdc29bee895741a07"),
    "dbpedia14": ("fancyzhx/dbpedia_14", "9abd46cf7fc8b4c64290f26993c540b92aa145ac"),
    "yahoo_answers10": ("community-datasets/yahoo_answers_topics", "6652a1e7c94f7260a0bfd0c9092dd48e2d536ea1"),
    "snips7": ("benayas/snips", "16915b895754dd4028068abe52b6851a904977d7"),
    "scotus13": ("MAdAiLab/lex_glue_scotus", "b1260ac4756b9d2ed8e75ff9e9cfd6a8e76b2bf7"),
}
LANGUAGE_NAMES = {"en":"English","de":"German","zh":"Chinese","et":"Estonian","ne":"Nepali","ro":"Romanian","si":"Sinhala"}


def clean_html(value) -> str:
    return html.unescape(BeautifulSoup(str(value or ""), "html.parser").get_text(" ")).strip()


def hf_rows(source: str, split: str, cache_dir: Path, config: str | None = None) -> Iterable[dict]:
    repo, revision = REPOS[source]
    return load_dataset(repo, config, split=split, revision=revision, streaming=True,
                        cache_dir=str(cache_dir), trust_remote_code=True)


def standard_contexts(source: str, cache_dir: Path) -> Iterator[str]:
    if source == "metacritic_games":
        for r in hf_rows(source, "train", cache_dir):
            yield f"Game: {r.get('Game','')}\nPlatform: {r.get('Platform','')}\nReview: {r.get('Review','')}"
    elif source == "pitchfork_albums":
        for r in hf_rows(source, "train", cache_dir): yield str(r.get("description") or "").strip()
    elif source == "wine_quality":
        for r in hf_rows(source, "test", cache_dir):
            yield (f"Wine: {r.get('title') or ''}\nOrigin: {r.get('country') or ''}, {r.get('province') or ''}\n"
                   f"Variety: {r.get('variety') or ''}\nReview: {r.get('description') or ''}")
    elif source == "civil_comments_toxicity":
        for r in hf_rows(source, "test", cache_dir): yield str(r.get("text") or "").strip()
    elif source == "measuring_hate_speech":
        for r in hf_rows(source, "train", cache_dir): yield str(r.get("text") or "").strip()
    elif source == "stsbenchmark_similarity":
        for split in ("train", "dev", "test"):
            for r in hf_rows(source, split, cache_dir):
                yield f"Sentence 1: {r.get('sentence1','')}\nSentence 2: {r.get('sentence2','')}"
    elif source == "wmt20_translation_quality":
        for config in get_dataset_config_names(REPOS[source][0], revision=REPOS[source][1], trust_remote_code=True):
            if "-" not in config: continue
            a, b = config.split("-", 1)
            for r in hf_rows(source, "train", cache_dir, config):
                trans = r.get("translation") or {}
                yield (f"Source ({LANGUAGE_NAMES.get(a,a)}): {trans.get(a,'')}\n"
                       f"Translation ({LANGUAGE_NAMES.get(b,b)}): {trans.get(b,'')}")
    elif source == "word_concreteness":
        for r in hf_rows(source, "train", cache_dir):
            yield f"Word or expression: {r.get('Word') or r.get('word') or ''}"
    elif source == "openreview_recommendation":
        for r in hf_rows(source, "raw", cache_dir):
            for review in r.get("reviews") or []:
                sections = [f"Paper title: {r.get('title','')}", f"Abstract: {r.get('abstract','')}",
                            f"Review summary: {review.get('summary','')}", f"Strengths: {review.get('strengths','')}",
                            f"Weaknesses: {review.get('weaknesses','')}"]
                yield "\n".join(sections).strip()
                yield "\n".join(sections + [f"Questions: {review.get('questions','')}"]).strip()
    elif source == "shp_community_approval":
        for r in hf_rows(source, "validation", cache_dir):
            domain = r.get("domain") or r.get("subreddit") or r.get("community") or ""
            post = r.get("history") or r.get("post") or r.get("prompt") or ""
            yield f"Community: {domain}\nPost: {post}".strip()
    elif source == "stackexchange_answer_score":
        for r in hf_rows(source, "train", cache_dir):
            question = clean_html(r.get("question"))
            for answer in r.get("answers") or []:
                yield f"Question: {question}\nAnswer: {clean_html(answer.get('text'))}"
    elif source == "realtoxicity_continuation":
        for r in hf_rows(source, "train", cache_dir):
            yield f"Prompt: {r['prompt']['text']}\nContinuation: {r['continuation']['text']}"
    elif source == "webtext_quality":
        for r in hf_rows(source, "train", cache_dir, "en"):
            for key in ("selected_chunk", "text"):
                if r.get(key): yield str(r[key]).strip()
    elif source == "dbpedia14":
        for r in hf_rows(source, "test", cache_dir, "dbpedia_14"):
            yield f"Title: {r.get('title','')}\nDescription: {str(r.get('content','')).strip()}"
    elif source == "yahoo_answers10":
        for r in hf_rows(source, "test", cache_dir, "yahoo_answers_topics"):
            yield (f"Question title: {r.get('question_title','')}\nQuestion: {r.get('question_content','')}\n"
                   f"Best answer: {r.get('best_answer','')}")
    elif source == "snips7":
        for split in ("train", "validation", "test"):
            try:
                for r in hf_rows(source, split, cache_dir): yield str(r.get("text") or "").strip()
            except Exception:
                continue
    elif source == "scotus13":
        for split in ("train", "validation", "test"):
            try:
                for r in hf_rows(source, split, cache_dir): yield str(r.get("text") or "").strip()
            except Exception:
                continue
    else:
        raise KeyError(source)


def git_contexts(source: str, source_root: Path) -> Iterator[str]:
    root = source_root / source
    if source == "emobank_valence":
        import csv
        candidates = list(root.rglob("*.csv"))
        for path in candidates:
            with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
                for r in csv.DictReader(handle):
                    text = r.get("text") or r.get("sentence") or r.get("content")
                    if text: yield str(text).strip()
    elif source == "humicroedit_funniness":
        import csv
        for path in list(root.rglob("*.csv")) + list(root.rglob("*.tsv")):
            delimiter = "\t" if path.suffix == ".tsv" else ","
            with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
                for r in csv.DictReader(handle, delimiter=delimiter):
                    original = r.get("original") or r.get("headline") or ""
                    edit = r.get("edit") or r.get("replacement") or ""
                    edited = re.sub(r"<[^<>]*/>", edit, original)
                    for headline in (edited, r.get("edited") or r.get("sentence") or ""):
                        if headline: yield f"Edited headline: {headline}\nReplacement word: {edit}".strip()
    elif source == "stanford_politeness":
        from convokit import Corpus
        corpus = Corpus(filename=str(root))
        for utterance in corpus.iter_utterances():
            if utterance.text: yield utterance.text.strip()
    else:
        raise KeyError(source)


def iter_contexts(source: str, artifacts: Path) -> Iterator[str]:
    if source in {"emobank_valence", "humicroedit_funniness", "stanford_politeness"}:
        yield from git_contexts(source, artifacts)
    else:
        yield from standard_contexts(source, artifacts / "_hf_cache")

