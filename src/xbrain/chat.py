"""Dashboard chat over the generated XBrain markdown vault."""

from __future__ import annotations

import re
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xbrain.llm_client import LlmProvider, build_llm_client
from xbrain.llm_json import json_from_response
from xbrain.notes_io import GEN_END, GEN_START

MAX_QUESTION_CHARS = 1200
MAX_SOURCES = 6
CHUNK_CHAR_LIMIT = 1800
ANSWER_MAX_TOKENS = 900

_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]{3,}")
_STOPWORDS = frozenset(
    {
        "aqui",
        "about",
        "algo",
        "also",
        "and",
        "are",
        "como",
        "con",
        "cual",
        "cuales",
        "del",
        "des",
        "desde",
        "donde",
        "els",
        "esta",
        "estan",
        "este",
        "esto",
        "for",
        "hay",
        "las",
        "los",
        "para",
        "per",
        "por",
        "que",
        "the",
        "una",
        "what",
        "with",
    }
)


@dataclass(frozen=True)
class MarkdownChunk:
    """One searchable chunk from a generated markdown file."""

    title: str
    path: Path
    text: str
    index: int


@dataclass(frozen=True)
class ChatSource:
    """A source snippet sent to the LLM and returned to the dashboard."""

    id: str
    title: str
    path: Path
    excerpt: str
    score: float

    def to_payload(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "path": str(self.path),
            "excerpt": self.excerpt,
            "score": round(self.score, 3),
        }


@dataclass(frozen=True)
class ChatAnswer:
    """Structured response returned by `/api/chat`."""

    answer: str
    sources: list[ChatSource]
    provider: LlmProvider
    model: str
    scanned_files: int
    retrieved_sources: int

    def to_payload(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "sources": [source.to_payload() for source in self.sources],
            "provider": self.provider,
            "model": self.model,
            "scanned_files": self.scanned_files,
            "retrieved_sources": self.retrieved_sources,
        }


def _generated_region(markdown: str) -> str:
    """Return only XBrain's generated block when markers are present."""
    start = markdown.find(GEN_START)
    if start == -1:
        return markdown
    start += len(GEN_START)
    end = markdown.find(GEN_END, start)
    if end == -1:
        return markdown[start:]
    return markdown[start:end]


def _plain_text(markdown: str) -> str:
    """Normalize markdown into searchable prose while keeping useful labels."""
    text = re.sub(r"```.*?```", " ", markdown, flags=re.S)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"!\[\[([^\]]+)\]\]", r" \1 ", text)
    text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^---\s*$.*?^---\s*$", " ", text, flags=re.S | re.M)
    text = re.sub(r"[#>*_`~|-]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _title_for(path: Path, markdown: str) -> str:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or path.stem
    return path.stem.replace("-", " ")


def _markdown_paths(output_dir: Path) -> list[Path]:
    """Return generated item/topic pages, excluding index/log noise."""
    paths: list[Path] = []
    for dirname in ("items", "topics"):
        directory = output_dir / dirname
        if directory.exists():
            paths.extend(sorted(directory.glob("*.md")))
    return paths


def _split_chunks(text: str, *, limit: int = CHUNK_CHAR_LIMIT) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        if len(paragraph) > limit:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            for piece in textwrap.wrap(
                paragraph,
                width=limit,
                break_long_words=False,
                replace_whitespace=False,
            ):
                chunks.append(piece)
            continue
        next_len = current_len + len(paragraph) + (2 if current else 0)
        if current and next_len > limit:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_len = len(paragraph)
        else:
            current.append(paragraph)
            current_len = next_len
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def load_markdown_chunks(output_dir: Path) -> tuple[list[MarkdownChunk], int]:
    """Load searchable chunks from the generated markdown vault."""
    chunks: list[MarkdownChunk] = []
    paths = _markdown_paths(output_dir)
    for path in paths:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        generated = _generated_region(raw)
        plain = _plain_text(generated)
        if not plain:
            continue
        title = _title_for(path, generated)
        for index, chunk in enumerate(_split_chunks(plain)):
            chunks.append(MarkdownChunk(title=title, path=path.resolve(), text=chunk, index=index))
    return chunks, len(paths)


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in (match.group(0).lower() for match in _WORD_RE.finditer(text))
        if token not in _STOPWORDS
    ]


def _score_chunk(query: Counter[str], chunk: MarkdownChunk) -> float:
    body = Counter(_tokens(chunk.text))
    title = Counter(_tokens(chunk.title))
    if not body and not title:
        return 0.0
    score = 0.0
    for token, weight in query.items():
        score += min(body.get(token, 0), 3) * weight
        score += min(title.get(token, 0), 2) * weight * 1.8
    return score


def retrieve_sources(
    output_dir: Path,
    question: str,
    *,
    limit: int = MAX_SOURCES,
) -> tuple[list[ChatSource], int]:
    """Retrieve the best source snippets for a question using local text search."""
    query = Counter(_tokens(question))
    chunks, scanned_files = load_markdown_chunks(output_dir)
    if not query or not chunks:
        return [], scanned_files

    scored = [
        (score, chunk)
        for chunk in chunks
        if (score := _score_chunk(query, chunk)) > 0
    ]
    scored.sort(key=lambda pair: (-pair[0], str(pair[1].path), pair[1].index))

    sources: list[ChatSource] = []
    seen_paths: set[Path] = set()
    for score, chunk in scored:
        if chunk.path in seen_paths:
            continue
        seen_paths.add(chunk.path)
        sources.append(
            ChatSource(
                id=f"S{len(sources) + 1}",
                title=chunk.title,
                path=chunk.path,
                excerpt=chunk.text[:CHUNK_CHAR_LIMIT].strip(),
                score=score,
            )
        )
        if len(sources) >= limit:
            break
    return sources, scanned_files


def _system_prompt() -> str:
    return (
        "Eres el chat de biblioteca de XBrain. Responde solo con la informacion "
        "incluida en los fragmentos de markdown proporcionados.\n"
        "Reglas obligatorias:\n"
        "- No uses conocimiento externo ni supongas hechos que no aparezcan en los fragmentos.\n"
        "- Si los fragmentos no bastan, dilo de forma clara.\n"
        "- Responde en el mismo idioma de la pregunta.\n"
        "- Cita solo IDs de fuente existentes, como S1 o S2.\n"
        "- Devuelve un unico objeto JSON: "
        '{"answer":"...","sources":["S1","S2"]}'
    )


def _user_prompt(question: str, sources: list[ChatSource]) -> str:
    blocks = [f"Pregunta del usuario:\n{question}", "", "Fragmentos disponibles:"]
    for source in sources:
        blocks += [
            "",
            f"[{source.id}] {source.title}",
            f"Path: {source.path}",
            "Contenido:",
            source.excerpt,
        ]
    return "\n".join(blocks)


def _selected_sources(data: dict[str, Any], sources: list[ChatSource]) -> list[ChatSource]:
    raw_ids = data.get("sources", [])
    if not isinstance(raw_ids, list):
        return []
    valid_ids = {str(source_id) for source_id in raw_ids}
    return [source for source in sources if source.id in valid_ids]


def answer_question(
    output_dir: Path,
    question: str,
    *,
    provider: LlmProvider,
    model: str,
    base_url: str | None = None,
    client: Any = None,
    max_sources: int = MAX_SOURCES,
) -> ChatAnswer:
    """Answer one question using only generated markdown context."""
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question must not be empty")
    if len(cleaned) > MAX_QUESTION_CHARS:
        raise ValueError(f"question is too long; max {MAX_QUESTION_CHARS} characters")

    sources, scanned_files = retrieve_sources(output_dir, cleaned, limit=max_sources)
    if not sources:
        return ChatAnswer(
            answer=(
                "No he encontrado informacion suficiente en los markdown generados de XBrain "
                "para responder a esa pregunta."
            ),
            sources=[],
            provider=provider,
            model=model,
            scanned_files=scanned_files,
            retrieved_sources=0,
        )

    active_client = client or build_llm_client(provider, base_url=base_url)
    response = active_client.messages.create(
        model=model,
        max_tokens=ANSWER_MAX_TOKENS,
        system=_system_prompt(),
        messages=[{"role": "user", "content": _user_prompt(cleaned, sources)}],
    )
    data = json_from_response(response, context="dashboard chat")
    answer = str(data.get("answer", "")).strip()
    if not answer:
        raise ValueError("dashboard chat response has no answer")
    selected = _selected_sources(data, sources)
    if not selected and "no he encontrado" not in answer.lower():
        selected = sources[: min(3, len(sources))]
    return ChatAnswer(
        answer=answer,
        sources=selected,
        provider=provider,
        model=model,
        scanned_files=scanned_files,
        retrieved_sources=len(sources),
    )
