"""Number citations by first appearance, emit the AMA bibliography and a Zotero BibTeX file."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
MANUSCRIPT = PROJECT / "manuscript"
ORDER = ["introduction", "methods", "results", "discussion"]

JOURNAL_ABBREV = {
    "Electroencephalography and Clinical Neurophysiology": "Electroencephalogr Clin Neurophysiol",
    "New England Journal of Medicine": "N Engl J Med",
    "IEEE Transactions on Neural Systems and Rehabilitation Engineering":
        "IEEE Trans Neural Syst Rehabil Eng",
    "Journal of Neural Engineering": "J Neural Eng",
    "PLoS ONE": "PLoS One",
    "Biomedical Physics & Engineering Express": "Biomed Phys Eng Express",
    "The Lancet": "Lancet",
    "Journal of the Royal Statistical Society Series B: Statistical Methodology":
        "J R Stat Soc Series B Stat Methodol",
    "Frontiers in Neuroscience": "Front Neurosci",
    "Scientific Reports": "Sci Rep",
    "Biomedical Physics & Engineering Express": "Biomed Phys Eng Express",
}

CONFERENCE_MARKERS = ("Proceedings of the",)


def clean(text: str) -> str:
    """Undo publisher metadata artefacts: HTML entities and stray whitespace."""
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()


def normalize_author(name: str) -> str:
    """Render any supplied author form as AMA surname plus initials."""
    name = clean(name)
    if not name:
        return ""
    if "," in name:
        family, _, given = name.partition(",")
    elif re.fullmatch(r"[^\W\d_][\w'\- ]* [A-Z]{1,3}", name, flags=re.UNICODE):
        # Already in AMA form, including multi-word surnames such as "von Elm E".
        return name
    else:
        parts = name.split()
        family, given = parts[-1], " ".join(parts[:-1])
    initials = "".join(part[0].upper() for part in re.split(r"[\s\-]+", given.strip()) if part)
    return f"{family.strip()} {initials}".strip()

PREPRINT_PREFIXES = ("10.1101/", "10.64898/")


def collect_order() -> list[str]:
    keys: list[str] = []
    for name in ORDER:
        text = (MANUSCRIPT / f"{name}.md").read_text()
        for match in re.finditer(r"\[(REF-[A-Z0-9-]+)\]", text):
            if match.group(1) not in keys:
                keys.append(match.group(1))
    return keys


def author_string(authors: list[str]) -> str:
    cleaned = [normalize_author(a) for a in authors if clean(a) not in {"", ":", "Qwen"}]
    cleaned = [a for a in cleaned if a]
    if not cleaned:
        return "Qwen Team"
    if len(cleaned) > 6:
        return ", ".join(cleaned[:6]) + ", et al"
    return ", ".join(cleaned)


def ama_entry(record: dict) -> str:
    authors = author_string(record.get("authors", []))
    title = clean(record.get("title", "")).rstrip(".")
    doi = record.get("doi")
    year = record.get("year")
    if record.get("arxiv"):
        return f"{authors}. {title}. arXiv preprint arXiv:{record['arxiv']}."
    if doi and doi.startswith(PREPRINT_PREFIXES):
        return f"{authors}. {title}. bioRxiv preprint. doi:{doi}"
    if record.get("type") == "dataset":
        return f"{authors}. {title}. PhysioNet; {year}. doi:{doi}"
    if record.get("type") == "webpage":
        when = f"{record['month']} {year}" if record.get("month") and year else (str(year) if year else "")
        published = f"Published {when}. " if when else ""
        return f"{authors}. {title}. {published}{record.get('url', '')}"
    container = clean(record.get("container") or "")
    container = JOURNAL_ABBREV.get(container, container)
    volume = record.get("volume") or ""
    issue = f"({record['issue']})" if record.get("issue") else ""
    pages = f":{record['page']}" if record.get("page") else ""
    if any(marker in container for marker in CONFERENCE_MARKERS):
        page_text = f":{record['page']}" if record.get("page") else ""
        return f"{authors}. {title}. In: *{container}*. {year}{page_text}. doi:{doi}"
    locus = f"{year};{volume}{issue}{pages}" if year else ""
    return f"{authors}. {title}. *{container}*. {locus}. doi:{doi}"


def bibtex_entry(key: str, number: int, record: dict) -> str:
    fields = {
        "title": record.get("title", "").replace("{", "").replace("}", ""),
        "author": " and ".join(record.get("authors", [])),
        "year": record.get("year", ""),
        "journal": record.get("container", ""),
        "volume": record.get("volume", ""),
        "number": record.get("issue", ""),
        "pages": record.get("page", ""),
        "doi": record.get("doi", ""),
        "url": record.get("url", ""),
        "month": record.get("month", ""),
    }
    if record.get("arxiv"):
        fields["eprint"] = record["arxiv"]
        fields["archivePrefix"] = "arXiv"
    body = ",\n  ".join(f"{k} = {{{v}}}" for k, v in fields.items() if v)
    kind = "misc" if record.get("arxiv") or record.get("type") in ("dataset", "webpage") else "article"
    return f"@{kind}{{ref{number}_{key.lower().replace('-', '_')},\n  {body}\n}}"


def main() -> None:
    verified = json.loads((MANUSCRIPT / "refs_verified.json").read_text())
    numbering_path = MANUSCRIPT / "refs_numbering.json"
    keys = collect_order()
    if not keys and numbering_path.exists():
        stored = json.loads(numbering_path.read_text())
        keys = [key for key, _ in sorted(stored.items(), key=lambda kv: kv[1])]
    missing = [key for key in keys if key not in verified or verified[key]["status"] != "OK"]
    if missing:
        raise SystemExit(f"unverified references cited in text: {missing}")
    numbering = {key: index + 1 for index, key in enumerate(keys)}
    numbering_path.write_text(json.dumps(numbering, indent=2))

    for name in ORDER:
        path = MANUSCRIPT / f"{name}.md"
        text = path.read_text()
        text = re.sub(
            r"(?:\[(REF-[A-Z0-9-]+)\],?)+",
            lambda m: "^" + ",".join(
                str(numbering[k]) for k in re.findall(r"REF-[A-Z0-9-]+", m.group(0))
            ) + "^",
            text,
        )
        path.write_text(text)

    lines = [f"{numbering[key]}. {ama_entry(verified[key])}" for key in keys]
    (MANUSCRIPT / "references.md").write_text("## References\n\n" + "\n\n".join(lines) + "\n")
    (MANUSCRIPT / "references.bib").write_text(
        "\n\n".join(bibtex_entry(key, numbering[key], verified[key]) for key in keys) + "\n"
    )
    print(f"{len(keys)} references numbered and formatted")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
