#!/usr/bin/env python3
"""Zotero-grounded RAG evidence for polymer design priors.

This module is intentionally sidecar-only.  It snapshots a Zotero collection
into local, hashed chunks; retrieves evidence for polymer-design contexts; and
proposes design-prior profiles that still require deterministic validation
before they can support manuscript or workflow claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from polymer_design_validator import get_adapter
from pelcb_design_origin import build_design_origin_outputs


SCRIPT_PATH = Path(__file__).resolve()
MD_ANALYSIS_DIR = SCRIPT_PATH.parent
DEFAULT_ZOTERO_DIR = Path.home() / "Zotero"
DEFAULT_COLLECTION_KEY = "9F4WKNX3"
DEFAULT_OUTPUT_DIR = MD_ANALYSIS_DIR / "output" / "pelcb100_zotero_rag"
PROJECT_ID = "pelcb_design_priors"
ALLOWLIST_TAG = "PELCB_MCP_EXECUTE_DESIGN_PRIOR_PROFILE"

OUTPUT_FILENAMES = {
    "zotero_rag_corpus_manifest_json": "zotero_rag_corpus_manifest.json",
    "zotero_rag_chunks_jsonl": "zotero_rag_chunks.jsonl",
    "zotero_rag_refresh_ledger_jsonl": "zotero_rag_refresh_ledger.jsonl",
    "zotero_design_prior_extractions_jsonl": "zotero_design_prior_extractions.jsonl",
    "zotero_design_prior_summary_md": "zotero_design_prior_summary.md",
    "case_design_origin_table_rag_overlay_csv": "case_design_origin_table_rag_overlay.csv",
    "rag_design_origin_ledger_jsonl": "rag_design_origin_ledger.jsonl",
}


# These are the three manuscript sources used to define the scientific design
# axes for PE-LCB100. They do not, by themselves, define every discrete value
# in the 100-case grid; the carbon budget and registered builder grammar do that.
DEFAULT_PELCB_DESIGN_DOIS = (
    "10.1021/ma00192a029",
    "10.1021/acs.macromol.8b00958",
    "10.1007/s10965-025-04467-0",
)

PELCB_DESIGN_SOURCE_ROLES = {
    "10.1021/ma00192a029": {
        "prior_type": "polyethylene_topology_context",
        "design_variable": "branch_topology",
        "suggested_range_or_category": (
            "linear and star-branched polyethylene, with experimentally studied arm counts from 3 to 18"
        ),
        "claim_boundary": (
            "Supports the use of polyethylene topology and arm number as design axes; "
            "it does not define the PE-LCB100 branch-length grid."
        ),
        "retrieval_terms": ["arm numbers ranging from 3 to 18", "star-branched polyethylene", "arm lengths"],
    },
    "10.1021/acs.macromol.8b00958": {
        "prior_type": "branch_parameter_context",
        "design_variable": "branch_length_count_and_position",
        "suggested_range_or_category": (
            "methyl-to-hexyl branches, one- and two-branch cases, and central or differently spaced placements"
        ),
        "claim_boundary": (
            "Supports branch length, branch count and placement as variables in polyethylene; "
            "it studies short-chain branches and does not directly define the full PE-LCB100 grid."
        ),
        "retrieval_terms": [
            "one methyl",
            "one ethyl",
            "one propyl",
            "one butyl",
            "one hexyl",
            "two methyl",
            "two ethyl",
            "branches are located",
        ],
    },
    "10.1007/s10965-025-04467-0": {
        "prior_type": "branch_parameter_context",
        "design_variable": "branch_length_content_and_distribution",
        "suggested_range_or_category": (
            "branch lengths of 2, 4, 6 and 8 coarse-grained beads, branch contents of 1-5%, "
            "and side-rich or center-rich placement"
        ),
        "claim_boundary": (
            "Supports systematic variation of branch length, content and distribution in polyethylene; "
            "the exact PE-LCB100 discrete values remain project-defined."
        ),
        "retrieval_terms": [
            "branches of 2,4,6 and 8 beads",
            "branch content (%)",
            "center-rich",
            "side-rich",
            "number, length, and distribution",
        ],
    },
}


def _normalize_doi(value: Any) -> str:
    text = str(value or "").strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text.strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return value


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text or ""))
    return re.sub(r"\s+", " ", text).strip()


def _year_from_date(text: Any) -> str:
    match = re.search(r"(19|20)\d{2}", str(text or ""))
    return match.group(0) if match else ""


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_+\-.]+", text.lower())


def _chunk_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return []
    if len(clean) <= chunk_size:
        return [clean]
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(clean):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def _connect_zotero(zotero_dir: Path) -> sqlite3.Connection:
    db_path = zotero_dir / "zotero.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Zotero SQLite database not found: {db_path}")
    con = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _field_map(cur: sqlite3.Cursor, item_id: int) -> dict[str, str]:
    cur.execute(
        """
        select f.fieldName, v.value
        from itemData d
        join fields f on f.fieldID=d.fieldID
        join itemDataValues v on v.valueID=d.valueID
        where d.itemID=?
        """,
        (item_id,),
    )
    return {str(row["fieldName"]): str(row["value"] or "") for row in cur.fetchall()}


def _creators(cur: sqlite3.Cursor, item_id: int) -> list[str]:
    cur.execute(
        """
        select cr.firstName, cr.lastName
        from itemCreators ic
        join creators cr on cr.creatorID=ic.creatorID
        where ic.itemID=?
        order by ic.orderIndex
        """,
        (item_id,),
    )
    names: list[str] = []
    for row in cur.fetchall():
        name = " ".join(part for part in [row["firstName"], row["lastName"]] if part)
        if name:
            names.append(name)
    return names


def _tags(cur: sqlite3.Cursor, item_id: int) -> list[str]:
    cur.execute(
        """
        select tags.name
        from itemTags it
        join tags on tags.tagID=it.tagID
        where it.itemID=?
        order by tags.name
        """,
        (item_id,),
    )
    return [str(row["name"]) for row in cur.fetchall()]


def _notes(cur: sqlite3.Cursor, item_id: int) -> list[str]:
    try:
        cur.execute("select note from itemNotes where parentItemID=?", (item_id,))
    except sqlite3.OperationalError:
        return []
    return [_strip_html(str(row["note"] or "")) for row in cur.fetchall() if str(row["note"] or "").strip()]


def _attachments(cur: sqlite3.Cursor, item_id: int, zotero_dir: Path) -> list[dict[str, Any]]:
    cur.execute(
        """
        select child.itemID, child.key, ia.contentType, ia.path, ia.linkMode
        from itemAttachments ia
        join items child on child.itemID=ia.itemID
        where ia.parentItemID=?
        """,
        (item_id,),
    )
    out: list[dict[str, Any]] = []
    for row in cur.fetchall():
        attachment_key = str(row["key"] or "")
        storage_dir = zotero_dir / "storage" / attachment_key
        ft_cache = storage_dir / ".zotero-ft-cache"
        raw_path = str(row["path"] or "")
        source_path = ""
        if raw_path.startswith("storage:"):
            source_path = str(storage_dir / raw_path.split("storage:", 1)[1])
        elif raw_path:
            source_path = raw_path
        fulltext = ""
        fulltext_status = "fulltext_missing"
        if ft_cache.exists():
            fulltext = ft_cache.read_text(encoding="utf-8", errors="replace")
            fulltext_status = "fulltext_available" if fulltext.strip() else "fulltext_empty"
        out.append(
            {
                "attachment_key": attachment_key,
                "content_type": row["contentType"],
                "link_mode": row["linkMode"],
                "source_path": source_path,
                "fulltext_cache_path": str(ft_cache) if ft_cache.exists() else "",
                "fulltext_status": fulltext_status,
                "fulltext": fulltext,
            }
        )
    return out


def snapshot_zotero_collection(
    *,
    zotero_dir: str | Path = DEFAULT_ZOTERO_DIR,
    collection_key: str = DEFAULT_COLLECTION_KEY,
    include_dois: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Read one Zotero collection plus explicitly requested library items.

    ``include_dois`` is used for project-critical sources that may live outside
    the selected Zotero collection. Exact DOI matching prevents a broad
    lexical query from silently substituting topically similar papers.
    """

    zotero_dir = Path(zotero_dir)
    con = _connect_zotero(zotero_dir)
    cur = con.cursor()
    cur.execute("select collectionID, collectionName, key from collections where key=?", (collection_key,))
    collection = cur.fetchone()
    if collection is None:
        con.close()
        raise ValueError(f"Zotero collection key not found: {collection_key}")
    cur.execute(
        """
        select i.itemID, i.key, it.typeName
        from collectionItems ci
        join items i on i.itemID=ci.itemID
        join itemTypes it on it.itemTypeID=i.itemTypeID
        where ci.collectionID=?
        order by ci.orderIndex, i.itemID
        """,
        (collection["collectionID"],),
    )
    selected_rows = {int(row["itemID"]): row for row in cur.fetchall()}
    requested_dois = sorted({_normalize_doi(value) for value in (include_dois or []) if _normalize_doi(value)})
    for doi in requested_dois:
        cur.execute(
            """
            select i.itemID, i.key, it.typeName
            from itemData d
            join fields f on f.fieldID=d.fieldID
            join itemDataValues v on v.valueID=d.valueID
            join items i on i.itemID=d.itemID
            join itemTypes it on it.itemTypeID=i.itemTypeID
            where lower(f.fieldName)='doi' and lower(v.value)=?
            """,
            (doi,),
        )
        for row in cur.fetchall():
            selected_rows[int(row["itemID"])] = row
    items: list[dict[str, Any]] = []
    for row in sorted(selected_rows.values(), key=lambda value: int(value["itemID"])):
        if str(row["typeName"]) in {"attachment", "note"}:
            continue
        item_id = int(row["itemID"])
        fields = _field_map(cur, item_id)
        attachments = _attachments(cur, item_id, zotero_dir)
        notes = _notes(cur, item_id)
        items.append(
            {
                "zotero_item_id": item_id,
                "zotero_item_key": str(row["key"] or ""),
                "item_type": str(row["typeName"] or ""),
                "title": fields.get("title", ""),
                "DOI": fields.get("DOI", ""),
                "url": fields.get("url", ""),
                "date": fields.get("date", ""),
                "year": _year_from_date(fields.get("date", "")),
                "publicationTitle": fields.get("publicationTitle", ""),
                "abstractNote": fields.get("abstractNote", ""),
                "creators": _creators(cur, item_id),
                "tags": _tags(cur, item_id),
                "notes": notes,
                "attachments": attachments,
            }
        )
    con.close()
    return {
        "collection_key": str(collection["key"]),
        "collection_name": str(collection["collectionName"]),
        "zotero_dir": str(zotero_dir),
        "include_dois": requested_dois,
        "items": items,
    }


def _metadata_text(item: dict[str, Any]) -> str:
    return "\n".join(
        part
        for part in [
            f"Title: {item.get('title', '')}",
            f"Authors: {', '.join(item.get('creators', []) or [])}",
            f"Year: {item.get('year', '')}",
            f"Publication: {item.get('publicationTitle', '')}",
            f"DOI: {item.get('DOI', '')}",
            f"URL: {item.get('url', '')}",
            f"Tags: {', '.join(item.get('tags', []) or [])}",
            f"Abstract: {item.get('abstractNote', '')}",
            "Notes: " + " ".join(item.get("notes", []) or []),
        ]
        if str(part).strip()
    )


def _make_chunk(
    *,
    item: dict[str, Any],
    chunk_type: str,
    text: str,
    index: int,
    attachment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attachment = attachment or {}
    chunk_id = f"{item.get('zotero_item_key')}:{chunk_type}:{index:04d}"
    fulltext_status = attachment.get("fulltext_status") or (
        "metadata_only" if chunk_type in {"metadata", "note"} else "fulltext_missing"
    )
    return {
        "schema_version": "zotero_rag_chunk.v1",
        "chunk_id": chunk_id,
        "chunk_type": chunk_type,
        "chunk_index": index,
        "chunk_hash": _sha256_text(text),
        "zotero_item_key": item.get("zotero_item_key", ""),
        "title": item.get("title", ""),
        "DOI": item.get("DOI", ""),
        "url": item.get("url", ""),
        "year": item.get("year", ""),
        "tags": item.get("tags", []),
        "attachment_key": attachment.get("attachment_key", ""),
        "source_path": attachment.get("source_path", ""),
        "fulltext_cache_path": attachment.get("fulltext_cache_path", ""),
        "fulltext_status": fulltext_status,
        "text": text,
    }


def build_zotero_rag_outputs(
    *,
    zotero_dir: str | Path = DEFAULT_ZOTERO_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    collection_key: str = DEFAULT_COLLECTION_KEY,
    include_dois: Iterable[str] | None = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 160,
) -> dict[str, Any]:
    """Snapshot a Zotero collection and write local RAG chunks."""

    output_dir = Path(output_dir)
    snapshot = snapshot_zotero_collection(
        zotero_dir=zotero_dir,
        collection_key=collection_key,
        include_dois=include_dois,
    )
    chunks: list[dict[str, Any]] = []
    for item in snapshot["items"]:
        meta = _metadata_text(item)
        for i, text in enumerate(_chunk_text(meta, chunk_size=chunk_size, chunk_overlap=0)):
            chunks.append(_make_chunk(item=item, chunk_type="metadata", text=text, index=i))
        for note_i, note in enumerate(item.get("notes", []) or []):
            for j, text in enumerate(_chunk_text(note, chunk_size=chunk_size, chunk_overlap=0)):
                chunks.append(_make_chunk(item=item, chunk_type="note", text=text, index=note_i * 1000 + j))
        for attachment in item.get("attachments", []) or []:
            for j, text in enumerate(
                _chunk_text(str(attachment.get("fulltext", "")), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            ):
                chunks.append(_make_chunk(item=item, chunk_type="fulltext", text=text, index=j, attachment=attachment))

    manifest = {
        "schema_version": "zotero_rag_corpus_manifest.v1",
        "created_at": _utc_now(),
        "collection_key": snapshot["collection_key"],
        "collection_name": snapshot["collection_name"],
        "zotero_dir": snapshot["zotero_dir"],
        "include_dois": snapshot.get("include_dois", []),
        "item_count": len(snapshot["items"]),
        "chunk_count": len(chunks),
        "items": [
            {
                key: value
                for key, value in item.items()
                if key not in {"attachments"} or not isinstance(value, list)
            }
            | {
                "attachments": [
                    {k: v for k, v in attachment.items() if k != "fulltext"}
                    for attachment in item.get("attachments", []) or []
                ]
            }
            for item in snapshot["items"]
        ],
    }
    outputs = {
        name: str(output_dir / filename)
        for name, filename in OUTPUT_FILENAMES.items()
        if name in {"zotero_rag_corpus_manifest_json", "zotero_rag_chunks_jsonl", "zotero_rag_refresh_ledger_jsonl"}
    }
    _write_json(Path(outputs["zotero_rag_corpus_manifest_json"]), manifest)
    _write_jsonl(Path(outputs["zotero_rag_chunks_jsonl"]), chunks)
    ledger = {
        "timestamp_utc": _utc_now(),
        "event": "refresh_zotero_rag_corpus",
        "collection_key": collection_key,
        "item_count": len(snapshot["items"]),
        "chunk_count": len(chunks),
        "canonical_manifest_modified": False,
        "outputs": outputs,
    }
    _append_jsonl(Path(outputs["zotero_rag_refresh_ledger_jsonl"]), ledger)
    return {
        "status": "ok",
        "summary": {
            "collection_key": collection_key,
            "item_count": len(snapshot["items"]),
            "chunk_count": len(chunks),
            "fulltext_chunk_count": sum(1 for chunk in chunks if chunk["chunk_type"] == "fulltext"),
            "metadata_chunk_count": sum(1 for chunk in chunks if chunk["chunk_type"] == "metadata"),
            "canonical_manifest_modified": False,
        },
        "outputs": outputs,
    }


def _context_query_terms(context: dict[str, Any], query: str | None = None) -> list[str]:
    parts: list[str] = []
    for key in ["natural_language_goal", "target", "polymer_family", "architecture", "repeat_unit_smiles"]:
        value = context.get(key)
        if value:
            parts.append(str(value))
    for key in ["design_variables", "synonyms", "properties"]:
        value = context.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    if query:
        parts.append(query)
    text = " ".join(parts)
    if any(term in text.lower() for term in ["pe-lcb", "pe_lcb", "branched polyethylene", "branch"]):
        parts.extend(
            [
                "long chain branched polyethylene branch topology branch length branch distribution comb pom-pom",
                "LCB topology branch position pattern",
            ]
        )
    if any(term in text.lower() for term in ["automation", "md", "polymer"]):
        parts.extend(["automated polymer molecular dynamics RadonPy SPACIER high-throughput"])
    return _tokens(" ".join(parts))


def _score_chunk(chunk: dict[str, Any], terms: list[str]) -> float:
    haystack = " ".join(
        [
            str(chunk.get("title", "")),
            str(chunk.get("text", "")),
            " ".join(chunk.get("tags", []) or []),
        ]
    ).lower()
    if not terms:
        return 0.0
    score = 0.0
    unique_terms = sorted(set(term for term in terms if len(term) >= 2))
    for term in unique_terms:
        count = haystack.count(term)
        if count:
            score += min(count, 4)
    for phrase in ["long chain branched", "branch topology", "polymer md", "radonpy", "spacier"]:
        if phrase in haystack:
            score += 2.0
    if chunk.get("chunk_type") == "fulltext":
        score += 0.25
    return score


def retrieve_design_prior_evidence(
    context_json: str | dict[str, Any],
    *,
    query: str | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_k: int = 8,
) -> dict[str, Any]:
    """Return top lexical RAG chunks for a polymer-design context."""

    context = json.loads(context_json) if isinstance(context_json, str) else dict(context_json)
    output_dir = Path(output_dir)
    chunks = _read_jsonl(output_dir / OUTPUT_FILENAMES["zotero_rag_chunks_jsonl"])
    if not chunks:
        return {
            "status": "not_found",
            "reason": "zotero RAG chunks are missing; refresh corpus first",
            "records": [],
            "retrieved_count": 0,
        }
    terms = _context_query_terms(context, query)
    required_dois = sorted(
        {
            _normalize_doi(value)
            for value in (context.get("required_dois", []) or [])
            if _normalize_doi(value)
        }
    )
    scored: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_doi = _normalize_doi(chunk.get("DOI"))
        if required_dois and chunk_doi not in required_dois:
            continue
        score = _score_chunk(chunk, terms)
        source_role = PELCB_DESIGN_SOURCE_ROLES.get(chunk_doi, {})
        source_text = f"{chunk.get('title', '')} {chunk.get('text', '')}".lower()
        for phrase in source_role.get("retrieval_terms", []) or []:
            if str(phrase).lower() in source_text:
                score += 8.0
        if score > 0:
            scored.append({**chunk, "score": score})
    scored.sort(key=lambda row: (-float(row.get("score", 0.0)), str(row.get("chunk_id", ""))))
    selected: list[dict[str, Any]] = []
    if required_dois:
        # Preserve at least one evidence chunk for every required paper before
        # filling the remaining top-k slots. This prevents one long PDF from
        # crowding all other sources out of a lexical ranking.
        for doi in required_dois:
            match = next((row for row in scored if _normalize_doi(row.get("DOI")) == doi), None)
            if match is not None:
                selected.append(match)
    selected_ids = {str(row.get("chunk_id")) for row in selected}
    for row in scored:
        if len(selected) >= max(1, int(top_k)):
            break
        if str(row.get("chunk_id")) not in selected_ids:
            selected.append(row)
            selected_ids.add(str(row.get("chunk_id")))
    retrieved_dois = sorted({_normalize_doi(row.get("DOI")) for row in selected if _normalize_doi(row.get("DOI"))})
    missing_dois = sorted(set(required_dois) - set(retrieved_dois))
    return {
        "status": "review_required" if missing_dois else "ok",
        "query": query or " ".join(terms[:24]),
        "retrieved_count": len(selected),
        "required_dois": required_dois,
        "retrieved_dois": retrieved_dois,
        "missing_dois": missing_dois,
        "records": _json_safe(selected),
        "corpus_path": str(output_dir / OUTPUT_FILENAMES["zotero_rag_chunks_jsonl"]),
    }


def _adapter_status(context: dict[str, Any]) -> tuple[str, list[str]]:
    architecture = str(context.get("architecture") or "")
    polymer_family = str(context.get("polymer_family") or "")
    adapter = get_adapter(architecture, polymer_family) if architecture else None
    if adapter is None:
        return "", ["unregistered_adapter"]
    return adapter.adapter_id, []


def _prior_from_record(record: dict[str, Any]) -> dict[str, Any]:
    text = str(record.get("text", ""))
    lower = text.lower()
    doi = _normalize_doi(record.get("DOI"))
    source_role = PELCB_DESIGN_SOURCE_ROLES.get(doi)
    if source_role:
        prior_type = str(source_role["prior_type"])
        design_variable = str(source_role["design_variable"])
        suggested = str(source_role["suggested_range_or_category"])
        claim_boundary = str(source_role["claim_boundary"])
    elif any(term in lower for term in ["branch", "long-chain", "long chain", "pom-pom", "comb"]):
        prior_type = "topology_grammar"
        design_variable = "branch_topology_and_position"
        suggested = "branch-point, branch-length, comb/pom-pom/clustered placement categories within registered adapter limits"
        claim_boundary = "Generic branching context; review before assigning exact design values."
    elif any(term in lower for term in ["radonpy", "spacier", "automated", "high-throughput"]):
        prior_type = "automation_context"
        design_variable = "workflow_context"
        suggested = "automated all-atom polymer-MD context for design and property-evidence workflows"
        claim_boundary = "Workflow context only; not a molecular-design parameter source."
    elif any(term in lower for term in ["density", "diffusion", "thermal", "property"]):
        prior_type = "rule_profile_context"
        design_variable = "property_evidence"
        suggested = "candidate public/control evidence source for a frozen rule_profile"
        claim_boundary = "Property context only; not a molecular-design parameter source."
    else:
        prior_type = "literature_context"
        design_variable = "design_context"
        suggested = "supporting literature context; review before use as an active design prior"
        claim_boundary = "No parameter-level support was extracted from this chunk."
    return {
        "prior_type": prior_type,
        "polymer_family": "",
        "design_variable": design_variable,
        "suggested_range_or_category": suggested,
        "source_doi": doi,
        "claim_boundary": claim_boundary,
        "evidence_chunk_ids": [record.get("chunk_id", "")],
        "evidence_item_keys": [record.get("zotero_item_key", "")],
        "rationale": text[:600],
        "confidence": 0.72 if prior_type != "literature_context" else 0.55,
        "review_flags": [] if prior_type != "literature_context" else ["generic_context_only"],
    }


def propose_design_prior_profile_from_rag(
    context_json: str | dict[str, Any],
    retrieval_records: str | list[dict[str, Any]],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Create a guarded design-prior profile proposal from retrieved chunks."""

    context = json.loads(context_json) if isinstance(context_json, str) else dict(context_json)
    records = json.loads(retrieval_records) if isinstance(retrieval_records, str) else list(retrieval_records)
    output_dir = Path(output_dir)
    adapter_id, review_flags = _adapter_status(context)
    if not records:
        review_flags.append("no_evidence_chunks")
    required_dois = sorted(
        {
            _normalize_doi(value)
            for value in (context.get("required_dois", []) or [])
            if _normalize_doi(value)
        }
    )
    retrieved_dois = sorted({_normalize_doi(row.get("DOI")) for row in records if _normalize_doi(row.get("DOI"))})
    missing_dois = sorted(set(required_dois) - set(retrieved_dois))
    if missing_dois:
        review_flags.append("required_literature_missing")
    prior_records = [_prior_from_record(record) for record in records]
    for prior in prior_records:
        prior["polymer_family"] = str(context.get("polymer_family") or "")
    profile_basis = {
        "polymer_family": context.get("polymer_family", ""),
        "architecture": context.get("architecture", ""),
        "target": context.get("target") or context.get("natural_language_goal") or "",
        "evidence_chunk_ids": sorted(
            {
                str(chunk_id)
                for prior in prior_records
                for chunk_id in prior.get("evidence_chunk_ids", [])
                if chunk_id
            }
        ),
    }
    profile_id = "zotero_rag_" + hashlib.sha256(
        json.dumps(_json_safe(profile_basis), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    activation_status = "eligible_for_machine_freeze" if prior_records and not review_flags else "review_required"
    profile = {
        "schema_version": "zotero_design_prior_profile.v1",
        "profile_id": profile_id,
        "created_at": _utc_now(),
        "source_layer": "zotero_rag",
        "polymer_family": context.get("polymer_family", ""),
        "architecture": context.get("architecture", ""),
        "adapter_id": adapter_id,
        "target": profile_basis["target"],
        "required_dois": required_dois,
        "retrieved_dois": retrieved_dois,
        "missing_dois": missing_dois,
        "design_value_provenance": context.get("design_value_provenance", {}),
        "retrieval_record_count": len(records),
        "design_prior_records": prior_records,
        "evidence_chunk_ids": profile_basis["evidence_chunk_ids"],
        "review_flags": sorted(set(review_flags)),
        "activation_status": activation_status,
        "canonical_manifest_modified": False,
        "rule_thresholds_modified": False,
    }
    profile_dir = output_dir / "design_prior_profiles"
    _write_json(profile_dir / f"{profile_id}.json", profile)
    _write_jsonl(output_dir / OUTPUT_FILENAMES["zotero_design_prior_extractions_jsonl"], prior_records)
    return {"status": "ok", "profile": profile, "affected_path": str(profile_dir / f"{profile_id}.json")}


def activate_design_prior_profile(
    profile_json: str | dict[str, Any],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    actor: str = "codex",
    reason: str = "",
) -> dict[str, Any]:
    profile = json.loads(profile_json) if isinstance(profile_json, str) else dict(profile_json)
    output_dir = Path(output_dir)
    if profile.get("activation_status") != "eligible_for_machine_freeze":
        return {
            "status": "rejected",
            "reason": "design-prior profile is not eligible_for_machine_freeze",
            "profile_id": profile.get("profile_id"),
        }
    profile = {**profile, "activated_at": _utc_now(), "activated_by": actor, "activation_reason": reason}
    active_path = output_dir / "active_design_prior_profiles" / f"{profile['profile_id']}.json"
    _write_json(active_path, profile)
    _append_jsonl(
        output_dir / "design_prior_freeze_ledger.jsonl",
        {
            "timestamp_utc": _utc_now(),
            "event": "activate_design_prior_profile",
            "profile_id": profile["profile_id"],
            "actor": actor,
            "reason": reason,
            "affected_path": str(active_path),
            "canonical_manifest_modified": False,
            "rule_thresholds_modified": False,
        },
    )
    return {"status": "ok", "profile": profile, "affected_path": str(active_path)}


def get_design_prior_profile(profile_id: str, *, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir = Path(output_dir)
    for path in [
        output_dir / "active_design_prior_profiles" / f"{profile_id}.json",
        output_dir / "design_prior_profiles" / f"{profile_id}.json",
    ]:
        if path.exists():
            return {"status": "ok", "profile": _read_json(path, {}), "path": str(path)}
    return {"status": "not_found", "profile_id": profile_id}


def get_zotero_rag_corpus_summary(*, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir = Path(output_dir)
    manifest = _read_json(output_dir / OUTPUT_FILENAMES["zotero_rag_corpus_manifest_json"], {})
    if not manifest:
        return {"status": "not_found", "path": str(output_dir / OUTPUT_FILENAMES["zotero_rag_corpus_manifest_json"])}
    return {
        "status": "ok",
        "summary": {
            "collection_key": manifest.get("collection_key"),
            "collection_name": manifest.get("collection_name"),
            "item_count": manifest.get("item_count", 0),
            "chunk_count": manifest.get("chunk_count", 0),
            "created_at": manifest.get("created_at"),
        },
        "manifest_path": str(output_dir / OUTPUT_FILENAMES["zotero_rag_corpus_manifest_json"]),
        "chunks_path": str(output_dir / OUTPUT_FILENAMES["zotero_rag_chunks_jsonl"]),
    }


def build_pelcb100_rag_design_origin_overlay(
    profile_json: str | dict[str, Any],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR / "pelcb100_design_origin_rag_overlay",
) -> dict[str, Any]:
    """Attach a RAG profile to the existing 100-row design-origin table."""

    profile = json.loads(profile_json) if isinstance(profile_json, str) else dict(profile_json)
    output_dir = Path(output_dir)
    base_dir = output_dir / "base_design_origin"
    base = build_design_origin_outputs(output_dir=base_dir)
    table_path = Path(base["outputs"]["case_design_origin_table_csv"])
    rows: list[dict[str, Any]] = []
    with table_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(dict(row))
    chunk_ids = ";".join(str(item) for item in profile.get("evidence_chunk_ids", []) or [])
    out_rows = [
        {
            **row,
            "rag_profile_id": profile.get("profile_id", ""),
            "rag_activation_status": profile.get("activation_status", ""),
            "rag_evidence_chunk_ids": chunk_ids,
            "rag_source_layer": "zotero_rag",
        }
        for row in rows
    ]
    overlay_path = output_dir / OUTPUT_FILENAMES["case_design_origin_table_rag_overlay_csv"]
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    with overlay_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)

    prior_summary = [
        "# Zotero-RAG design-prior summary",
        "",
        f"- Profile: `{profile.get('profile_id', '')}`",
        f"- Polymer family: `{profile.get('polymer_family', '')}`",
        f"- Architecture: `{profile.get('architecture', '')}`",
        f"- Activation status: `{profile.get('activation_status', '')}`",
        f"- Evidence chunks: `{len(profile.get('evidence_chunk_ids', []) or [])}`",
        f"- Linked PE-LCB100 cases: `{len(out_rows)}`",
        "- Canonical manifest modified: `false`",
        "",
        "This overlay records retrieval-grounded design-prior evidence. It does not modify the PE-LCB100 manifest or launch MD.",
    ]
    summary_path = output_dir / OUTPUT_FILENAMES["zotero_design_prior_summary_md"]
    summary_path.write_text("\n".join(prior_summary) + "\n", encoding="utf-8")
    ledger_path = output_dir / OUTPUT_FILENAMES["rag_design_origin_ledger_jsonl"]
    _append_jsonl(
        ledger_path,
        {
            "timestamp_utc": _utc_now(),
            "event": "build_pelcb100_rag_design_origin_overlay",
            "profile_id": profile.get("profile_id", ""),
            "case_count": len(out_rows),
            "canonical_manifest_modified": False,
            "affected_path": str(overlay_path),
        },
    )
    return {
        "status": "ok",
        "summary": {
            "case_count": len(out_rows),
            "rag_profile_id": profile.get("profile_id", ""),
            "canonical_manifest_modified": False,
        },
        "outputs": {
            "case_design_origin_table_rag_overlay_csv": str(overlay_path),
            "zotero_design_prior_summary_md": str(summary_path),
            "rag_design_origin_ledger_jsonl": str(ledger_path),
        },
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zotero-dir", default=str(DEFAULT_ZOTERO_DIR))
    parser.add_argument("--collection-key", default=DEFAULT_COLLECTION_KEY)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--include-doi",
        action="append",
        default=None,
        help="Include an exact DOI from anywhere in the Zotero library; may be repeated.",
    )
    parser.add_argument(
        "--query",
        default="branched polyethylene branch number branch length branch distribution star topology",
    )
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--build-overlay", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    required_dois = tuple(args.include_doi or DEFAULT_PELCB_DESIGN_DOIS)
    result = build_zotero_rag_outputs(
        zotero_dir=args.zotero_dir,
        collection_key=args.collection_key,
        output_dir=output_dir,
        include_dois=required_dois,
    )
    context = {
        "polymer_family": "PE",
        "architecture": "pe_lcb",
        "target": args.query,
        "design_variables": ["branch_points", "branch_carbons", "branch_position_pattern"],
        "required_dois": list(required_dois),
        "design_value_provenance": {
            "literature_role": "defines the branch topology, length/count and placement/distribution axes",
            "project_role": (
                "defines the discrete bp/bc/pd grid under the approximately C100 carbon budget "
                "and registered builder constraints"
            ),
        },
    }
    retrieval = retrieve_design_prior_evidence(context, query=args.query, output_dir=output_dir, top_k=args.top_k)
    profile = propose_design_prior_profile_from_rag(context, retrieval.get("records", []), output_dir=output_dir)
    payload: dict[str, Any] = {"corpus": result, "retrieval": retrieval, "profile": profile}
    if args.build_overlay:
        payload["overlay"] = build_pelcb100_rag_design_origin_overlay(profile["profile"], output_dir=output_dir / "pelcb100_design_origin_rag_overlay")
    print(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
