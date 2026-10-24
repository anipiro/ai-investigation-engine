"""Stable document-id conventions shared by the corpus builder and the evaluation."""

from __future__ import annotations

import zlib

from engine.ingestion.pipeline import slugify


def hotpot_doc_id(title: str) -> str:
    # crc suffix keeps ids unique when two titles slugify identically ("A.B." vs "A B").
    return f"hotpot::{slugify(title, 60)}-{zlib.crc32(title.encode('utf-8')) & 0xFFFF:04x}"


def postmortem_doc_id(company: str, n: int) -> str:
    return f"postmortems::{slugify(company, 40)}-{n:03d}"


def northwind_doc_id(filename: str) -> str:
    return f"northwind::{slugify(filename.rsplit('.', 1)[0])}"
