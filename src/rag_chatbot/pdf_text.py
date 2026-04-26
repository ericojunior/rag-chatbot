from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, Iterable, List


@dataclass(frozen=True)
class PdfPageText:
    page: int
    text: str


def extract_pdf_text(pdf_file: BinaryIO) -> List[PdfPageText]:
    """
    Extrai texto do PDF preservando o número da página (1-based).
    Tenta pdfplumber primeiro (melhor layout); cai para pypdf se necessário.
    """
    pdf_file.seek(0)
    try:
        import pdfplumber  # type: ignore

        pages: List[PdfPageText] = []
        with pdfplumber.open(pdf_file) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                txt = page.extract_text() or ""
                txt = txt.strip()
                if txt:
                    pages.append(PdfPageText(page=i, text=txt))
        return pages
    except Exception:
        pass

    pdf_file.seek(0)
    from pypdf import PdfReader  # type: ignore

    reader = PdfReader(pdf_file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        txt = (page.extract_text() or "").strip()
        if txt:
            pages.append(PdfPageText(page=i, text=txt))
    return pages

