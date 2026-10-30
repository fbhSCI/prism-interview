"""Optional, externally supplied recording fixtures are never redistributed."""
from pathlib import Path
import pytest


def read_optional_recording(path: Path) -> str:
    if not path.is_file():
        pytest.skip("Third-party interview recording fixture is excluded from the public source release")
    return path.read_text(encoding="utf-8")


def write_synthetic_pdf(path: Path, blocks: list[tuple[int, int, str]]) -> None:
    """Create deterministic text fixtures using the production pypdf dependency."""
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"):
        DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
    parts = []
    for x, y, text in blocks:
        escaped = text.replace(chr(92), chr(92)*2).replace("(", chr(92)+"(").replace(")", chr(92)+")")
        rows = escaped.splitlines()
        parts.append(f"BT /F1 12 Tf {x} {y} Td 14 TL " + " Tj T* ".join(f"({row})" for row in rows) + " Tj ET")
    stream = DecodedStreamObject()
    stream.set_data("\n".join(parts).encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(path)
