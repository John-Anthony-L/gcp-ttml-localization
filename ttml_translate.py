from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Tuple
import xml.etree.ElementTree as ET


# Known namespaces used in the provided TTML
NS_TT = "http://www.w3.org/ns/ttml"
NS_TTS = "http://www.w3.org/ns/ttml#styling"
NS_TTM = "http://www.w3.org/ns/ttml#metadata"
NS_TTP = "http://www.w3.org/ns/ttml#parameter"
NS_EBUTTS = "urn:ebu:tt:style"
NS_TTVA = "http://skynav.com/ns/ttv/annotations"


def _register_namespaces():
    ET.register_namespace("", NS_TT)
    ET.register_namespace("tts", NS_TTS)
    ET.register_namespace("ttm", NS_TTM)
    ET.register_namespace("ttp", NS_TTP)
    ET.register_namespace("ebutts", NS_EBUTTS)
    ET.register_namespace("ttva", NS_TTVA)


def _q(tag: str) -> str:
    return f"{{{NS_TT}}}{tag}"


def collect_span_lines(p_elem: ET.Element) -> List[ET.Element]:
    """Collect direct child <span> elements of a <p> in order.

    This matches the sample file where each visual line is a <span> separated by <br/>.
    """
    return [ch for ch in list(p_elem) if ch.tag == _q("span")]


def translate_ttml(
    input_path: str,
    translate_fn: Callable[[List[str], str], List[str]],
    target_language: str,
) -> Tuple[ET.ElementTree, int]:
    """Translate TTML subtitle text while preserving structure.

    - Only text inside <span> children of <p> is translated.
    - Styles, regions, timing, and attributes are unchanged.
    - Returns the modified ElementTree and a count of translated lines.
    """
    _register_namespaces()
    tree = ET.parse(input_path)
    root = tree.getroot()

    total_lines = 0
    # Iterate over all <p> elements
    for p in root.findall(f".//{_q('p')}"):
        spans = collect_span_lines(p)
        if not spans:
            # Fallback: translate p.text if present
            if p.text and p.text.strip():
                src_lines = [p.text]
                tgt_lines = translate_fn(src_lines, target_language)
                if tgt_lines and len(tgt_lines) == 1:
                    p.text = tgt_lines[0]
                    total_lines += 1
            continue

        src_lines = [(sp.text or "") for sp in spans]
        if any(s.strip() for s in src_lines):
            tgt_lines = translate_fn(src_lines, target_language)
            if len(tgt_lines) != len(spans):
                # Do not corrupt; skip this <p> if mismatch
                continue
            for sp, txt in zip(spans, tgt_lines):
                sp.text = txt
            total_lines += len(spans)

    return tree, total_lines


def write_ttml(tree: ET.ElementTree, output_path: str) -> None:
    _register_namespaces()
    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
