from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Tuple, Sequence
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


def collect_line_nodes(p_elem: ET.Element) -> List[Tuple[ET.Element, str]]:
    """Collect all display lines under a <p> with robust handling of <span>, <br/>, and text tails.

    Returns a list of (node, attr) pairs where attr is either 'text' or 'tail'.
    We treat each direct <span> as a line. Additionally, if a <br/> is followed by
    tail text (not wrapped in <span>), we treat that tail as a line.
    If the <p> has leading text (p.text) and no spans exist, that is treated as a line.
    """
    lines: List[Tuple[ET.Element, str]] = []

    has_spans = any(ch.tag == _q("span") for ch in list(p_elem))

    # Leading text directly on <p> (rare in our inputs). Consider it a line when no spans exist.
    if not has_spans and (p_elem.text and p_elem.text.strip()):
        lines.append((p_elem, "text"))

    children = list(p_elem)
    for idx, ch in enumerate(children):
        if ch.tag == _q("span"):
            # Standard case: each span is a separate line
            lines.append((ch, "text"))
            continue
        # Handle <br/> followed by tail text as a separate line when not immediately followed by a span
        if ch.tag == _q("br"):
            # If next sibling is a span, that span will be captured as its own line
            if ch.tail and ch.tail.strip():
                next_is_span = (idx + 1 < len(children) and children[idx + 1].tag == _q("span"))
                if not next_is_span:
                    lines.append((ch, "tail"))

    return lines


def translate_ttml(
    input_path: str,
    translate_fn: Callable[[List[str], str], List[str]],
    target_language: str,
) -> Tuple[ET.ElementTree, int]:
    """Translate TTML subtitle text while preserving structure.

    Batched translation across the entire document to minimize API calls and
    preserve broader context. Only text inside <span> children of <p> is
    translated. Styles, regions, timing, and attributes are unchanged.

    Returns the modified ElementTree and a count of translated lines.
    """
    _register_namespaces()
    tree = ET.parse(input_path)
    root = tree.getroot()

    # Collect all line nodes in document order
    line_nodes: List[Tuple[ET.Element, str]] = []
    for p in root.findall(f".//{_q('p')}"):
        line_nodes.extend(collect_line_nodes(p))

    texts: List[str] = []
    for node, attr in line_nodes:
        val = getattr(node, attr, None)
        texts.append(val or "")

    total_lines = 0
    if texts:
        translated = translate_fn(texts, target_language)
        if len(translated) == len(line_nodes):
            for (node, attr), txt in zip(line_nodes, translated):
                setattr(node, attr, txt)
            total_lines = len(line_nodes)

    return tree, total_lines


def write_ttml(tree: ET.ElementTree, output_path: str) -> None:
    _register_namespaces()
    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
