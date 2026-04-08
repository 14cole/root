from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Segment:
    name: str
    seg_type: Optional[str]
    properties: List[str]
    x: List[float]
    y: List[float]


def parse_geometry(text: str) -> Tuple[str, List[Segment], List[List[str]], List[List[str]]]:
    lines = [ln.strip() for ln in text.splitlines()]
    title = "Geometry"
    segments: List[Segment] = []
    ibcs_entries: List[List[str]] = []
    dielectric_entries: List[List[str]] = []

    state = "segments"
    current_name: Optional[str] = None
    current_type: Optional[str] = None
    current_props: List[str] = []
    cur_x: List[float] = []
    cur_y: List[float] = []

    def flush_segment() -> None:
        if current_name is not None:
            segments.append(
                Segment(
                    name=current_name,
                    seg_type=current_type,
                    properties=current_props[:],
                    x=cur_x[:],
                    y=cur_y[:],
                )
            )

    for ln in lines:
        if not ln or ln.startswith("#"):
            continue
        low = ln.lower()
        if low.startswith("title"):
            title = ln.split(":", 1)[1].strip() or title
            continue
        if state == "segments" and low.startswith("ibcs:"):
            flush_segment()
            state = "ibcs"
            continue
        if low.startswith("dielectrics:"):
            if state == "segments":
                flush_segment()
            state = "dielectrics"
            continue

        if state == "segments":
            if low.startswith("segment:"):
                flush_segment()
                parts = ln.split(":", 1)[1].strip().split()
                if not parts:
                    current_name, current_type = "Unnamed", None
                elif len(parts) == 1:
                    current_name, current_type = parts[0], None
                else:
                    current_name, current_type = parts[0], parts[1]
                current_props = []
                cur_x.clear()
                cur_y.clear()
                continue
            if low.startswith("properties:"):
                current_props = ln.split(":", 1)[1].strip().split()
                continue

            tokens = ln.split()
            if len(tokens) != 4:
                raise ValueError(f"Geometry line must have 4 numbers, got {len(tokens)} {ln}")
            try:
                x1, y1, x2, y2 = map(float, tokens)
            except ValueError:
                raise ValueError(f"Geometry line must contain valid numbers: {ln}")
            cur_x.extend([x1, x2])
            cur_y.extend([y1, y2])
        elif state == "ibcs":
            tokens = ln.split()
            if tokens:
                ibcs_entries.append(tokens)
        elif state == "dielectrics":
            tokens = ln.split()
            if tokens:
                dielectric_entries.append(tokens)

    if state == "segments":
        flush_segment()

    return title, segments, ibcs_entries, dielectric_entries


def build_geometry_text(
    title: str,
    segments: List[Segment],
    ibcs_entries: List[List[str]],
    dielectric_entries: List[List[str]],
) -> str:
    lines: List[str] = [f"Title: {title}"]
    for seg in segments:
        if seg.seg_type:
            lines.append(f"Segment: {seg.name} {seg.seg_type}")
        else:
            lines.append(f"Segment: {seg.name}")

        props = list(seg.properties)
        if len(props) < 6:
            props.extend([""] * (6 - len(props)))
        lines.append("properties: " + " ".join(p if p is not None else "" for p in props))

        if len(seg.x) != len(seg.y) or len(seg.x) % 2 != 0:
            raise ValueError(
                f"Segment {seg.name} has mismatched or odd number of coordinates."
            )
        for i in range(0, len(seg.x), 2):
            x1, y1, x2, y2 = seg.x[i], seg.y[i], seg.x[i + 1], seg.y[i + 1]
            lines.append(f"{x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f}")

    lines.append("IBCS:")
    for row in ibcs_entries:
        lines.append(" ".join(row))
    lines.append("Dielectrics:")
    for row in dielectric_entries:
        lines.append(" ".join(row))
    return "\n".join(lines) + "\n"


def build_geometry_snapshot(
    title: str,
    segments: List[Segment],
    ibcs_entries: List[List[str]],
    dielectric_entries: List[List[str]],
) -> Dict[str, Any]:
    segments_payload = []
    for seg in segments:
        point_pairs = []
        for i in range(0, min(len(seg.x), len(seg.y)), 2):
            if i + 1 >= len(seg.x) or i + 1 >= len(seg.y):
                break
            point_pairs.append(
                {
                    "x1": seg.x[i],
                    "y1": seg.y[i],
                    "x2": seg.x[i + 1],
                    "y2": seg.y[i + 1],
                }
            )
        segments_payload.append(
            {
                "name": seg.name,
                "seg_type": seg.seg_type,
                "properties": list(seg.properties),
                "point_pairs": point_pairs,
            }
        )

    return {
        "title": title,
        "segment_count": len(segments),
        "segments": segments_payload,
        "ibcs": [list(row) for row in ibcs_entries],
        "dielectrics": [list(row) for row in dielectric_entries],
    }
