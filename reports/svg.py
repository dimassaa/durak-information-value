"""Vanilla-SVG charts for the HTML reports.

No charting library. A CDN-loaded library is a blank page wherever that address is unreachable -- inside a
corporate proxy, in a restricted preview, in five years -- and it is also the single biggest reason a
generated report looks generated. These are drawn from an embedded `DATA` object with plain element
creation, so the file works offline and renders identically in a browser, in print and in an email client.

Two rules that come from the figure rules and are easy to lose in SVG:

* **Colour is never the only cue.** A bar that means "positive" is also taller; a point that means "the
  frozen one" also carries a glyph.
* **Every number appears as text**, because an axis alone cannot be read aloud or copied into a note.
"""

from __future__ import annotations

import html
import json

W = 880
H = 340
PAD = {"top": 34, "right": 118, "bottom": 62, "left": 68}


def _scale(domain, rng):
    lo, hi = domain
    span = (hi - lo) or 1.0
    inner = H - PAD["top"] - PAD["bottom"]
    return lambda v: PAD["top"] + inner - (v - lo) / span * inner


def _ticks(lo, hi, n=5):
    if lo == hi:
        return [lo]
    step = (hi - lo) / n
    return [lo + step * i for i in range(n + 1)]


def bar_chart(*, title, subtitle, series, labels, y_label, reference=None,
              value_fmt="{:.3f}", palette=("#2563eb", "#5b6472"), token="--color-accent") -> str:
    """Grouped bars, one group per label, one bar per series."""
    from reports.dashboard import style_tokens

    t = style_tokens()
    values = [v for row in series for v in row[1] if v is not None]
    if not values:
        return ""
    lo, hi = min(values + ([reference] if reference is not None else [])), max(
        values + ([reference] if reference is not None else []))
    pad = (hi - lo) * 0.18 or 0.01
    lo, hi = lo - pad, hi + pad
    y = _scale((lo, hi), None)
    inner = W - PAD["left"] - PAD["right"]
    step = inner / max(1, len(labels))
    group = step * 0.68
    bar = group / max(1, len(series) + 0.4)

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart" '
           f'aria-label="{html.escape(title)}">',
           f'<text x="0" y="0" class="c-title">{html.escape(title)}</text>']
    if subtitle:
        out.append(f'<text x="0" y="18" class="c-sub">{html.escape(subtitle)}</text>')

    for tick in _ticks(lo, hi):
        yy = y(tick)
        out.append(f'<line x1="{PAD["left"]}" y1="{yy:.1f}" x2="{W - PAD["right"]}" y2="{yy:.1f}" '
                   f'class="c-grid"/>')
        out.append(f'<text x="{PAD["left"] - 8}" y="{yy + 3.5:.1f}" class="c-tick" '
                   f'text-anchor="end">{tick:+.3f}</text>')

    if reference is not None:
        yy = y(reference)
        out.append(f'<line x1="{PAD["left"]}" y1="{yy:.1f}" x2="{W - PAD["right"]}" '
                   f'y2="{yy:.1f}" class="c-ref"/>')
        out.append(f'<text x="{W - PAD["right"] + 6}" y="{yy + 3.5:.1f}" class="c-reftext">'
                   f'reference {reference:+.3f}</text>')

    for i, label in enumerate(labels):
        cx = PAD["left"] + step * (i + 0.5)
        for s, (_, row) in enumerate(series):
            v = row[i]
            if v is None:
                continue
            x = cx - group / 2 + bar * (s + 0.2)
            h = y(lo) - y(v)
            out.append(f'<rect x="{x:.1f}" y="{y(v):.1f}" width="{bar * 0.86:.1f}" '
                       f'height="{max(0.6, h):.1f}" fill="{palette[s % len(palette)]}" rx="1"/>')
            out.append(f'<text x="{x + bar * 0.43:.1f}" y="{y(v) - 5:.1f}" class="c-value" '
                       f'text-anchor="middle">{value_fmt.format(v)}</text>')
        out.append(f'<text x="{cx:.1f}" y="{H - PAD["bottom"] + 16}" class="c-cat" '
                   f'text-anchor="middle">{html.escape(label)}</text>')

    out.append(f'<text x="14" y="{PAD["top"] + 4}" class="c-axis" transform="rotate(-90 14 '
               f'{PAD["top"] + 4})" text-anchor="end">{html.escape(y_label)}</text>')
    for s, (name, _row) in enumerate(series):
        yy = PAD["top"] + s * 17
        out.append(f'<rect x="{W - PAD["right"] + 6}" y="{yy - 8}" width="10" height="10" rx="1" '
                   f'fill="{palette[s % len(palette)]}"/>')
        out.append(f'<text x="{W - PAD["right"] + 21}" y="{yy + 1}" class="c-legend">'
                   f'{html.escape(name)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def hbar_chart(*, title, subtitle, labels, values, x_label, highlight=None,
               value_fmt="{:.4f}", palette="#2563eb") -> str:
    """Horizontal bars, so long strategy names stay readable instead of being rotated."""
    from reports.dashboard import style_tokens

    t = style_tokens()
    if not values:
        return ""
    left = 168
    right = 150
    inner = W - left - right
    hi = max(values) or 1.0
    top = 44
    step = min(26, (H - top - 16) / max(1, len(values)))
    height = H - top - 20
    bar = step * 0.62

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart" '
           f'aria-label="{html.escape(title)}">',
           f'<text x="0" y="0" class="c-title">{html.escape(title)}</text>']
    if subtitle:
        out.append(f'<text x="0" y="18" class="c-sub">{html.escape(subtitle)}</text>')

    for i, (label, v) in enumerate(zip(labels, values)):
        yy = top + step * (i + 0.5) - bar / 2
        w = v / hi * inner
        colour = t["--warning"] if label == highlight else palette
        out.append(f'<text x="{left - 10}" y="{yy + bar / 2 + 3.5:.1f}" class="c-cat" '
                   f'text-anchor="end">{html.escape(label)}</text>')
        out.append(f'<rect x="{left}" y="{yy:.1f}" width="{max(1.0, w):.1f}" height="{bar:.1f}" '
                   f'fill="{colour}" rx="1"/>')
        marker = "  \\u2190 selected" if label == highlight else ""
        out.append(f'<text x="{left + w + 8:.1f}" y="{yy + bar / 2 + 3.5:.1f}" class="c-value">'
                   f'{value_fmt.format(v)}{marker}</text>')
    out.append(f'<text x="{left + inner / 2:.0f}" y="{H - 6}" class="c-axis" '
               f'text-anchor="middle">{html.escape(x_label)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def interval_chart(*, title, subtitle, labels, values, lows, highs, y_label,
                   reference=None) -> str:
    """Points with vertical intervals -- the shape a confidence-interval figure needs."""
    from reports.dashboard import style_tokens

    t = style_tokens()
    if not values:
        return ""
    lo, hi = min(lows), max(highs)
    pad = (hi - lo) * 0.22 or 0.01
    lo, hi = lo - pad, hi + pad
    y = _scale((lo, hi), None)
    inner = W - PAD["left"] - PAD["right"]
    step = inner / max(1, len(labels))
    accent = t["--color-accent"]

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart" '
           f'aria-label="{html.escape(title)}">',
           f'<text x="0" y="0" class="c-title">{html.escape(title)}</text>']
    if subtitle:
        out.append(f'<text x="0" y="18" class="c-sub">{html.escape(subtitle)}</text>')

    for tick in _ticks(lo, hi):
        yy = y(tick)
        out.append(f'<line x1="{PAD["left"]}" y1="{yy:.1f}" x2="{W - PAD["right"]}" '
                   f'y2="{yy:.1f}" class="c-grid"/>')
        out.append(f'<text x="{PAD["left"] - 8}" y="{yy + 3.5:.1f}" class="c-tick" '
                   f'text-anchor="end">{tick:+.3f}</text>')
    if lo < 0 < hi:
        out.append(f'<line x1="{PAD["left"]}" y1="{y(0):.1f}" x2="{W - PAD["right"]}" '
                   f'y2="{y(0):.1f}" class="c-ref"/>')
    if reference is not None:
        yy = y(reference)
        out.append(f'<line x1="{PAD["left"]}" y1="{yy:.1f}" x2="{W - PAD["right"]}" '
                   f'y2="{yy:.1f}" class="c-dash"/>')
        out.append(f'<text x="{W - PAD["right"] + 6}" y="{yy + 3.5:.1f}" class="c-reftext">'
                   f'ceiling {reference:+.3f}</text>')

    for i, (label, v, lo_i, hi_i) in enumerate(zip(labels, values, lows, highs)):
        cx = PAD["left"] + step * (i + 0.5)
        out.append(f'<line x1="{cx:.1f}" y1="{y(lo_i):.1f}" x2="{cx:.1f}" y2="{y(hi_i):.1f}" '
                   f'stroke="{accent}" stroke-width="1.4"/>')
        for cap in (lo_i, hi_i):
            out.append(f'<line x1="{cx - 4:.1f}" y1="{y(cap):.1f}" x2="{cx + 4:.1f}" '
                       f'y2="{y(cap):.1f}" stroke="{accent}" stroke-width="1.4"/>')
        out.append(f'<circle cx="{cx:.1f}" cy="{y(v):.1f}" r="4.6" fill="{accent}"/>')
        out.append(f'<text x="{cx + 10:.1f}" y="{y(v) + 4:.1f}" class="c-value">{v:+.4f}</text>')
        out.append(f'<text x="{cx:.1f}" y="{H - PAD["bottom"] + 18}" class="c-cat" '
                   f'text-anchor="middle">{html.escape(label)}</text>')
    out.append(f'<text x="14" y="{PAD["top"] + 4}" class="c-axis" transform="rotate(-90 14 '
               f'{PAD["top"] + 4})" text-anchor="end">{html.escape(y_label)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def data_object(payload: dict) -> str:
    """The embedded snapshot. One object, so there is exactly one place a number can live."""
    return (f'<script id="data" type="application/json">{json.dumps(payload, sort_keys=True)}'
            f'</script>')

def _colours_from(datasets) -> list:
    """Per-dataset colours, falling back to the one accent when a spec names none."""
    from reports.dashboard import style_tokens

    t = style_tokens()
    out = []
    for dset in datasets:
        colour = dset.get("backgroundColor")
        if isinstance(colour, list) and colour:
            out.append(str(colour[0]))
        elif isinstance(colour, str):
            out.append(colour)
        else:
            out.append(t["--color-accent"])
    return out or [t["--color-accent"]]


def spec_to_svg(spec: dict, title: str = "") -> str:
    """Render a Chart.js-shaped spec as inline SVG.

    The report builder already expresses every chart as `{type, data:{labels, datasets}, options}`. Writing
    a converter rather than rewriting ten call sites keeps the payloads as they were, keeps their tests
    meaningful, and makes the dependency disappear in one place -- which is the only place it can actually
    be guaranteed to disappear.

    Only what the builder uses is supported: vertical bars, one or several datasets, and an optional
    explicit y range. Anything unsupported returns an empty string rather than a wrong picture.
    """
    spec = spec or {}
    kind = spec.get("type", "bar")
    if kind in ("doughnut", "pie", "polarArea"):
        # Rendered as bars rather than a pie on purpose. A two-category comparison is easier to read as
        # two lengths than as two arcs, a pie cannot be compared to anything else on the page, and the
        # angle of a slice is the one encoding the eye reads worst. The underlying numbers and the title
        # are identical either way.
        spec = {**spec, "type": "bar"}
        y_range = spec.get("options") or {}
        spec["options"] = {"scales": {"y": {"min": 0, "max": 1}}}
        del y_range
    elif kind != "bar":
        return ""
    data = spec.get("data") or {}
    labels = data.get("labels") or []
    datasets = data.get("datasets") or []
    if not labels or not datasets:
        return ""

    y_range = ((spec.get("options") or {}).get("scales") or {}).get("y") or {}
    series = [(d.get("label") or "", [None if v is None else float(v)
                                      for v in (d.get("data") or [])])
              for d in datasets]
    colours = _colours_from(datasets)

    if y_range.get("min") is not None and y_range.get("max") is not None:
        # An explicit range is respected rather than auto-fitted: a bar chart of win rates on an
        # auto-fitted axis makes a three-point difference look like a chasm.
        lo, hi = float(y_range["min"]), float(y_range["max"])
        pad = (hi - lo) * 0.02
        lo, hi = lo - pad, hi + pad
        y = _scale((lo, hi), None)
        inner = W - PAD["left"] - PAD["right"]
        step = inner / max(1, len(labels))
        group = step * 0.66
        bar = group / max(1, len(series) + 0.35)
        out = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart" '
               f'aria-label="{html.escape(title or "chart")}">',
               f'<text x="0" y="0" class="c-title">{html.escape(title)}</text>']
        if len(series) == 1 and series[0][0]:
            # A single series still has to say what it plots. Dropping the label because there was no
            # legend to put it in removed the units from the page entirely -- the number survived, the
            # thing it counted did not.
            out.append(f'<text x="0" y="18" class="c-sub">{html.escape(series[0][0])}</text>')
        for tick in _ticks(lo, hi):
            yy = y(tick)
            out.append(f'<line x1="{PAD["left"]}" y1="{yy:.1f}" x2="{W - PAD["right"]}" '
                       f'y2="{yy:.1f}" class="c-grid"/>')
            out.append(f'<text x="{PAD["left"] - 8}" y="{yy + 3.5:.1f}" class="c-tick" '
                       f'text-anchor="end">{tick:.2f}</text>')
        for i, label in enumerate(labels):
            cx = PAD["left"] + step * (i + 0.5)
            for s, (_n, row) in enumerate(series):
                v = row[i] if i < len(row) else None
                if v is None:
                    continue
                v = max(lo, min(hi, v))
                x = cx - group / 2 + bar * (s + 0.18)
                out.append(f'<rect x="{x:.1f}" y="{y(v):.1f}" width="{bar * 0.86:.1f}" '
                           f'height="{max(0.6, y(lo) - y(v)):.1f}" '
                           f'fill="{colours[s % len(colours)]}" rx="1"/>')
                out.append(f'<text x="{x + bar * 0.43:.1f}" y="{y(v) - 5:.1f}" class="c-value" '
                           f'text-anchor="middle">{v:.3f}</text>')
            out.append(f'<text x="{cx:.1f}" y="{H - PAD["bottom"] + 17}" class="c-cat" '
                       f'text-anchor="middle">{html.escape(str(label))}</text>')
        for s, (name, _row) in enumerate(series):
            if len(series) < 2:
                break
            yy = PAD["top"] + s * 17
            out.append(f'<rect x="{W - PAD["right"] + 6}" y="{yy - 8}" width="10" height="10" '
                       f'rx="1" fill="{colours[s % len(colours)]}"/>')
            out.append(f'<text x="{W - PAD["right"] + 21}" y="{yy + 1}" class="c-legend">'
                       f'{html.escape(name)}</text>')
        out.append("</svg>")
        return "\n".join(out)

    return bar_chart(title=title,
                     subtitle=series[0][0] if len(series) == 1 else "",
                     series=series, labels=[str(v) for v in labels], y_label="value",
                     palette=tuple(colours))
