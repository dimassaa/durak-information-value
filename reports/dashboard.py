"""Deterministic standalone HTML reports.

One file, no server, no network beyond the pinned Chart.js CDN tag. Everything the page needs is
embedded, so a report can be attached to a review, archived, or opened on a machine with no access to
this project.

Determinism is the whole design constraint, and it is enforced rather than hoped for
----------------------------------------------------------------------------
Two renders of the same JSON produce byte-identical files. Three things would break that, and each is
closed explicitly:

* **The clock.** There is no `new Date()`, no `toLocaleString`, no `Intl`. The build date is a recorded
  fact read from `meta.json` -- when the dataset was produced -- not the moment someone pressed a button.
  A report whose date moves on every build cannot be diffed, and a diff that always shows a change is a
  diff nobody reads.
* **Iteration order.** Python dicts iterate in insertion order, so a document assembled from the same
  mappings in a different order would emit different bytes. Every mapping is sorted before it is
  written, and table rows keep the caller's order because row order is meaning, not an accident.
* **Floating dependencies.** The Chart.js version is pinned and written into the file. A report whose
  appearance depends on when it was opened is not a record of anything.

Escaping is applied to every value that reaches the page. A strategy name or a title containing `<` must
not become markup.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

#: Pinned and asserted. A floating CDN tag means two readers can see different charts from the same file.
CHARTJS_VERSION = "4.5.1"

#: The reference dashboard's visual language as CSS custom properties, lifted rather than invented so
#: the reports read as one set. The values are literals here on purpose: a token that reads from a theme
#: file would make the output depend on the environment it was rendered in.
#: One design system, used by the figures AND the pages, so a PNG next to the text that cites it does
#: not read as two different projects.
#:
#: It is a LIGHT palette on purpose. A dark background with saturated blue/green/red and 8px rounded
#: cards is the default look of generated dashboards, and it is exactly the thing that made the previous
#: reports look machine-made. Hairline borders, one accent, and a print-first background read as a
#: research document instead -- and GitHub renders light, so the figures also match the page they sit on.
#:
#: Colours are never the only carrier of meaning: every chart that uses the positive/negative pair also
#: marks direction with a glyph, and every figure states its numbers in text beside it.
_TOKENS = {
    # surfaces -- four steps is enough to separate page, band, card and header without a shadow
    "--bg-primary": "#ffffff",
    "--bg-card": "#ffffff",
    "--bg-header": "#f7f8fa",
    "--bg-band": "#f2f4f7",
    # hairlines, never shadows
    "--border": "#e3e6ea",
    "--border-strong": "#cbd2d9",
    # text
    "--text-primary": "#14181f",
    "--text-secondary": "#5b6472",
    "--text-muted": "#8b94a3",
    # ONE accent. Positive/negative/warning are semantic, not decorative.
    "--color-accent": "#2563eb",
    "--color-accent-soft": "#dbe6fd",
    "--positive": "#0f7b47",
    "--negative": "#b42318",
    "--warning": "#a15c07",
    # neutral series for the categorical charts (matchup matrix, per-strategy bars)
    "--series-1": "#2563eb",
    "--series-2": "#5b6472",
    "--series-3": "#8b94a3",
    "--series-4": "#a15c07",
    "--series-5": "#0f7b47",
    "--series-6": "#7c3aed",
    "--series-7": "#b42318",
    # geometry: 4px, not 8. A large radius is the other half of the template look.
    "--radius": "4px",
    "--gap": "16px",
    "--measure": "1320px",
}


def style_tokens() -> dict:
    """The CSS custom properties, in a fixed order.

    Sorted on return so that a caller embedding them cannot reintroduce an insertion-order dependency.
    """
    return dict(sorted(_TOKENS.items()))


def _e(value) -> str:
    """HTML-escape any value. Called on every value that reaches the page."""
    return html.escape(str(value), quote=True)


def _j(value) -> str:
    """Serialise to JSON for embedding, stably.

    `sort_keys` because the embedded payload is part of the file's bytes and dict order would otherwise
    leak. `<` is escaped to `<` so a value can never close the enclosing `<script>` tag.
    """
    return json.dumps(value, sort_keys=True, indent=None, separators=(",", ":")).replace("<", "\\u003c")


class Document:
    """Accumulates sections, then renders one standalone HTML file.

    Sections are kept as structured data until :meth:`render`, so rendering is a pure function of what
    was added. That is what makes byte-identical output achievable: there is no state carried between
    sections that could depend on when a section was added.
    """

    def __init__(self) -> None:
        self._sections: list = []

    # --- sections ---------------------------------------------------------------

    def header(self, title: str, subtitle: str = "", meta: dict | None = None) -> "Document":
        """The page header. `meta` is rendered as sorted key/value pairs."""
        self._sections.append(("header", {"title": title, "subtitle": subtitle,
                                           "meta": dict(sorted((meta or {}).items()))}))
        return self

    def provenance(self, items) -> "Document":
        """The block that says where every number on the page came from.

        A research report that does not state its dataset, its instrument and its seed cannot be
        re-derived by a reader, which is the difference between a result and an assertion. Rendered last
        and visibly, not as a footnote nobody reads.
        """
        self._sections.append(("provenance", {"items": [(str(k), str(v)) for k, v in items]}))
        return self

    def kpi_row(self, items) -> "Document":
        """A row of headline numbers. `items` is a sequence of `(label, value)` pairs, kept in order."""
        self._sections.append(("kpi_row", [(str(label), str(value)) for label, value in items]))
        return self

    def chart_block(self, chart_id: str, title: str, chart_spec: dict, full_width: bool = False) -> "Document":
        """One chart. `title` states the FINDING, not the chart type -- "Information is worth eight
        points", not "Bar chart". A reader scanning the page should get the conclusions from the headings."""
        self._sections.append(("chart", {"id": chart_id, "title": title, "spec": chart_spec,
                                         "full_width": bool(full_width)}))
        return self

    def chart_row(self, left: dict, right: dict) -> "Document":
        """Two charts side by side. Each is a `chart_block`-shaped dict."""
        self._sections.append(("chart_row", {"left": left, "right": right}))
        return self

    def table(self, columns, rows, sort_keys=()) -> "Document":
        """A data table. Row order is the caller's, because row order carries meaning here; only the
        COLUMN headers are sorted for the sortable set, and `sort_keys` names which are."""
        self._sections.append(("table", {
            "columns": [str(c) for c in columns],
            "rows": [[_cell(v) for v in row] for row in rows],
            "sort_keys": list(sort_keys),
        }))
        return self

    def quality_block(self, flags: dict) -> "Document":
        """The data-quality section. Rendered last and visually distinct: a reader must see it whether
        or not they read anything else."""
        self._sections.append(("quality", dict(sorted(flags.items()))))

    def note_block(self, title: str, text: str) -> "Document":
        """A caveat or an interpretive note, placed WHERE it belongs rather than at the end.

        Separate from :meth:`quality_block` because a quality block is a list of flags under a fixed
        "Data quality" heading, and using it for prose put six headings reading "Data quality" on
        report 3 -- over statements about what the rules mean, which are not data-quality findings
        at all. A reader scanning for caveats would find them filed under the wrong heading, which
        is the same as hiding them.

        Position is preserved because a caveat is only useful next to the number it qualifies. The
        lower-bound caveat on report 1 already depends on that: separated from its delta by a
        screen of charts, it would be read later or not at all.
        """
        self._sections.append(("note", {"title": title, "text": text}))
        return self

    # --- rendering --------------------------------------------------------------

    def _render_section(self, kind: str, payload) -> str:
        if kind == "header":
            parts = [f'<h1 class="title">{_e(payload["title"])}</h1>']
            if payload["subtitle"]:
                parts.append(f'<p class="subtitle">{_e(payload["subtitle"])}</p>')
            if payload["meta"]:
                meta = "".join(
                    f'<span class="meta-item"><b>{_e(k)}</b> {_e(v)}</span>'
                    for k, v in payload["meta"].items()
                )
                parts.append(f'<div class="meta">{meta}</div>')
            return f'<header class="page-header">{"".join(parts)}</header>'

        if kind == "provenance":
            rows = "".join(
                f'<div class="prov-row"><span class="prov-key">{_e(k)}</span>'
                f'<span class="prov-val">{_e(v)}</span></div>'
                for k, v in payload["items"]
            )
            return ('<section class="band provenance"><h2>Sources and method</h2>'
                    f'<div class="prov-grid">{rows}</div></section>')

        if kind == "kpi_row":
            cells = "".join(
                f'<div class="kpi"><div class="kpi-label">{_e(label)}</div>'
                f'<div class="kpi-value">{_e(value)}</div></div>'
                for label, value in payload
            )
            return f'<div class="kpi-row">{cells}</div>'

        if kind == "chart":
            return self._render_chart(payload)

        if kind == "chart_row":
            return (f'<div class="chart-row">{self._render_chart(payload["left"])}'
                    f'{self._render_chart(payload["right"])}</div>')

        if kind == "table":
            head = []
            for index, column in enumerate(payload["columns"]):
                sortable = f' data-sort-key="{index}"' if index in payload["sort_keys"] else ""
                head.append(f'<th{sortable}>{_e(column)}</th>')
            body = "".join(
                "<tr>" + "".join(f"<td>{_e(v)}</td>" for v in row) + "</tr>"
                for row in payload["rows"]
            )
            return (f'<div class="table-wrap"><table><thead><tr>{"".join(head)}</tr></thead>'
                    f"<tbody>{body}</tbody></table></div>")

        if kind == "quality":
            items = "".join(
                f'<span class="quality-item"><b>{_e(k)}</b> {_e(v)}</span>'
                for k, v in payload.items()
            )
            return f'<section class="quality"><h2>Data quality</h2><div>{items}</div></section>'

        if kind == "note":
            return (f'<section class="note"><h2>{_e(payload["title"])}</h2>'
                    f'<p>{_e(payload["text"])}</p></section>')

        raise ValueError(f"unknown section kind {kind!r}")

    def _render_chart(self, payload: dict) -> str:
        """Charts are INLINE SVG built from the embedded snapshot.

        The previous version loaded Chart.js from a CDN and drew onto a canvas. That makes the page blank
        wherever the address is unreachable -- a corporate proxy, a restricted preview, an email client --
        and a canvas cannot be read by a screen reader, printed at the right size, or saved into a note.
        Inline SVG is markup: it renders offline, prints, scales, and carries an aria-label.
        """
        cls = "chart-block full-width" if payload.get("full_width") else "chart-block"
        from reports.svg import spec_to_svg

        svg = payload.get("svg") or spec_to_svg(payload.get("spec") or {},
                                                 payload.get("title", ""))
        return f'<figure class="{cls}">{svg}</figure>'

    def render(self, path) -> None:
        """Write the complete standalone file. Deterministic: same sections in, same bytes out."""
        body = "".join(self._render_section(kind, payload) for kind, payload in self._sections)
        document = (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>Durak analysis</title>\n"
            f"<style>{_css()}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
        )
        Path(path).write_text(document, encoding="utf-8")


def _js_id(chart_id: str) -> str:
    """A valid JS identifier from a chart id, without depending on any clock or counter."""
    return "chart_" + "".join(ch if ch.isalnum() else "_" for ch in str(chart_id))


def _cell(value) -> str:
    """Render a table cell. Floats keep full precision: a report rounds, an intermediate never did."""
    if isinstance(value, float):
        return repr(value)
    return "" if value is None else str(value)


def _extra_css() -> str:
    """Styles for the inline-SVG charts and for print.

    Print is not an afterthought here: a chart report is read as a PDF, and without this block a two-column
    row crushes against the 816 CSS pixel page width and the type is sized for a screen.
    """
    t = style_tokens()
    return f"""
.provenance {{ border-top: 1px solid {t['--border']}; }}
.prov-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 4px 22px; }}
.prov-row {{ display: flex; gap: 8px; font-size: 12.5px; padding: 2px 0; }}
.prov-key {{ color: {t['--text-secondary']}; min-width: 132px; }}
.prov-val {{ color: {t['--text-primary']}; font-variant-numeric: tabular-nums; }}
.chart {{ margin: 0; width: 100%; height: auto; overflow: visible; }}
.chart .c-title {{ font-size: 15px; font-weight: 650; fill: {t['--text-primary']}; }}
.chart .c-sub {{ font-size: 12px; fill: {t['--text-muted']}; }}
.chart .c-tick {{ font-size: 11px; fill: {t['--text-muted']};
  font-variant-numeric: tabular-nums; }}
.chart .c-cat {{ font-size: 11.5px; fill: {t['--text-secondary']}; }}
.chart .c-value {{ font-size: 11px; fill: {t['--text-primary']};
  font-variant-numeric: tabular-nums; font-weight: 600; }}
.chart .c-legend {{ font-size: 11px; fill: {t['--text-secondary']}; }}
.chart .c-axis {{ font-size: 11px; fill: {t['--text-secondary']}; }}
.chart .c-reftext {{ font-size: 10.5px; fill: {t['--warning']}; }}
.chart .c-grid {{ stroke: {t['--border']}; stroke-width: 1; }}
.chart .c-ref {{ stroke: {t['--border-strong']}; stroke-width: 1.2; stroke-dasharray: 5 4; }}
.chart .c-dash {{ stroke: {t['--warning']}; stroke-width: 1.4; stroke-dasharray: 6 4; }}
@media print {{
  @page {{ margin: 14mm; }}
  body {{ font-size: 10.5pt; }}
  .no-print {{ display: none !important; }}
  figure, table, .callout {{ break-inside: avoid; }}
  .band {{ padding: 8pt 0; }}
  h2 {{ break-after: avoid; }}
  .chart {{ max-width: 100%; }}
}}
"""


def _css() -> str:
    """The stylesheet, built from the token block so the tokens are the source of the styling."""
    tokens = style_tokens()
    declarations = "\n".join(f"  {name}: {value};" for name, value in tokens.items()) + _extra_css()
    return f""":root {{
{declarations}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 24px;
  background: var(--bg-primary); color: var(--text-primary);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}}
.page-header {{ background: var(--bg-header); padding: 20px 24px; border-radius: var(--radius);
  margin-bottom: var(--gap); }}
.title {{ margin: 0 0 6px; font-size: 22px; }}
.subtitle {{ margin: 0 0 10px; color: var(--text-secondary); }}
.meta, .quality div {{ display: flex; flex-wrap: wrap; gap: 12px; color: var(--text-secondary); }}
.meta-item, .quality-item {{ background: var(--bg-card); padding: 4px 10px; border-radius: var(--radius); }}
.kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: var(--gap); margin-bottom: var(--gap); }}
.kpi {{ background: var(--bg-card); padding: 14px 16px; border-radius: var(--radius); }}
.kpi-label {{ color: var(--text-secondary); font-size: 12px; text-transform: uppercase;
  letter-spacing: .04em; }}
.kpi-value {{ font-size: 24px; font-weight: 600; margin-top: 6px; }}
.chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: var(--gap); margin-bottom: var(--gap); }}
.chart-block {{ background: var(--bg-card); padding: 16px; border-radius: var(--radius);
  margin-bottom: var(--gap); }}
.chart-block.full-width {{ grid-column: 1 / -1; }}
.chart-block h2 {{ font-size: 15px; margin: 0 0 12px; font-weight: 600; }}
.chart-block canvas {{ max-height: 320px; }}
.table-wrap {{ background: var(--bg-card); padding: 16px; border-radius: var(--radius);
  margin-bottom: var(--gap); overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--bg-header); }}
th[data-sort-key] {{ cursor: pointer; color: var(--color-accent1); }}
th[data-sort-key]::after {{ content: " \\2195"; opacity: .5; }}
tbody tr:hover {{ background: var(--bg-header); }}
.quality {{ background: var(--bg-card); padding: 16px; border-radius: var(--radius);
  border-left: 3px solid var(--color-accent2); margin-bottom: var(--gap); }}
.quality h2 {{ font-size: 15px; margin: 0 0 10px; }}
/* A caveat reads as prose, not as a flag, and is marked on the left edge in the accent colour so it
   is scannable without being mistaken for a data-quality finding. */
.note {{ background: var(--bg-card); padding: 14px 16px; border-radius: var(--radius);
  border-left: 3px solid var(--color-accent3); margin-bottom: var(--gap); }}
.note h2 {{ font-size: 13px; margin: 0 0 8px; text-transform: uppercase; letter-spacing: .04em;
  color: var(--color-accent3); }}
.note p {{ margin: 0; font-size: 14px; line-height: 1.5; color: var(--text-primary); }}
"""