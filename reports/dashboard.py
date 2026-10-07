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
_TOKENS = {
    "--bg-primary": "#0f1117",
    "--bg-card": "#171a23",
    "--bg-header": "#1e222d",
    "--text-primary": "#e8eaed",
    "--text-secondary": "#9aa0ac",
    "--color-accent1": "#4f9cf9",
    "--color-accent2": "#52c41a",
    "--color-accent3": "#faad14",
    "--color-accent4": "#f5222d",
    "--positive": "#52c41a",
    "--negative": "#f5222d",
    "--gap": "16px",
    "--radius": "8px",
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

        raise ValueError(f"unknown section kind {kind!r}")

    def _render_chart(self, payload: dict) -> str:
        cls = "chart-block full-width" if payload.get("full_width") else "chart-block"
        return (
            f'<div class="{cls}"><h2>{_e(payload["title"])}</h2>'
            f'<canvas id="{_e(payload["id"])}"></canvas>'
            f'<script>const {_js_id(payload["id"])} = {_j(payload["spec"])};'
            f'new Chart(document.getElementById("{payload["id"]}"), {_js_id(payload["id"])});</script>'
            f"</div>"
        )

    def render(self, path) -> None:
        """Write the complete standalone file. Deterministic: same sections in, same bytes out."""
        body = "".join(self._render_section(kind, payload) for kind, payload in self._sections)
        document = (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>Durak analysis</title>\n"
            f'<script src="https://cdn.jsdelivr.net/npm/chart.js@{CHARTJS_VERSION}"></script>\n'
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


def _css() -> str:
    """The stylesheet, built from the token block so the tokens are the source of the styling."""
    tokens = style_tokens()
    declarations = "\n".join(f"  {name}: {value};" for name, value in tokens.items())
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
  border-left: 3px solid var(--color-accent2); }}
.quality h2 {{ font-size: 15px; margin: 0 0 10px; }}
"""