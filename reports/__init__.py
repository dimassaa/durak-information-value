"""Report rendering: the JSON aggregates in, a standalone HTML file out.

Separate from `durak` and from `analysis` because nothing here runs a game. `durak` produces the
dataset and must stay importable under `nopython`; this package writes files and fetches nothing.
"""
