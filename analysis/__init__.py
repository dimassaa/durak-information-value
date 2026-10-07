"""Analysis-side code: the dataset's readers, its statistics, and its published SQL.

A separate top-level package rather than part of `durak`, because nothing here runs a game. `durak`
produces the dataset and must stay importable in a Numba `nopython` context; these modules talk to
Parquet, DuckDB and the filesystem, none of which compiles.
"""
