"""Version stamps recorded on every telemetry row and shown in every report.

Three independent versions, because they answer different questions:

* ``ENGINE_VERSION`` — simulation behaviour (deal, rules, strategies, search).
  Bump it when a change could alter a result.
* ``RULES_VERSION`` — the normative rule set. Bump it when a *rule* changes, so a
  dataset produced before the fix can never be pooled with one produced after.
* ``SCHEMA_VERSION`` — column shape. Bump it when a column is added, removed, or
  retyped.

The research dataset mixes these in derived aggregates, so a stale stamp is not a
bookkeeping detail: it silently mixes two different experiments into one number.
"""

ENGINE_VERSION = 1
RULES_VERSION = 1
SCHEMA_VERSION = 1