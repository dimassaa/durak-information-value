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

ENGINE_VERSION = 2
# 2: after a take the original attacker keeps attacking and the taker skips the round. Version 1 had the
# turn handed to the taker, which let it re-attack the card it had just taken; the two players passed one
# card back and forth indefinitely and 16-21% of games per arm hit the round cap. Any dataset produced
# under version 1 contains that bias and must not be pooled with anything produced after this bump.
RULES_VERSION = 2
SCHEMA_VERSION = 1