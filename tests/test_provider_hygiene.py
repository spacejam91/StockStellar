"""Guards against generated data reaching a real measurement.

There is one bug shape this file exists to prevent, and it already happened
once: a synthetic provider is added, the exclusion filter is not updated, and
its rows quietly join the measured base rates. The filter read
`r.provider <> 'mock'` while providers named 'null' and 'signal' existed.

'null' would have diluted a real base rate toward 50%. 'signal' is worse --
it plants a strong artificial edge by design, so its rows would have reported
a fabricated edge as a measurement, in the one output of this project that is
supposed to be a fact about the world.

Nothing about that failure is visible from the output. The number just looks
better. Hence a test.

    uv run python -m tests.test_provider_hygiene
"""

from __future__ import annotations

import inspect

from app import market_data as md

# Providers that read real market data. Everything else must be declared
# synthetic. Update this deliberately when a real provider is added.
REAL_PROVIDERS = {"yahoo"}


def _provider_classes() -> dict[str, type]:
    out = {}
    for _name, obj in inspect.getmembers(md, inspect.isclass):
        if obj.__module__ == md.__name__ and isinstance(getattr(obj, "name", None), str):
            out[obj.name] = obj
    return out


def test_every_provider_is_classified():
    """No provider may be neither synthetic nor declared real."""
    names = set(_provider_classes())
    unclassified = names - md.SYNTHETIC_PROVIDERS - REAL_PROVIDERS
    assert not unclassified, (
        f"provider(s) {sorted(unclassified)} are classified neither synthetic nor real. "
        f"Add each to app.market_data.SYNTHETIC_PROVIDERS if its bars are generated, "
        f"or to REAL_PROVIDERS here if they are observed. Leaving one unclassified "
        f"means generated rows can reach the measured base rates."
    )
    return {"providers": sorted(names), "synthetic": sorted(md.SYNTHETIC_PROVIDERS)}


def test_synthetic_list_has_no_phantoms():
    """Every name in SYNTHETIC_PROVIDERS must correspond to a real class."""
    names = set(_provider_classes())
    phantom = md.SYNTHETIC_PROVIDERS - names
    assert not phantom, (
        f"SYNTHETIC_PROVIDERS names {sorted(phantom)} with no matching provider class. "
        f"A stale name is not harmful on its own, but it suggests the list and the "
        f"providers have drifted -- which is how the real one goes missing."
    )
    return {"checked": sorted(md.SYNTHETIC_PROVIDERS)}


def test_base_rate_query_excludes_every_synthetic_provider():
    """The generated SQL must filter all of them, not just 'mock'."""
    import sqlite3
    from unittest.mock import patch

    from scanner import store

    captured = {}
    real_read = store.pd.read_sql_query

    def spy(q, con, params=None, **kw):
        captured["sql"] = q
        captured["params"] = list(params or [])
        return real_read(q, con, params=params, **kw)

    with patch.object(store.pd, "read_sql_query", spy):
        try:
            store.score_band_base_rates(horizon=5)
        except sqlite3.Error:
            pass

    sql, params = captured.get("sql", ""), captured.get("params", [])
    missing = [p for p in md.SYNTHETIC_PROVIDERS if p not in params]
    assert not missing, (
        f"base-rate query does not exclude {sorted(missing)}. Generated rows from "
        f"those providers will be counted as real measurements.\nparams={params}"
    )
    assert "NOT IN" in sql.upper(), f"expected a NOT IN exclusion, got:\n{sql}"
    return {"excluded": sorted(params), "n_excluded": len(params)}


def main() -> int:
    checks = [
        ("every provider is classified", test_every_provider_is_classified),
        ("no phantom names in the list", test_synthetic_list_has_no_phantoms),
        ("base-rate SQL excludes all synthetic", test_base_rate_query_excludes_every_synthetic_provider),
    ]
    print("=" * 72)
    print("  Provider hygiene — can generated data reach a measurement?")
    print("=" * 72)
    failed = 0
    for label, fn in checks:
        try:
            detail = fn()
            print(f"\n  PASS  {label}")
            for k, v in (detail or {}).items():
                print(f"          {k}: {v}")
        except AssertionError as e:
            failed += 1
            print(f"\n  FAIL  {label}\n        {e}")
    print("\n" + "=" * 72)
    print(f"  {len(checks) - failed}/{len(checks)} passed")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
