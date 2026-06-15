"""
cache_manifest — settings-manifest sidecars for cached output trees.
Last modified: 2026-06-10

Each cached output tree carries a _settings_manifest.json recording the settings
that shaped it. Every use_cache_* consumer verifies the manifest besides plain
file existence; a missing, unreadable or mismatching manifest reads as STALE and
forces a recompute, which rewrites the manifest. The per-tree key lists live
here so producers and consumers can never drift.
"""

import json
import os

import settings

MANIFEST_NAME = '_settings_manifest.json'

# Settings recorded per cached tree: the core run-shaping settings plus the
# generator knobs specific to what the tree contains. Runtime-resolved
# identifiers (infra/svc version, svc_int_id) are passed via `versions`.
TREE_KEYS = {
    'catchment_4a':  ('CATCHMENT_METHOD', 'TRAVEL_COST_METHOD',
                      'TRANSFER_COST_MODEL', 'start_year_scenario'),
    'station_od_4b': ('CATCHMENT_METHOD', 'TRAVEL_COST_METHOD',
                      'TRANSFER_COST_MODEL', 'OD_ATTRIBUTION_MODE',
                      'OD_BLEND_POP_RATE', 'OD_BLEND_EMPL_RATE',
                      'OD_SCALING_POP_WEIGHT', 'OD_SCALING_EMPL_WEIGHT',
                      'start_year_scenario'),
    'assignment_4c': ('CATCHMENT_METHOD', 'ROUTING_ASSIGNMENT_METHOD',
                      'ROUTING_K_PATHS', 'ROUTING_COST_WINDOW_MIN',
                      'ROUTING_COST_WINDOW_PCT', 'ROUTING_MAX_TRANSFERS',
                      'ROUTING_MAX_EXAMINE', 'ROUTING_LOGIT_ENGINE',
                      'TRAVEL_COST_METHOD', 'TRANSFER_COST_MODEL',
                      'OD_ATTRIBUTION_MODE'),
    'flows_6d':      ('ROUTING_ASSIGNMENT_METHOD', 'CATCHMENT_METHOD'),
    'infra_ints_5a': ('INFRA_INT_MODE', 'CC_DESIGN_SPEED_KMH', 'CC_CANT_MM',
                      'CC_CANT_DEFICIENCY_MM', 'CC_BACKTRACK_ANGLE_DEG'),
    'svc_ints_5b':   ('SVC_INT_MODE', 'EXT_BUFFER_RADIUS_M',
                      'EXT_MAX_CANDIDATES', 'EXT_MIN_FREQ_DEP_PER_H',
                      'NDC_FREQ_DEP_PER_H', 'NDC_MAX_TERMINI_PER_END',
                      'DEV_ID_START_EXT', 'DEV_ID_START_NDC'),
    'svc_int_cap_5c': ('SVC_INT_CAP_THRESHOLD_TPHPD', 'CAPACITY_MODE',
                       'CAPACITY_GROUPING_STRATEGY', 'SVC_INT_MODE'),
    'scenarios_7':   ('amount_of_scenarios', 'start_year_scenario',
                      'end_year_scenario', 'OD_ATTRIBUTION_MODE',
                      'CATCHMENT_METHOD'),
    # Phase 8 benefits/costs share one combo dir; each writes its own manifest
    # filename (see write_manifest `name`). cost_parameters.py constants (VTTS,
    # per-metre op cost, KDG, ref daily dep) are NOT settings attrs → uncaptured.
    'costs_8a':      ('TRAVEL_COST_METHOD', 'TRANSFER_COST_MODEL',
                      'OD_ATTRIBUTION_MODE', 'CATCHMENT_METHOD',
                      'ROUTING_ASSIGNMENT_METHOD', 'amount_of_scenarios',
                      'start_year_scenario', 'end_year_scenario',
                      'start_valuation_year'),
    'costs_8b':      ('SVC_INT_MODE', 'SVC_INT_CAP_THRESHOLD_TPHPD',
                      'CAPACITY_MODE', 'CAPACITY_GROUPING_STRATEGY'),
    # V-track snapshots (validation_core.snapshot_baseline): the full GC-relevant
    # fingerprint, so every archived baseline/combo records what produced it.
    'validation_snapshot': ('CATCHMENT_METHOD', 'ROUTING_ASSIGNMENT_METHOD',
                            'TRAVEL_COST_METHOD', 'TRANSFER_COST_MODEL',
                            'OD_ATTRIBUTION_MODE', 'OD_BLEND_POP_RATE',
                            'OD_BLEND_EMPL_RATE', 'start_year_scenario',
                            'ROUTING_K_PATHS', 'ROUTING_COST_WINDOW_MIN',
                            'ROUTING_COST_WINDOW_PCT', 'ROUTING_MAX_TRANSFERS',
                            'ROUTING_LOGIT_ENGINE'),
}


def write_manifest(tree_dir, tree_key: str, versions: dict,
                   name: str = MANIFEST_NAME) -> None:
    """Write the settings manifest for a cached output tree (atomic replace).

    Args:
        tree_dir: Directory the cached outputs live in.
        tree_key: TREE_KEYS entry naming the settings that shape this tree.
        versions: Runtime-resolved identifiers the tree depends on
            (e.g. {'infra_version': ..., 'svc_network': ...}).
        name: Manifest filename — override when two trees share a directory
            (e.g. Phase 8 benefits/costs) so neither overwrites the other.
    """
    os.makedirs(str(tree_dir), exist_ok=True)
    path = os.path.join(str(tree_dir), name)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(_snapshot(tree_key, versions), f, indent=2, default=str)
    os.replace(tmp, path)


def check_manifest(tree_dir, tree_key: str, versions: dict,
                   name: str = MANIFEST_NAME) -> bool:
    """True when the tree's manifest matches the current settings + versions.

    A missing, unreadable or mismatching manifest prints a warning naming the
    changed keys and returns False (the cache is treated as stale; the
    recompute rewrites the manifest).

    Args:
        tree_dir: Directory the cached outputs live in.
        tree_key: TREE_KEYS entry naming the settings that shape this tree.
        versions: Runtime-resolved identifiers, same shape as at write time.
        name: Manifest filename — must match the write_manifest call for this
            tree (see Phase 8 benefits/costs sharing one directory).
    """
    path = os.path.join(str(tree_dir), name)
    if not os.path.exists(path):
        print(f"  [manifest] no {name} in {tree_dir} — "
              f"treating cache as stale")
        return False
    try:
        with open(path, encoding='utf-8') as f:
            stored = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"  [manifest] unreadable manifest at {path} ({exc}) — "
              f"treating cache as stale")
        return False
    current = json.loads(json.dumps(_snapshot(tree_key, versions), default=str))
    diffs = []
    if stored.get('tree') != tree_key:
        diffs.append(f"tree: cached={stored.get('tree')!r} current={tree_key!r}")
    for section in ('versions', 'settings'):
        cur, old = current.get(section, {}), stored.get(section, {})
        for k in sorted(set(cur) | set(old)):
            if cur.get(k) != old.get(k):
                diffs.append(f"{k}: cached={old.get(k)!r} current={cur.get(k)!r}")
    if diffs:
        print(f"  [manifest] settings changed since {tree_dir} was written — "
              f"recomputing:")
        for d in diffs:
            print(f"    {d}")
        return False
    return True


def _norm_setting(v):
    """Normalise a settings value for cache comparison.

    List-like values (e.g. SVC_INT_MODE = ['EXT', 'NDC']) become a sorted,
    lowercased list so element order and case never spuriously invalidate a
    cached tree. Scalars pass through unchanged.
    """
    if isinstance(v, (list, tuple, set)):
        return sorted(str(x).strip().lower() for x in v)
    return v


def _snapshot(tree_key: str, versions: dict) -> dict:
    return {'tree': tree_key,
            'versions': {k: str(v) for k, v in (versions or {}).items()},
            'settings': {k: _norm_setting(getattr(settings, k, None))
                         for k in TREE_KEYS[tree_key]}}
