"""Detect rotational symmetry per object (writes info/symmetry.json).

    python src/symmetry.py <obj> [<obj> ...]
    python src/symmetry.py --all --overwrite
"""

import argparse
import sys
from functools import partial

from object_processing.runner import symmetry_object, resolve_objects, run_batch, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("objects", nargs="*", help="object names (or use --all)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--rel-tol", type=float, default=0.01,
                    help="acceptance threshold as a fraction of object scale")
    ap.add_argument("--n-samples", type=int, default=4000,
                    help="surface samples for geometric verification")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    objs = resolve_objects(a.objects, a.all)
    print(f"Detecting symmetry for {len(objs)} object(s)")
    fn = partial(symmetry_object, rel_tol=a.rel_tol, n_samples=a.n_samples, overwrite=a.overwrite)
    sys.exit(1 if report(run_batch(objs, fn, workers=1)) else 0)


if __name__ == "__main__":
    main()
