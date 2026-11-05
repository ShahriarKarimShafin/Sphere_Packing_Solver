#!/usr/bin/env python3
"""Launch pack.py with longer-solve presets; explicit options take precedence.

Use `new --psd FILE --run RUN` for a new PSD or `new --from-export DIR --run RUN`
to branch from verified coordinates. `extend --run RUN` retains saved search
settings. Omit --rounds to stop on the saved density-plateau rule. --dry-run
prints the command without starting a search.
"""
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PRESET = {
    '--box': 'rectangular',
    '--boundary': 'periodic',
    '--seed': '1',
    '--proposals-per-round': '1200000',
    '--relax-iterations': '5000',
    '--maxfun': '1000000',
    '--refinement': 'hybrid',
    '--refine-every': '2',
    '--slp-steps': '1',
    '--slp-max-particles': '10000',
    '--lp-seconds': '120',
    '--lp-pair-limit': '250000',
    '--backend': 'native',
}


def _has_option(arguments, key):
    return any(arg == key or arg.startswith(key + '=') for arg in arguments)


def command_for(arguments):
    """Resolve an optional export seed and forward the ordinary engine arguments."""
    args = list(arguments)
    dry_run = '--dry-run' in args
    args = [arg for arg in args if arg != '--dry-run']
    if not args:
        return None, dry_run
    command, rest = args[0], args[1:]
    from_export = None
    forwarded = []
    index = 0
    while index < len(rest):
        token = rest[index]
        if token == '--from-export' or token.startswith('--from-export='):
            if command != 'new' or from_export is not None:
                raise ValueError('--from-export is allowed once, with new only')
            if token == '--from-export':
                index += 1
                if index >= len(rest) or rest[index].startswith('--'):
                    raise ValueError('--from-export requires an export folder')
                value = rest[index]
            else:
                value = token.split('=', 1)[1]
            if not value:
                raise ValueError('--from-export requires an export folder')
            from_export = Path(value).resolve()
        else:
            forwarded.append(token)
        index += 1
    imported = []
    if from_export is not None:
        for key in ('--psd', '--initial', '--box-file', '--starts'):
            if _has_option(forwarded, key):
                raise ValueError('--from-export supplies ' + key + '; do not combine them')
        coordinates = from_export / 'packing_coordinates.csv'
        if not coordinates.is_file():
            coordinates = from_export / 'x_y_z_radius.csv'
        files = (from_export / 'psd.csv', coordinates, from_export / 'box.json')
        if not all(path.is_file() for path in files):
            raise ValueError('Export must contain psd.csv, box.json and a coordinate CSV')
        imported = [
            '--psd', str(files[0]), '--initial', str(files[1]),
            '--box-file', str(files[2]), '--starts', '0',
        ]
    # argparse uses the last occurrence: user options follow the preset values.
    defaults = [part for pair in PRESET.items() for part in pair] if command == 'new' else []
    return [
        sys.executable, '-u', str(HERE / 'pack.py'), command,
        *defaults, *imported, *forwarded,
    ], dry_run


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments or arguments in (['--help'], ['-h']):
        print(__doc__)
        print('New-run defaults: ' + json.dumps(PRESET, indent=2))
        print('For all engine options: python pack.py new --help')
        return 0
    command, dry_run = command_for(arguments)
    print(json.dumps({
        'launcher': 'longer-solve preset',
        'dry_run': dry_run,
        'command_arguments': command,
        'note': ('extend retains its saved search configuration; convergence settings '
                 'may be changed explicitly; no worker parallelism is added'),
    }, indent=2), flush=True)
    if dry_run:
        return 0
    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        # The child receives Ctrl+C too and preserves its completed checkpoints.
        return 130


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        raise SystemExit(1)
