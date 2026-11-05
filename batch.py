#!/usr/bin/env python3
"""Run independent stages.py cases concurrently, one CPU per Linux case.

Place beside stages.py. Example:
python -u batch.py --psd-dir examples/psd/DOE_2to1_16_PSDs \
    --output-dir output/DOE_2to1 --workers 16

Extra options are forwarded to every stages.py invocation. The default is a
bimodal crystal seed followed by none -> hybrid. Supply several --psd-dir
paths to run multiple DOE folders in one batch (set --expected-cases to their
total). Crystal searches share a process-safe cache under the batch output.
"""

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time


THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
RESERVED = {'--psd', '--name', '--output-dir', '--dry-run'}


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError('Use a positive integer')
    return value


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def one_thread_environment():
    environment = os.environ.copy()
    environment.update({name: '1' for name in THREAD_VARIABLES})
    environment.update(OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE', PYTHONUNBUFFERED='1')
    return environment


def select_cpus(workers, allow_smt=False):
    """Respect the current affinity mask and prefer distinct physical cores."""
    if not sys.platform.startswith('linux') or not hasattr(os, 'sched_getaffinity'):
        return [None] * workers, dict(pinned=False, reason='CPU affinity is implemented for Linux')
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < workers:
        raise ValueError(f'Only {len(allowed)} CPUs are available to this session; requested {workers}')
    groups = {}
    for cpu in allowed:
        folder = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        try:
            key = (int((folder / 'physical_package_id').read_text()),
                   int((folder / 'core_id').read_text()))
        except (OSError, ValueError):
            return allowed[:workers], dict(pinned=True, physical_cores_known=False,
                                          allowed_cpus=allowed)
        if min(key) < 0:
            return allowed[:workers], dict(pinned=True, physical_cores_known=False,
                                          allowed_cpus=allowed)
        groups.setdefault(key, []).append(cpu)
    distinct = sorted(min(group) for group in groups.values())
    if len(distinct) < workers and not allow_smt:
        raise ValueError(f'Only {len(distinct)} distinct physical cores are available. '
                         'Reduce --workers, request more cores, or explicitly use --allow-smt')
    order = distinct + [cpu for cpu in allowed if cpu not in distinct]
    return order[:workers], dict(pinned=True, physical_cores_known=True,
                                available_physical_cores=len(groups),
                                uses_smt=workers > len(groups), allowed_cpus=allowed)


def inspect_psd(path):
    """Preflight basic input validity without changing decimal radius text."""
    particles = rows = 0
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f'Empty PSD: {path}')
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        if not {'Radius_m', 'Count'}.issubset(reader.fieldnames):
            raise ValueError(f'PSD needs Radius_m,Count: {path}')
        for row in reader:
            try:
                radius, count = Decimal(row['Radius_m']), Decimal(row['Count'])
            except (InvalidOperation, TypeError) as error:
                raise ValueError(f'Invalid radius/count in {path}') from error
            if (not radius.is_finite() or radius <= 0 or not count.is_finite() or
                    count < 0 or count != count.to_integral_value()):
                raise ValueError(f'Invalid radius/count in {path}')
            particles += int(count)
            rows += 1
    if not 0 < particles <= 2**31 - 1:
        raise ValueError(f'Invalid particle count in {path}')
    return dict(particles=particles, psd_rows=rows,
                psd_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def parser():
    options = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    options.add_argument('--psd-dir', required=True, nargs='+', type=Path)
    options.add_argument('--output-dir', required=True, type=Path)
    options.add_argument('--workers', type=positive_int, default=16)
    options.add_argument('--expected-cases', type=positive_int, default=16)
    options.add_argument('--pattern', default='*.csv')
    options.add_argument('--allow-smt', action='store_true',
                         help='Permit more workers than available physical cores')
    options.add_argument('--dry-run', action='store_true',
                         help='Check all inputs and stage commands without starting cases')
    return options


def plan_cases(root, args, forwarded):
    for token in forwarded:
        option = token.split('=', 1)[0]
        for prefix in ('--none-', '--hybrid-'):
            if option.startswith(prefix):
                option = '--' + option[len(prefix):]
                break
        if option in RESERVED:
            raise ValueError(f'{option} is managed by batch.py')
    directories = [path.expanduser().resolve() for path in args.psd_dir]
    output = args.output_dir.expanduser().resolve()
    if not all(path.is_dir() for path in directories) or not (root / 'stages.py').is_file():
        raise ValueError('Supply an existing PSD directory and place batch.py beside stages.py')
    files = sorted(path.resolve() for directory in directories
                   for path in directory.glob(args.pattern) if path.is_file())
    if len(set(files)) != len(files):
        raise ValueError('PSD folders overlap or are repeated; each input must occur once')
    if len(files) != args.expected_cases:
        raise ValueError(f'Expected {args.expected_cases} PSD files; found {len(files)} across {directories}')
    if (output / 'batch.json').exists():
        raise ValueError('This output directory already has batch.json; use a new output directory')
    cpus, affinity = select_cpus(min(args.workers, len(files)), args.allow_smt)
    cases, names = [], set()
    environment = one_thread_environment()
    for index, path in enumerate(files):
        name = path.stem
        identity = os.path.normcase(name)
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name) or identity in names:
            raise ValueError(f'Case names must be unique and contain letters, numbers, underscores or hyphens: {name}')
        names.add(identity)
        folder = output / name
        if folder.exists():
            raise ValueError(f'Case output already exists: {folder}; use a new batch directory')
        command = [sys.executable, '-u', str(root / 'stages.py'), '--psd', str(path),
                   '--name', name, '--output-dir', str(folder), *forwarded]
        print(f'Checking {index + 1}/{len(files)}: {name}', flush=True)
        stage_plan = check_command(root, command, environment, name)
        if (stage_plan['initializer']['mode'] == 'ordered' and
                not any(token.split('=', 1)[0] == '--lattice-cache' for token in forwarded)):
            command.extend(['--lattice-cache', str(output / '_lattice_cache')])
            stage_plan = check_command(root, command, environment, name)
        cases.append(dict(name=name, psd=str(path), output=str(folder), log=str(folder / 'case.log'),
                          command=command, state='pending', returncode=None, cpu=None,
                          stage_plan=stage_plan, **inspect_psd(path)))
    return dict(created_utc=now(), host=socket.gethostname(), pid=os.getpid(),
                output_dir=str(output), input_dirs=list(map(str, directories)),
                concurrent_cases=len(cpus), cpus=cpus, affinity=affinity,
                numerical_threads_per_case=1, forwarded_stage_options=forwarded,
                stages_sha256=hashlib.sha256((root / 'stages.py').read_bytes()).hexdigest(),
                cases=cases, global_optimality_proved=False)


def check_command(root, command, environment, name):
    """Validate the same stage command that a worker will execute."""
    check = subprocess.run([*command, '--dry-run'], cwd=root, env=environment,
                           stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if check.returncode:
        raise RuntimeError(f'Stage options failed preflight for {name}:\n{check.stderr or check.stdout}')
    try:
        return json.loads(check.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f'stages.py did not return a JSON dry-run plan for {name}') from error


def save_status(report):
    folder = Path(report['output_dir'])
    report['updated_utc'] = now()
    temporary = folder / 'batch.json.tmp'
    temporary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    temporary.replace(folder / 'batch.json')
    temporary = folder / 'summary.csv.tmp'
    fields = ['name', 'state', 'cpu', 'returncode', 'particles', 'packing_fraction',
              'porosity', 'started_utc', 'finished_utc', 'log', 'error']
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['cases'])
    temporary.replace(folder / 'summary.csv')


def verified_result(case):
    folder = Path(case['output']) / f"result_{case['name']}_hybrid"
    check = json.loads((folder / 'verification.json').read_text(encoding='utf-8'))
    if (not check.get('passed') or check.get('overlapping_pairs_exact') != 0 or
            check.get('self_image_violations_exact') != 0 or
            not check.get('original_radii_and_counts_exact')):
        raise RuntimeError('Final hybrid export failed exact verification')
    box = json.loads((folder / 'box.json').read_text(encoding='utf-8'))
    if box['particles'] != case['particles']:
        raise RuntimeError('Final particle count differs from input')
    return dict(packing_fraction=box['packing_fraction'], porosity=box['porosity'],
                exact_verification_passed=True)


def worker(job_file):
    specification = json.loads(job_file.read_text(encoding='utf-8'))
    cpu = specification['cpu']
    if cpu is not None:
        if not hasattr(os, 'sched_setaffinity'):
            raise RuntimeError('CPU affinity is unavailable in this worker')
        os.sched_setaffinity(0, {cpu})
    environment = one_thread_environment()
    os.chdir(specification['root'])
    if os.name == 'posix':
        os.execvpe(specification['command'][0], specification['command'], environment)
    return subprocess.run(specification['command'], env=environment, check=False).returncode


def interrupt_cases(running):
    """Signal only this batch's case process groups, allowing checkpoints to save."""
    for process, _, _ in running.values():
        try:
            if os.name == 'posix':
                os.killpg(process.pid, signal.SIGINT)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
    try:
        for process, _, _ in running.values():
            process.wait()
    except KeyboardInterrupt:
        for process, _, _ in running.values():
            if process.poll() is None:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
        for process, _, _ in running.values():
            process.wait()


def run_batch(root, report):
    folder = Path(report['output_dir'])
    folder.mkdir(parents=True, exist_ok=True)
    # Exclusive creation reserves the batch before any case is started.
    with (folder / 'batch.json').open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    pending = list(report['cases'])
    available = list(enumerate(report['cpus']))
    running = {}
    interrupted = False
    next_heartbeat = time.monotonic() + 30
    def stop_requested(signum, frame):
        raise KeyboardInterrupt
    previous_term = signal.signal(signal.SIGTERM, stop_requested)
    print(json.dumps(dict(cases=len(pending), workers=len(available), cpus=report['cpus'],
                          numerical_threads_per_case=1, affinity=report['affinity'])), flush=True)
    try:
        while pending or running:
            while pending and available:
                slot, cpu = available.pop(0)
                case = pending.pop(0)
                case_folder = Path(case['output'])
                case_folder.mkdir(exist_ok=False)
                case.update(cpu=cpu, state='running', started_utc=now())
                job_file = case_folder / 'job.json'
                job_file.write_text(json.dumps(dict(root=str(root), cpu=cpu,
                                                   command=case['command']), indent=2) + '\n')
                stream = Path(case['log']).open('x', encoding='utf-8')
                try:
                    process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                                '_worker', str(job_file)], cwd=root,
                        env=one_thread_environment(), stdin=subprocess.DEVNULL,
                        stdout=stream, stderr=subprocess.STDOUT,
                        start_new_session=os.name == 'posix')
                except OSError as error:
                    stream.close()
                    case.update(state='failed', error=str(error), finished_utc=now(), returncode=-1)
                    available.append((slot, cpu))
                    continue
                case['pid'] = process.pid
                running[slot] = process, stream, case
                observed_running = sum(job.poll() is None for job, _, _ in running.values())
                report['maximum_concurrent_cases'] = max(report.get('maximum_concurrent_cases', 0), observed_running)
                print(f"START {case['name']} cpu={cpu} pid={process.pid}", flush=True)
                save_status(report)
            for slot, (process, stream, case) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                case.update(returncode=code, finished_utc=now(), state='completed' if code == 0 else 'failed')
                if code == 0:
                    try:
                        case.update(verified_result(case))
                    except (OSError, ValueError, KeyError, RuntimeError) as error:
                        case.update(state='failed', error=str(error))
                else:
                    case['error'] = f'stages.py exited with status {code}; see case.log'
                    if os.name == 'posix':
                        try:
                            os.killpg(process.pid, signal.SIGINT)
                        except ProcessLookupError:
                            pass
                print(f"{case['state'].upper()} {case['name']} "
                      f"density={case.get('packing_fraction', 'unavailable')}", flush=True)
                available.append((slot, case['cpu']))
                del running[slot]
                save_status(report)
            if time.monotonic() >= next_heartbeat:
                print(json.dumps({state: sum(c['state'] == state for c in report['cases'])
                                  for state in ('pending', 'running', 'completed', 'failed')}), flush=True)
                next_heartbeat = time.monotonic() + 30
            if running:
                time.sleep(.5)
    except KeyboardInterrupt:
        interrupted = True
        print('Stopping this batch; waiting for case checkpoints. Interrupt again to terminate.', flush=True)
        interrupt_cases(running)
    except Exception:
        interrupted = True
        interrupt_cases(running)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        for process, stream, case in running.values():
            stream.close()
            case.update(state='interrupted', returncode=process.poll(), finished_utc=now())
        if interrupted:
            for case in pending:
                case['state'] = 'not_started'
        report.update(finished_utc=now(), interrupted=interrupted)
        save_status(report)
    failed = sum(case['state'] != 'completed' for case in report['cases'])
    print(f"Batch finished: {len(report['cases']) - failed} completed, {failed} unsuccessful. "
          f"Summary: {folder / 'summary.csv'}", flush=True)
    return 130 if interrupted else 1 if failed else 0


def main(argv=None):
    options = parser()
    args, forwarded = options.parse_known_args(argv)
    root = Path(__file__).resolve().parent
    report = plan_cases(root, args, forwarded)
    if args.dry_run:
        print(json.dumps(report, indent=2), flush=True)
        return 0
    return run_batch(root, report)


if __name__ == '__main__':
    try:
        if len(sys.argv) == 3 and sys.argv[1] == '_worker':
            raise SystemExit(worker(Path(sys.argv[2])))
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f'ERROR: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
