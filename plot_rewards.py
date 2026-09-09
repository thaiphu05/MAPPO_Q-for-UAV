import sys, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def smooth(arr, window=5):
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window)/window, mode='valid')


def load_scalars(logdir, tag):
    ea = EventAccumulator(logdir)
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    if tag not in tags:
        return None, None
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=np.float64)
    values = np.array([e.value for e in events], dtype=np.float64)
    return steps, values


def plot_traces(logdir, label, color, linestyle, ax_ret, ax_satisfied, window=50):
    # Episodic return
    steps_ret, ret = load_scalars(logdir, 'charts/episodic_return')
    if ret is not None:
        ax_ret.scatter(steps_ret, ret, s=1, color=color, alpha=0.2)
        if len(ret) >= window:
            ax_ret.plot(steps_ret[window-1:], smooth(ret, window),
                        color=color, linewidth=1.5, linestyle=linestyle, label=label)
        print(f'  {label}: episodic_ret {len(ret)} values, last={ret[-1]:.1f}')

    # Avg satisfied users
    steps_sat, sat = load_scalars(logdir, 'charts/avg_satisfied_users')
    if sat is not None:
        ax_satisfied.plot(steps_sat, sat, color=color, linewidth=1.5,
                          linestyle=linestyle, label=label)


def find_mappo_run(model_tag):
    """Latest runs/ dir for the given MAPPO variant ('Quantum' or 'Normal')."""
    tagged = sorted(glob.glob(f'runs/env_multiUAV_MAPPO_{model_tag}_*'))
    if tagged:
        return tagged[-1]
    if model_tag == 'Quantum':
        # Runs logged before the Quantum/Normal tag existed were all quantum.
        legacy = sorted(d for d in glob.glob('runs/env_multiUAV_MAPPO_*')
                         if '_Normal_' not in d and '_Quantum_' not in d)
        if legacy:
            return legacy[-1]
    return None


def compare_mappo_vs_mappoq(quantum_logdir=None, normal_logdir=None,
                             output='runs/mappo_vs_mappoq_compare.png', window=50):
    """Overlay the latest MAPPO-Q and MAPPO (normal) reward curves on one chart."""
    quantum_logdir = quantum_logdir or find_mappo_run('Quantum')
    normal_logdir = normal_logdir or find_mappo_run('Normal')
    if not quantum_logdir or not normal_logdir:
        missing = 'MAPPO-Q' if not quantum_logdir else 'MAPPO (normal)'
        print(f'Khong tim thay run cho {missing} trong runs/')
        return None

    fig, (ax_ret, ax_satisfied) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    plot_traces(quantum_logdir, f'MAPPO-Q ({os.path.basename(quantum_logdir)})',
                '#1f77b4', '-', ax_ret, ax_satisfied, window)
    plot_traces(normal_logdir, f'MAPPO ({os.path.basename(normal_logdir)})',
                '#d62728', '--', ax_ret, ax_satisfied, window)

    ax_ret.set_ylabel('Episodic Return')
    ax_ret.legend(fontsize=9, loc='lower right')
    ax_ret.grid(True, alpha=0.3)

    ax_satisfied.set_xlabel('Total Steps')
    ax_satisfied.set_ylabel('Avg Satisfied Users')
    ax_satisfied.legend(fontsize=9, loc='lower right')
    ax_satisfied.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    plt.savefig(output, dpi=150)
    plt.close()
    print(f'Saved: {output}')
    return output


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdirs', nargs='*', help='Path(s) to TensorBoard log dirs')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--single', action='store_true', help='Plot latest single UAV run only')
    group.add_argument('--mappo', action='store_true', help='Plot latest MAPPO run only')
    group.add_argument('--both', action='store_true', help='Plot both latest runs (default)')
    group.add_argument('--compare-mappoq', action='store_true',
                        help='Overlay latest MAPPO vs MAPPO-Q runs on one chart')
    parser.add_argument('--window', type=int, default=50, help='Smoothing window for episodic return')
    parser.add_argument('--output', default=None, help='Output image path')
    args = parser.parse_args()

    if args.compare_mappoq:
        compare_mappo_vs_mappoq(output=args.output or 'runs/mappo_vs_mappoq_compare.png',
                                 window=args.window)
        return

    if args.output is None:
        args.output = 'runs/reward_curve.png'

    logdirs = list(args.logdirs)

    if not logdirs:
        if args.single:
            runs = sorted(glob.glob('runs/env_UAV_Environment_*'))
            if runs:
                logdirs.append(runs[-1])
        elif args.mappo:
            runs = sorted(glob.glob('runs/env_multiUAV_MAPPO_*'))
            if runs:
                logdirs.append(runs[-1])
        else:
            single_runs = sorted(glob.glob('runs/env_UAV_Environment_*'))
            mappo_runs = sorted(glob.glob('runs/env_multiUAV_MAPPO_*'))
            if single_runs:
                logdirs.append(single_runs[-1])
            if mappo_runs:
                logdirs.append(mappo_runs[-1])

        if not logdirs:
            print('No runs found in runs/')
            sys.exit(1)
    else:
        print(f'Using provided logdirs: {logdirs}')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    fig, (ax_ret, ax_satisfied) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for i, ld in enumerate(logdirs):
        label = os.path.basename(ld)
        is_mappo = 'MAPPO' in label
        linestyle = '-' if is_mappo else '--'
        plot_traces(ld, label, colors[i % len(colors)], linestyle,
                    ax_ret, ax_satisfied, args.window)

    ax_ret.set_ylabel('Episodic Return')
    ax_ret.legend(fontsize=7, loc='lower right')
    ax_ret.grid(True, alpha=0.3)
    if not ax_ret.lines:
        ax_ret.set_visible(False)

    ax_satisfied.set_xlabel('Total Steps')
    ax_satisfied.set_ylabel('Avg Satisfied Users')
    ax_satisfied.legend(fontsize=7, loc='lower right')
    ax_satisfied.grid(True, alpha=0.3)
    if not ax_satisfied.lines:
        ax_satisfied.set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
