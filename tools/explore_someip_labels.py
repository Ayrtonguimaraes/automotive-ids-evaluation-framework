"""Second exploratory pass on the SOME/IP training captures: do the labels match the message content?

The first pass (tools/explore_someip.py) found that, in every attack folder, the share of attack
messages is the same for every message type and return code (about 4.4% of the requests, of the
responses, of the errors...). In Alkhatib et al. a packet is labelled as an attack only if it is the
attack itself (an error answering an error or a notification, a request with no response, a response
with no request), so attack labels should pile up on a few message types. This pass checks why:

    A. Per byte: does any of the 58 bytes have a value that is much more common in attack messages?
       (total variation distance between the value distributions of attack and normal messages)
    B. Per device: attack rate of the messages sent by each source IP. The generator has 8 servers,
       8 clients and 1 attacker, so attack messages could come from one address.
    C. In time: how attack labels cluster (run lengths, and the chance that the next message is an
       attack given that this one is).
    D. Alignment: pair each message with the label of the message k positions away (k from -50 to
       50, inside each capture) and measure how much the bytes then differ between attack and normal
       messages (the largest distance of part A). If labels and bytes are aligned, k = 0 wins. If the
       labels were shifted, another k wins. If no k stands out, the labels do not follow the bytes.

Only training captures are read. Usage:
    python tools/explore_someip_labels.py --dataset data/someip/SOME_IP_Intrusion_Detection_Dataset
"""
import argparse
import glob
import importlib.util
import os
import re

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "safe_pickle", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "feature_generator", "safe_pickle.py"))
_safe_pickle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_safe_pickle)

FIELDS = {**{b: "Ethernet" for b in range(14)}, **{b: "IPv4" for b in range(14, 34)}, **{b: "UDP" for b in range(34, 42)},
          42: "service", 43: "service", 44: "method", 45: "method", 46: "length", 47: "length", 48: "length",
          49: "length", 50: "client", 51: "client", 52: "session", 53: "session", 54: "proto ver",
          55: "iface ver", 56: "msg type", 57: "return code"}


def capture_index(path):
    return int(re.search(r'_(\d+)_x\.pickle$', path).group(1))


def ip(x, i):
    return [".".join(str(int(v)) for v in row) for row in x[:, i:i + 4]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    args = parser.parse_args()

    for attack_dir in sorted(glob.glob(f"{args.dataset}/*/")):
        folder = os.path.basename(os.path.normpath(attack_dir))
        paths = sorted(glob.glob(f"{attack_dir}*_x.pickle"), key=capture_index)
        paths = paths[:int(len(paths) * args.train_fraction)]
        X, Y = [], []
        for p in paths:
            X.append(np.asarray(_safe_pickle.safe_load(p)).astype(np.uint8))
            Y.append(np.asarray(_safe_pickle.safe_load(p.replace("_x.pickle", "_y.pickle"))).ravel().astype(int))
        x, y = np.concatenate(X), np.concatenate(Y)
        base = y.mean()
        print(f"\n################ {folder}: {len(paths)} captures, {len(y)} messages, attack rate {100 * base:.2f}% ################")

        # A. Per byte
        def distances(xs, ys, cols):
            out = []
            for b in cols:
                pa = np.bincount(xs[ys == 1, b], minlength=256) / max(1, (ys == 1).sum())
                pn = np.bincount(xs[ys == 0, b], minlength=256) / max(1, (ys == 0).sum())
                out.append(0.5 * np.abs(pa - pn).sum())
            return np.array(out)

        tvd = distances(x, y, range(x.shape[1]))
        top = np.argsort(tvd)[::-1][:6]
        print("[A] bytes whose values differ most between attack and normal messages "
              "(distance 0 = identical distributions, 1 = disjoint):")
        print("    " + ", ".join(f"byte {b} ({FIELDS[b]}): {tvd[b]:.3f}" for b in top))

        # B. Per source IP
        src = np.array(ip(x, 26))
        devices = sorted(set(src.tolist()), key=lambda d: -y[src == d].mean())
        print(f"[B] {len(devices)} source IPs; attack rate of the messages each one sends (highest first):")
        for d in devices[:5]:
            m = src == d
            print(f"    {d:>15}: {m.sum():8d} messages, {100 * y[m].mean():6.2f}% attack")
        print(f"    ... lowest: {devices[-1]} {100 * y[src == devices[-1]].mean():.2f}%")

        # C. In time
        runs = []
        for yc in Y:
            edges = np.flatnonzero(np.diff(np.concatenate([[0], yc, [0]])))
            runs.extend((edges[1::2] - edges[::2]).tolist())
        runs = np.array(runs)
        nxt = np.concatenate([yc[1:][yc[:-1] == 1] for yc in Y])
        print(f"[C] attack runs: {len(runs)}, length median {np.median(runs):.0f}, 90th percentile {np.percentile(runs, 90):.0f}, "
              f"max {runs.max()}; P(next is attack | this is attack) = {nxt.mean():.3f} (base rate {base:.3f})")

        # D. Alignment: pair message i with the label of message i + k, inside each capture
        cols = list(range(26, 34)) + list(range(42, 58))
        scores = {}
        for k in range(-50, 51):
            xs = np.concatenate([xc[max(0, -k):len(xc) - max(0, k)] for xc in X])
            ys = np.concatenate([yc[max(0, k):len(yc) - max(0, -k)] for yc in Y])
            scores[k] = distances(xs, ys, cols).max()
        best = sorted(scores, key=lambda k: -scores[k])[:3]
        others = np.array([v for k, v in scores.items() if k != 0])
        print(f"[D] largest byte distance when message i gets the label of message i+k: "
              f"k=0 -> {scores[0]:.3f}; best k: " + ", ".join(f"k={k:+d} -> {scores[k]:.3f}" for k in best)
              + f"; median over k != 0: {np.median(others):.3f}")


if __name__ == "__main__":
    main()
