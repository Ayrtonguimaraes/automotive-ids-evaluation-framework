"""Exploratory analysis of the SOME/IP captures, to choose what the pipeline should change.

Only the TRAINING captures are read (same 70% split as the feature generator), so nothing here is
tuned on the test set. Every number is computed from the 58 header bytes and the message labels.

Byte positions (Ethernet 14 + IPv4 20 + UDP 8 + SOME/IP 16 = 58 bytes):
    26-29 source IP, 30-33 destination IP, 34-35 / 36-37 UDP ports
    42-43 service ID, 44-45 method ID (bit 15 set = event), 50-51 client ID, 52-53 session ID
    56 message type (0x00 request, 0x01 request without return, 0x02 notification,
       0x80 response, 0x81 error), 57 return code

Sections:
    1. Messages: how many are attacks, and do attack messages look different on their own?
    2. Conversations: how much does a window in arrival order mix client-server conversations?
    3. Window labels: how many attack messages an "attack" window really has, in arrival order and
       when the messages are first split by conversation.
    4. Protocol rules: simple request/response bookkeeping on the header fields, compared with the
       labels. If these rules find the attacks, the information is in the 58 bytes and the problem is
       how the windows are built. The rules were written knowing the attack names, so they are only a
       check of what the bytes contain, never a proposed detector.

Usage: python tools/explore_someip.py --dataset data/someip/SOME_IP_Intrusion_Detection_Dataset
"""
import argparse
import collections
import glob
import importlib.util
import os
import re

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "safe_pickle", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "feature_generator", "safe_pickle.py"))
_safe_pickle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_safe_pickle)

WINDOW = 45  # messages per window (44 differences)
TYPES = {0x00: "REQUEST", 0x01: "REQ_NO_RETURN", 0x02: "NOTIFICATION", 0x80: "RESPONSE", 0x81: "ERROR"}


def capture_index(path):
    return int(re.search(r'_(\d+)_x\.pickle$', path).group(1))


def u16(x, i):
    return (x[:, i].astype(np.int64) << 8) | x[:, i + 1]


def u32(x, i):
    return (u16(x, i) << 16) | u16(x, i + 2)


def conversation_ids(x):
    """One id per client-server pair (unordered pair of IP addresses), as Alkhatib et al. group them."""
    a, b = u32(x, 26), u32(x, 30)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    _, ids = np.unique(lo * (1 << 32) + hi, return_inverse=True)
    return ids


def window_labels(y, conv=None):
    """Number of attack messages in each window of WINDOW messages (slide 1), in arrival order or per conversation."""
    streams = [y] if conv is None else [y[conv == c] for c in np.unique(conv)]
    counts = []
    for s in streams:
        if len(s) > WINDOW:
            csum = np.concatenate([[0], np.cumsum(s)])
            # Same as the feature generator: N messages give N - 45 windows, and window i is labelled by
            # the labels of messages i to i + 43
            counts.append(csum[WINDOW - 1:len(s) - 1] - csum[:len(s) - WINDOW])
    return np.concatenate(counts) if counts else np.array([], dtype=int)


def protocol_rules(x):
    """Flag messages that break request/response bookkeeping. Returns a dict of boolean arrays."""
    service, method = u16(x, 42), u16(x, 44)
    client, session, mtype = u16(x, 50), u16(x, 52), x[:, 56]
    open_requests = set()      # (service, method, client, session) waiting for an answer
    last_answer = {}           # same key -> type of the last answer seen
    flags = {name: np.zeros(len(x), dtype=bool) for name in ("answer_without_request", "repeated_error", "error_on_event")}
    for i in range(len(x)):
        key = (int(service[i]), int(method[i]), int(client[i]), int(session[i]))
        t = int(mtype[i])
        if t == 0x00:
            open_requests.add(key)
        elif t in (0x80, 0x81):
            if key not in open_requests:
                flags["answer_without_request"][i] = True
                # An error with no open request whose previous answer on the same key was also an error
                if t == 0x81 and last_answer.get(key) == 0x81:
                    flags["repeated_error"][i] = True
            open_requests.discard(key)
            last_answer[key] = t
        if t == 0x81 and method[i] & 0x8000:
            flags["error_on_event"][i] = True
    return flags


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    args = parser.parse_args()

    for attack_dir in sorted(glob.glob(f"{args.dataset}/*/")):
        folder = os.path.basename(os.path.normpath(attack_dir))
        paths = sorted(glob.glob(f"{attack_dir}*_x.pickle"), key=capture_index)
        paths = paths[:int(len(paths) * args.train_fraction)]
        X, Y, CONV, CAP = [], [], [], []
        for k, p in enumerate(paths):
            x = np.asarray(_safe_pickle.safe_load(p)).astype(np.uint8)
            y = np.asarray(_safe_pickle.safe_load(p.replace("_x.pickle", "_y.pickle"))).ravel().astype(int)
            X.append(x); Y.append(y); CONV.append(conversation_ids(x)); CAP.append(np.full(len(y), k))
        x, y = np.concatenate(X), np.concatenate(Y)
        print(f"\n################ {folder}: {len(paths)} training captures, {len(y)} messages ################")

        # 1. Messages
        print(f"\n[1] attack messages: {y.sum()} ({100 * y.mean():.2f}%)")
        print(f"{'message type':>15} {'normal':>10} {'attack':>10} {'% of type that is attack':>26}")
        for t in sorted(set(x[:, 56].tolist())):
            n, a = int(np.sum((x[:, 56] == t) & (y == 0))), int(np.sum((x[:, 56] == t) & (y == 1)))
            print(f"{TYPES.get(t, hex(t)):>15} {n:10d} {a:10d} {100 * a / (n + a):26.2f}")
        rc_attack = collections.Counter(x[y == 1, 57].tolist()).most_common(3)
        rc_normal = collections.Counter(x[y == 0, 57].tolist()).most_common(3)
        print(f"return codes, attack: {rc_attack} | normal: {rc_normal}")
        event_bit = (u16(x, 44) & 0x8000) > 0
        print(f"event bit set: normal {100 * event_bit[y == 0].mean():.1f}%, attack {100 * event_bit[y == 1].mean():.1f}%")

        # 2. Conversations
        same_next, distinct, n_conv, short = [], [], [], 0
        for xc, conv in zip(X, CONV):
            same_next.append(conv[1:] == conv[:-1])
            n_conv.append(conv.max() + 1)
            starts = np.arange(0, max(1, len(conv) - WINDOW + 1), 50)
            distinct.extend(len(np.unique(conv[s:s + WINDOW])) for s in starts)
            sizes = np.bincount(conv)
            short += int(sizes[sizes < WINDOW].sum())
        same_next = np.concatenate(same_next)
        print(f"\n[2] client-server conversations per capture: median {int(np.median(n_conv))}, max {max(n_conv)}")
        print(f"consecutive messages from the SAME conversation: {100 * same_next.mean():.1f}%")
        print(f"distinct conversations in a window of {WINDOW} messages (arrival order): "
              f"median {np.median(distinct):.0f}, 90th percentile {np.percentile(distinct, 90):.0f}")
        print(f"messages in conversations shorter than {WINDOW} (no window if split by conversation): "
              f"{short} ({100 * short / len(y):.2f}%)")

        # 3. Window labels
        arrival = np.concatenate([window_labels(yc) for yc in Y])
        by_conv = np.concatenate([window_labels(yc, conv) for yc, conv in zip(Y, CONV)])
        for name, counts in (("arrival order", arrival), ("by conversation", by_conv)):
            att = counts[counts > 0]
            print(f"\n[3] windows, {name}: {len(counts)} windows, {100 * np.mean(counts > 0):.2f}% labelled attack")
            if len(att):
                print(f"    attack messages inside an attack window: 1 -> {100 * np.mean(att == 1):.1f}%, "
                      f"2-4 -> {100 * np.mean((att >= 2) & (att <= 4)):.1f}%, 5+ -> {100 * np.mean(att >= 5):.1f}%, "
                      f"median {np.median(att):.0f} of {WINDOW - 1}")

        # 4. Protocol rules
        flags = [protocol_rules(xc) for xc in X]
        print(f"\n[4] protocol rules on the header fields (per message)")
        print(f"{'rule':>24} {'flagged':>8} {'precision':>10} {'recall':>8}")
        union = np.zeros(len(y), dtype=bool)
        for name in flags[0]:
            f = np.concatenate([fl[name] for fl in flags])
            union |= f
            prec = y[f].mean() if f.any() else float("nan")
            print(f"{name:>24} {int(f.sum()):8d} {prec:10.3f} {f[y == 1].mean():8.3f}")
        prec = y[union].mean() if union.any() else float("nan")
        print(f"{'any rule':>24} {int(union.sum()):8d} {prec:10.3f} {union[y == 1].mean():8.3f}")
        missed = x[(y == 1) & ~union]
        if len(missed):
            print("attack messages no rule flags, by type: "
                  + str({TYPES.get(k, hex(k)): v for k, v in collections.Counter(missed[:, 56].tolist()).most_common()}))


if __name__ == "__main__":
    main()
