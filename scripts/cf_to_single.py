"""Convert a CounterfactualReplayBuffer .npz to a single-action ReplayBuffer .npz.

Used by the CF-vs-single training comparison so that both runs consume the
exact same transitions: the CF run reads the original CF buffer, the
single-action baseline reads this derived buffer. The only difference between
the two training runs is then the loss function.

The derived row keeps `s`, copies `next_states[a_executed]` into `s_next`,
and copies `a_executed` into `a`. All info-dict fields are preserved.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from jepa_tetris.data.buffer_adapters import cf_to_single_action_buffer
from jepa_tetris.data.replay_buffer import CounterfactualReplayBuffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True,
                        help="Source CounterfactualReplayBuffer .npz")
    parser.add_argument("--out", required=True,
                        help="Destination ReplayBuffer .npz")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out)
    if not in_path.is_file():
        raise FileNotFoundError(in_path)

    cf = CounterfactualReplayBuffer.load(in_path)
    print(f"loaded {cf.size} CF rows from {in_path}")
    rb = cf_to_single_action_buffer(cf)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rb.save(out_path)
    print(f"wrote {rb.size} single-action rows to {out_path}")


if __name__ == "__main__":
    main()
