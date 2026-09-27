# NoC RL Framework

An end-to-end scaffold for reinforcement-learning traffic generation against a
2x2 mesh NoC.  It deliberately keeps the cocotb simulation loop asynchronous:
`AsyncNoCEnv.reset()` and `AsyncNoCEnv.step()` await clock edges themselves.
Do not wrap a live cocotb environment in Gymnasium's synchronous runners.

## Install

Create a Python environment and install the project dependencies:

```sh
cd noc_rl_framework
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Install one supported simulator.  On Debian/Ubuntu, Icarus Verilog is:

```sh
sudo apt update
sudo apt install iverilog
```

Alternatively install [Verilator](https://verilator.org/guide/latest/install.html)
with your platform's package manager.  The supplied Makefile uses Icarus by
default; use `make SIM=verilator` to select Verilator.

## Run

The included DUT is a compileable placeholder.  Replace it, and update the
placeholder signal names in `cocotb_tb/noc_interface.py` and
`cocotb_tb/traffic_driver.py`, before integrating a real NoC.

```sh
cd noc_rl_framework/dut
make
```

The cocotb test entry invokes one async Q-learning smoke-training pass.  The
mock-DUT test needs no simulator and validates the same async reset/step path:

```sh
cd noc_rl_framework
pytest -q tests/test_env_integration.py
```

## State and action shapes

The default observation is a `numpy.float32` vector with 138 values:

| Segment | Dimensions |
| --- | ---: |
| Four routers × five ports × six handshake/buffer features | 120 |
| Current traffic metadata | 14 |
| Performance counters | 4 |
| Coverage extension | 0 by default |

Set `NoCConfig(coverage_len=K)` to reserve `K` zero-valued coverage features
without changing consumers of the other segments. The coverage collector keeps
campaign data in `info["coverage"]`; the optional state extension stays
available for a future learned coverage-state encoding.

The action space is `MultiDiscrete([2, 4, 4, 2, 4, 2, 2, 4])`: inject, source,
destination, packet type, packet length (encoded as 0--3 then mapped to 1--4
flits), VC, priority, and injection gap.  This is 4,096 flattened actions for
the tabular Q learner and DQN.  Flit type is derived by the traffic driver,
never selected by the agent.

`AsyncNoCEnv.step()` returns the practical four-item async form
`(state, reward, done, info)`. Reward is the fraction of newly covered traffic
bins in that step, allowing the Q-learning and DQN agents to target coverage.

## Coverage progress

The initial functional coverage model measures 34 reachable traffic bins:
source, destination, packet type, packet length, VC, priority, injection gap,
route direction, hop count, and derived HEAD/BODY/TAIL/SINGLE flit type. It
does **not** claim RTL response or arbitration coverage; add those bins after
the actual DUT signal map is known.

Each async training episode prints local, human-readable progress, for example:

```text
2026-09-23 15:46:12 IST | Q-learning episode 1/10 | coverage 41.18% (14/34 bins; +2 new)
```

Coverage persists across environment resets during one training run and resets
only when the training loop starts a new coverage campaign.
