# False triggers: the metric nobody was measuring

Every robot paper that shows a reflex now reports two numbers: success rate and reaction time. Nobody reports the third one — how often the robot moves when it should not. We started measuring it, and the result was less flattering than we expected.

## The setup

Take a standard ball-catching task in MuJoCo. A paddle cruises on a table; a ball is launched at it. A "reflex" is supposed to fire only when the ball is actually incoming. We train PPO end-to-end with a latency-shaped reward (success − λ·reaction time), then measure:

- **Success rate (SR)**: did it catch the ball?
- **Reaction latency**: from launch to first meaningful movement (p50/p95/p99).
- **False trigger rate (FTR)**: fraction of idle trials where the policy makes a significant movement *before* any stimulus exists.

That last one is the number that matters if the policy ever touches hardware. A reflex that fires at shadows is worse than no reflex at all.

## What vanilla RL actually learns

PPO with λ=0 (no latency penalty) reaches 100% success. It also fidgets on 96% of idle trials. The policy has discovered something uncomfortable: **moving early is never punished in a way that matters.** Catching the ball pays immediately; unnecessary motion costs nothing the optimizer can feel.

We tried the obvious fixes:

| Fix attempted | Result |
|---|---|
| Latency penalty (λ > 0) | FTR stays ~0.96 |
| Adaptive α reward (Lagrangian on SR) | FTR stays ~0.96 |
| Idle-motion penalty (explicit) | FTR 0.96 → 0.89, still useless |

The conclusion is not "the reward needs more tuning". It is that **a single end-to-end policy cannot express "wait" as a skill.** Waiting is not an action you can reinforce cheaply; it is a structural property you have to build in.

## The architecture that fixes it

RoboReflex splits the system in two:

- A **slow planner** that handles the normal task (cruising, positioning), and
- A **fast reflex** that only runs when an independent **arbiter** says a stimulus is real.

The arbiter is not another neural network. It is a few microseconds of geometry: time-to-collision under a threshold, with hysteresis (enter at TTC 0.8s, exit at 1.1s, plus a 100 ms dwell to prevent ping-pong switching). If it does not fire, the reflex network is never invoked, and therefore cannot fidget.

The false-trigger penalty lives on the arbiter's side, not on the reflex policy. Miss a real stimulus and you lose more than you lose for a false alarm (C_miss > C_false_trigger > 0), but a false alarm is explicitly the arbiter's mistake, not something the reflex policy is asked to internalize. That keeps the reflex policy from "playing dead" to avoid punishment.

## Does it hold beyond one task?

Three reaction tasks (ball catch, whack-a-mole, rolling-ball interception), 200 trials each, no per-task retuning:

| Task | System | SR | FTR | p50 / p95 latency |
|---|---|---|---|---|
| Ball catch | handcrafted baseline | .980 | 0 | 10 / 20 ms |
| | PPO α-dynamic | .985 | .950 | 11 / 20 ms |
| | dual-pathway | .980 | **0** | 10 / 20 ms |
| Whack-a-mole | handcrafted baseline | 1.000 | 0 | 12 / 20 ms |
| | PPO α-dynamic | 1.000 | 1.000 | 10 / 20 ms |
| | dual-pathway | 1.000 | **0** | 10 / 20 ms |
| Rolling ball | handcrafted baseline | .980 | 0 | 10 / 20 ms |
| | PPO α-dynamic | .995 | .990 | 12 / 32 ms |
| | dual-pathway | .995 | **0** | 12 / 32 ms |

Same outcome every time: PPO matches on success, fails on discipline. Dual-pathway matches on both. And the handcrafted baseline already had FTR 0 by construction — the point is that a *learned* policy can reach that level without a human hand-holding it.

## The deadline experiment

One more control worth doing: impose a hard 20 ms decision deadline. A slow 5 Hz planner alone collapses to 49.5% success (91% of its steps time out). Dual-pathway keeps 100%, with full-loop decision time (sensing + arbitration + inference) at p99 ≈ 2 ms. The slow planner never gets to block the fast path, which is the entire point of hard decoupling.

## What we are not claiming

- Not "dual-pathway beats end-to-end RL on every task". On pure success rate, they are equal here.
- Not "the architecture is novel". Splitting fast and slow systems is decades old (Brooks' subsumption, System 1/2, recent VLA work). What we think is new is treating the reflex as a **first-class training and measurement target**, with FTR as a first-class metric.
- Not "this is validated on hardware". Everything above is simulation. The real-robot plan (SO-ARM101 + LeRobot) is on the roadmap, and the numbers will change when sensors are noisy.

If you build reflexes, measure FTR. If you do not measure it, you probably have it.

---

*Data: `assets/reflexsuite_bench.json` · Reproduce: `python -m src.bench.reflexsuite_bench --trials 200`*
