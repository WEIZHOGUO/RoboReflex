# Roadmap

Where RoboReflex is heading. Roughly in order. If any of these matches what you need, open an issue — demand changes priorities.

## Now

- **Dual-pathway necessity study**: a task where the stimulus approaches from outside the sensor field — the slow planner must reposition before the reflex can fire. Three-way ablation (fast-only / slow-only / dual) to show the split is load-bearing, not decorative.
- **Paper**: preprint covering the architecture, the α-dynamic reward, and the FTR/latency-profile methodology.

## Next

- **ReflexBench adapter**: run RoboReflex-trained policies against the official ReflexBench tasks once the benchmark code is public. (Our ReflexSuite-mini follows its task design as a stopgap.)
- **Real hardware**: SO-ARM101 arm + webcam over the LeRobot stack. Tabletop rolling-ball interception, latency profile measured end-to-end on device. No ROS, ONNX export path as-is.
- **MCU deployment**: hand-written static-C forward pass (no runtime allocation) on ESP32-class hardware, to show the exported reflex nets are small enough to live anywhere.

## Later / maybe

- More scenario packs (fall recovery, crowd avoidance) — scenario format is stable, contributions welcome.
- Streaming/online reflex variants for continuous-control settings.
- PyPI packaging once the API settles.

## Not planned

- ROS integration (deliberate: the platform stays dependency-light).
- Real-time guarantees on Windows hosts (we report wall-clock distributions; hard RT belongs on the target device).
