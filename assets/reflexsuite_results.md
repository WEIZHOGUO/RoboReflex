# ReflexSuite-mini 结果（三任务 × 三方案）

复刻上交大 ReflexBench 任务精神（接球 / 打地鼠 / 滚球拦截），
机映口径：50Hz 控制 / 500Hz 物理，每格试验数相同、种子序列相同。

延迟计入评分（机映特有维度，ReflexBench 原基准不含）：
`final_score = success_rate × min(1, T_control / p50_reaction)`，
反应延迟越接近控制步长下限（20ms）因子越接近 1。

| 任务 | 方案 | SR | FTR | p50 (ms) | p95 (ms) | 延迟因子 | 最终分 |
|---|---|---|---|---|---|---|---|
| 接球 Ball Catching | baseline | 0.980 | 0.000 | 10.0 | 20.0 | 1.000 | **0.980** |
| 接球 Ball Catching | PPO alpha-dyn | 0.985 | 0.950 | 11.0 | 20.0 | 1.000 | **0.985** |
| 接球 Ball Catching | dual-path | 0.980 | 0.000 | 10.0 | 20.0 | 1.000 | **0.980** |
| 打地鼠 Whack-a-Mole | baseline | 1.000 | 0.000 | 12.0 | 20.0 | 1.000 | **1.000** |
| 打地鼠 Whack-a-Mole | PPO alpha-dyn | 1.000 | 1.000 | 10.0 | 20.0 | 1.000 | **1.000** |
| 打地鼠 Whack-a-Mole | dual-path | 1.000 | 0.000 | 10.0 | 20.0 | 1.000 | **1.000** |
| 滚球拦截 Rolling Ball | baseline | 0.980 | 0.000 | 10.0 | 20.0 | 1.000 | **0.980** |
| 滚球拦截 Rolling Ball | PPO alpha-dyn | 0.995 | 0.990 | 12.0 | 32.1 | 1.000 | **0.995** |
| 滚球拦截 Rolling Ball | dual-path | 0.995 | 0.000 | 12.0 | 32.4 | 1.000 | **0.995** |

> 诚实声明：本表为对 ReflexBench 任务定义的复刻实现（非官方代码/官方评测），
> 与 ReflexVLA 公开成绩的对照见 README「ReflexSuite-mini」一节。
