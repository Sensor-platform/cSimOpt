# cSimOpt

cSimOpt jointly optimizes sensor parameters and ML model hyperparameters by implementing a Multi-Fidelity Bayesian Optimization (MFBO). In cSimOpt, a budget refers to the amount of data collected for a given sensor parameter.

It works by iteratively fitting a surrogate model (Gaussian Process (GP) or a deep power law (DPL)) on an observed set that consist of `(configuration, budget, performance)`, then using an acquisition function (expected improvement (EI) or upper confidence bound (UCB)) to pick the next configuration and budget to evaluate in a freeze-thaw manner, until a target performance is reached.

## Files

| File | Description |
|---|---|
| `cSimOpt.py` | Main script to execute cSimOpt (`BayesianOptimizerMF`) |
| `dpl_surrogate.py` | `DPL` (Deep Power Law) surrogate model, based on the architecture from Kadra et al. [1]. It approximates a power-law function based on given configurations and budgets |
| `sensor_performance.npy` | Pre-computed look-up table. Shape `(9000, 5)` — 9000 is the number of configurations (3×5×4×5×5×6), 5 is the budget level (1000, 2000, 3000, 4000, 5000). Performance values are obtained using the radar sensing setting from Kim and Kim [2]. |
| `requirements.txt` | Python package dependencies |

## Search space of configurations

- **Sensor parameters**: step size `[1,2,3]`, start point `[1,2,3,4,5]`, number of points `[1,2,3,4]` (Scaled values)
- **Hyperparameters**: number of units in 1st/2nd hidden layer `[1,2,3,4,5]`, learning rate `[1,2,3,4,5,6]` (Scaled values)

Each combination is encoded as an integer index (0~8999) and used to look up the performance in `sensor_performance.npy`.

## Usage

```bash
pip install -r requirements.txt
python cSimOpt.py
```

`if __name__ == '__main__':` block at the bottom of `cSimOpt.py` lets you adjust the experiment: surrogate (`gp`/`dpl`), acquisition function (`ei`/`ucb`), `max_cost`, `unit_cost`, and `seed`.

## Notes

- `layer_decoder`, `learning_rate_decoder`, and `sensor_parameter_decoder` convert encoded integer parameters into their real values (unit counts, learning rate, sensor settings).

## References

[1] Kadra, Arlind, et al. "Scaling laws for hyperparameter optimization." *Advances in Neural Information Processing Systems* 36 (2023): 47527-47553.

[2] S. Kim and J. Kim, "Parameter Optimization Framework for Enhancing Radar-Based Material Recognition," in *IEEE Sensors Journal*, vol. 24, no. 24, pp. 42219-42229, 15 Dec. 2024.

## License & Inqueries

Copyright © UIF (University Industry Foundation), Yonsei University. All rights reserved. For any inquiries regarding the usage or licensing of this code, please contact the UIF. (Contact: +82-2-2123-5176 / jemin17@yonsei.ac.kr)
