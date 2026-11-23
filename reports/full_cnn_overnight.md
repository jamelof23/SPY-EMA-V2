# SPY five-minute and daily chart CNNs

## Full-size CNN comparison

The complete standard CPU run saved 12 model checkpoints. It consumed zero Colab compute units. The lowest 2024 development loss occurred at epoch 1 for every architecture.

| 2025 architecture | Selected epoch | Accuracy | Balanced accuracy | AUC | Log loss |
|---|---:|---:|---:|---:|---:|
| Five-minute CNN | 1 | 52.40% | 50.00% | 0.5113 | 0.69184 |
| Paired CNN | 1 | 52.43% | 50.05% | 0.5100 | 0.69493 |
| Paired CNN plus price distances | 1 | 52.38% | 50.03% | 0.5029 | 0.70000 |

| Architecture | Epoch | Fit loss | Fit accuracy | 2024 loss | 2024 accuracy | 2024 AUC |
|---|---:|---:|---:|---:|---:|---:|
| Five-minute CNN | 1 | 0.69162 | 52.53% | 0.69158 | 52.34% | 0.5178 |
| Five-minute CNN | 2 | 0.69012 | 52.94% | 0.69245 | 51.75% | 0.5120 |
| Five-minute CNN | 3 | 0.68844 | 53.80% | 0.69280 | 52.10% | 0.5075 |
| Five-minute CNN | 4 | 0.68586 | 54.59% | 0.69632 | 51.20% | 0.5049 |
| Paired CNN | 1 | 0.69044 | 53.23% | 0.69442 | 52.22% | 0.5098 |
| Paired CNN | 2 | 0.68048 | 56.70% | 0.69934 | 51.17% | 0.5039 |
| Paired CNN | 3 | 0.66897 | 59.34% | 0.71562 | 49.40% | 0.4964 |
| Paired CNN | 4 | 0.66032 | 60.71% | 0.72487 | 49.46% | 0.5010 |
| Paired CNN plus price distances | 1 | 0.68953 | 53.56% | 0.69880 | 52.30% | 0.5046 |
| Paired CNN plus price distances | 2 | 0.67892 | 56.88% | 0.70275 | 50.35% | 0.4977 |
| Paired CNN plus price distances | 3 | 0.66631 | 59.71% | 0.72638 | 48.88% | 0.4934 |
| Paired CNN plus price distances | 4 | 0.65666 | 61.30% | 0.73112 | 48.76% | 0.4963 |

The paired model reached 60.71% training accuracy by epoch 4 while 2024 accuracy fell to 49.46%. The added price-distance features did not repair that overfitting on the 30-minute target. The saved archive includes optimizer state, each epoch, fixed 2025 predictions and the training log.

[Completed full-size training run](https://github.com/jamelof23/SPY-EMA-V2/actions/runs/45137557151)
