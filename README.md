# Convolution and Batch Normalization Fusion in PyTorch

To understand how it works, please refer to: https://nenadmarkus.com/p/fusing-batchnorm-and-conv/

## A few experiments

Colons can be used to align columns.

| Input Size        | Normal           | Fused  | Speedup |
| ------------- |:-------------:| -----:| -----:|
| (16, 3, 32, 32)      | 0.034 | 0.026 | 1.322x|
| (16, 3, 64, 64)      | 0.342     |   0.144 | 2.374x|
| (16, 3, 128, 128) | 1.484     |    0.854 |  1.738x|
| (16, 3, 256, 256) | 6.283     |    4.172 |  1.506x|