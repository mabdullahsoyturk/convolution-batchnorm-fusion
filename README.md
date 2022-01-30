# Convolution and Batch Normalization Fusion in PyTorch

To understand how it works, please refer to: https://nenadmarkus.com/p/fusing-batchnorm-and-conv/

## A few experiments

I ran both implementations for each input size 100 times.

Hardware: AMD Ryzen 9 5900HS

| Input Size        | Normal           | Fused  | Speedup |
| ------------- |:-------------:| -----:| -----:|
| (16, 3, 32, 32)      | 0.034 | 0.026 | 1.322x|
| (16, 3, 64, 64)      | 0.342     |   0.144 | 2.374x|
| (16, 3, 128, 128) | 1.484     |    0.854 |  1.738x|
| (16, 3, 256, 256) | 6.283     |    4.172 |  1.506x|

Hardware: GeForce RTX 3060 Mobile

| Input Size        | Normal           | Fused  | Speedup |
| ------------- |:-------------:| -----:| -----:|
| (16, 3, 32, 32)      | 0.542 | 0.005 | 111.341x|
| (16, 3, 64, 64)      | 0.014     |   0.007 | 2.093x|
| (16, 3, 128, 128) | 0.014     |    0.005 |  2.820x|
| (16, 3, 256, 256) | 0.013     |    0.006 |  2.183x|