# Convolution and Batch Normalization Fusion in PyTorch

To understand how it works, please refer to: https://nenadmarkus.com/p/fusing-batchnorm-and-conv/

## A few experiments

I ran both implementations for each input size 100 times. First 10 iterations are warm up iterations.

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
| (16, 3, 32, 32)      | 0.009 | 0.005 | 1.815x|
| (16, 3, 64, 64)      | 0.027     |   0.006 | 4.362x|
| (16, 3, 128, 128) | 0.024     |    0.005 |  4.387x|
| (16, 3, 256, 256) | 0.016     |    0.005 |  3.066x|