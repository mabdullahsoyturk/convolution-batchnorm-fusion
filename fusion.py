import torch
import torchvision
import time

def fuse_convolution_and_batchnorm(convolution, batchnorm):
    fused_convolution = torch.nn.Conv2d(
        convolution.in_channels,
        convolution.out_channels,
        kernel_size=convolution.kernel_size,
        stride=convolution.stride,
        padding=convolution.padding,
        bias=True
    ).cuda()

    # prepare filters
    conv_weights = convolution.weight.clone().view(convolution.out_channels, -1).cuda()
    batchnorm_weights = torch.diag(batchnorm.weight.div(torch.sqrt(batchnorm.eps + batchnorm.running_var)))
    updated_weight = torch.mm(batchnorm_weights, conv_weights).view(fused_convolution.weight.size())
    
    # copy updated weights
    fused_convolution.weight.copy_(updated_weight)

    # prepare bias
    conv_bias = torch.zeros(convolution.weight.size(0)).cuda()
    batchnorm_bias = batchnorm.bias - batchnorm.weight.mul(batchnorm.running_mean).div(torch.sqrt(batchnorm.running_var + batchnorm.eps))
    updated_bias = torch.matmul(batchnorm_weights, conv_bias) + batchnorm_bias

    # copy updated bias
    fused_convolution.bias.copy_(updated_bias)

    return fused_convolution

def main():
    torch.set_grad_enabled(False)

    input_sizes = [input_size for input_size in range(32, 257, 32)]

    rn18 = torchvision.models.resnet18(pretrained=True)
    rn18.cuda()

    net = torch.nn.Sequential(
        rn18.conv1,
        rn18.bn1
    )

    net.cuda()

    for size in input_sizes:
        x = torch.randn(16, 3, size, size).cuda()

        # normal run
        normal_time = 0
        y1 = None
        for iteration in range(100):
            start = time.time()
            y1 = net.forward(x)
            end = time.time()

            if iteration >= 10:
                normal_time += end - start
        # print(f'Normal run took: {(normal_time / 90):.3f} seconds on average')

        # fused run
        fused_convolution = fuse_convolution_and_batchnorm(net[0], net[1])

        fused_time = 0
        y2 = None
        for iteration in range(100):
            start = time.time()
            y2 = fused_convolution.forward(x)
            end = time.time()
            if iteration >= 10:
                fused_time += end - start
        # print(f'Fused run took: {(fused_time / 90):.3f} seconds on average')

        d = (y1 - y2).norm().div(y1.norm()).item()
        print("error: %.8f" % d)
        
        print(f'For input (16, 3, {size}, {size}), normal time: {normal_time:.3f}, fused_time: {fused_time:.3f} speedup: {(normal_time / fused_time):.3f}x')


if __name__ == '__main__':
    main()
