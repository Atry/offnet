import abc
import functools
import operator
import torch
from torch import nn


class LambdaModule(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class SequentialModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = None

    @abc.abstractproperty
    def layers(self):
        raise NotImplementedError

    def build(self):
        self.sequential = nn.Sequential(*self.layers())

    def forward(self, x):
        return self.sequential(x)


# ================
# Concrete Modules
# ================

class OffConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__()
        self.score = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.offset_x = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True
        )
        self.offset_y = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True
        )

    def forward(self, x):
        original_score = self.score(x)
        batch_size, number_of_channels, height, width = original_score.size()
        return nn.functional.grid_sample(
            original_score.view(-1, 1, height, width),
            (
                torch.nn.functional.affine_grid(torch.tensor((((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),), device=x.device), torch.Size((1, 1, height, width))) +
                torch.stack(
                    (
                        self.offset_x(x).view(-1, height, width),
                        self.offset_y(x).view(-1, height, width)
                    ),
                    dim=3
                ) * (2 / max(height, width))
            ).detach()
        ).view(batch_size, number_of_channels, height, width)


class FakeConv(nn.Module):
    def __init__(self, input_channels, output_channels, initial_offsets, stride=1, bias=False):
        super().__init__()
        (expand_multiplication, two) = initial_offsets.size()
        assert two == 2
        self.output_channels = output_channels
        self.expand_multiplication = expand_multiplication
        self.expand = OffConv2d(input_channels, output_channels * expand_multiplication, 1, stride=stride, padding=0, bias=bias)
        initial_x, initial_y = torch.unbind(initial_offsets, dim=1)
        def initialize_offset(bias, initial):
            with torch.no_grad():
                bias[:] = initial.unsqueeze(0).expand(output_channels, expand_multiplication).contiguous().view(output_channels * expand_multiplication)
        torch.nn.init.constant_(self.expand.offset_x.weight, 0.0)
        torch.nn.init.constant_(self.expand.offset_y.weight, 0.0)
        initialize_offset(self.expand.offset_x.bias, initial_x)
        initialize_offset(self.expand.offset_y.bias, initial_y)
#         self.bottleneck = OffConv2d(output_channels * expand_multiplication, output_channels, 1, stride=1, padding=0, bias=bias)
#     def forward(self, x):
#         return self.bottleneck(self.expand(x))
    def forward(self, x):
        expanded = self.expand(x)
        batch_size, channels, height, width = expanded.size()
        return expanded.view(batch_size, self.output_channels, self.expand_multiplication, height, width).sum(2)



class OffBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride):
        super().__init__()

        # 1
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = FakeConv(
            input_channels, output_channels,
            initial_offsets=torch.tensor(tuple((x, y) for x in range(-1, 2) for y in range(-1, 2))), stride=stride, bias=False
        )
        # 2
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = FakeConv(
            output_channels, output_channels,
            initial_offsets=torch.Tensor(tuple((x, y) for x in range(-1, 2) for y in range(-1, 2))), stride=1, bias=False
        )
        # transformation
        self.need_transform = input_channels != output_channels
        self.conv_transform = nn.Conv2d(
            input_channels, output_channels,
            kernel_size=1, stride=stride, padding=0, bias=False
        ) if self.need_transform else None

    def forward(self, x):
        x_nonlinearity_applied = self.relu1(self.bn1(x))
        y = self.conv1(x_nonlinearity_applied)
        y = self.conv2(self.relu2(self.bn2(y)))
        return y.add_(self.conv_transform(x) if self.need_transform else x)


class OffBlockGroup(SequentialModule):
    def __init__(self, block_number, input_channels, output_channels, stride):
        super().__init__()
        self.block_number = block_number
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.build()

    def layers(self):
        return [
            OffBlock(
                self.input_channels if i == 0 else self.output_channels,
                self.output_channels,
                self.stride if i == 0 else 1
            ) for i in range(self.block_number)
        ]


class Offnet(SequentialModule):
    def __init__(self, label, input_size, input_channels, classes,
                 total_block_number, widen_factor=1,
                 baseline_strides=None,
                 baseline_channels=None):
        super().__init__()
        
        # model name label.
        self.label = label
        
        # data specific hyperparameters.
        self.input_size = input_size
        self.input_channels = input_channels
        self.classes = classes
        
        # model hyperparameters.
        self.total_block_number = total_block_number
        self.widen_factor = widen_factor
        self.baseline_strides = baseline_strides or [1, 1, 2, 2]
        self.baseline_channels = baseline_channels or [16, 16, 32, 64]
        self.widened_channels = [
            w*widen_factor if i != 0 else w for i, w in
            enumerate(self.baseline_channels)
        ]
        self.group_number = len(self.widened_channels) - 1
        
        # validate total block number.
        assert len(self.baseline_channels) == len(self.baseline_strides)
        assert (
            self.total_block_number % (2*self.group_number) == 0 and
            self.total_block_number // (2*self.group_number) >= 1
        ), 'Total number of residual blocks should be multiples of 2 x N.'
        
        print(self.layers())
        
        # build the sequential model.
        self.build()
    
    @property
    def name(self):
        return (
            'WRN-{depth}-{widen_factor}-'
            '{label}-{size}x{size}x{channels}'
        ).format(
            depth=(self.total_block_number+4),
            widen_factor=self.widen_factor,
            size=self.input_size,
            channels=self.input_channels,
            label=self.label,
        )
    
    def layers(self):
        # define group configurations.
        blocks_per_group = self.total_block_number // self.group_number
        zipped_group_channels_and_strides = zip(
            self.widened_channels[:-1],
            self.widened_channels[1:],
            self.baseline_strides[1:]
        )

        # convolution layer.
        conv = nn.Conv2d(
            self.input_channels, self.widened_channels[0],
            kernel_size=3, stride=1, padding=1, bias=False
        )

        # residual block groups.
        residual_block_groups = [
            OffBlockGroup(blocks_per_group, i, o, s) for
            i, o, s in zipped_group_channels_and_strides
        ]

        # batchnorm & nonlinearity & pooling.
        bn = nn.BatchNorm2d(self.widened_channels[self.group_number])
        relu = nn.ReLU(inplace=True)
        pool = nn.AvgPool2d(
            self.input_size //
            functools.reduce(operator.mul, self.baseline_strides)
        )

        # classification scores from linear combinations of features.
        view = LambdaModule(lambda x: x.view(-1, self.widened_channels[-1]))
        fc = nn.Linear(self.widened_channels[self.group_number], self.classes)
        
        # softmax
        softmax = LambdaModule(lambda x: nn.functional.softmax(x, dim=1))

        # the final model structure.
        return [conv, *residual_block_groups, pool, bn, relu, view, fc, softmax]

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride):
        super().__init__()

        # 1
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            input_channels, output_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        # 2
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            output_channels, output_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        # transformation
        self.need_transform = input_channels != output_channels
        self.conv_transform = nn.Conv2d(
            input_channels, output_channels,
            kernel_size=1, stride=stride, padding=0, bias=False
        ) if self.need_transform else None

    def forward(self, x):
        x_nonlinearity_applied = self.relu1(self.bn1(x))
        y = self.conv1(x_nonlinearity_applied)
        y = self.conv2(self.relu2(self.bn2(y)))
        return y.add_(self.conv_transform(x) if self.need_transform else x)


class ResidualBlockGroup(SequentialModule):
    def __init__(self, block_number, input_channels, output_channels, stride):
        super().__init__()
        self.block_number = block_number
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.build()

    def layers(self):
        return [
            ResidualBlock(
                self.input_channels if i == 0 else self.output_channels,
                self.output_channels,
                self.stride if i == 0 else 1
            ) for i in range(self.block_number)
        ]


class WideResNet(SequentialModule):
    def __init__(self, label, input_size, input_channels, classes,
                 total_block_number, widen_factor=1,
                 baseline_strides=None,
                 baseline_channels=None):
        super().__init__()

        # model name label.
        self.label = label

        # data specific hyperparameters.
        self.input_size = input_size
        self.input_channels = input_channels
        self.classes = classes

        # model hyperparameters.
        self.total_block_number = total_block_number
        self.widen_factor = widen_factor
        self.baseline_strides = baseline_strides or [1, 1, 2, 2]
        self.baseline_channels = baseline_channels or [16, 16, 32, 64]
        self.widened_channels = [
            w*widen_factor if i != 0 else w for i, w in
            enumerate(self.baseline_channels)
        ]
        self.group_number = len(self.widened_channels) - 1

        # validate total block number.
        assert len(self.baseline_channels) == len(self.baseline_strides)
        assert (
            self.total_block_number % (2*self.group_number) == 0 and
            self.total_block_number // (2*self.group_number) >= 1
        ), 'Total number of residual blocks should be multiples of 2 x N.'

        # build the sequential model.
        self.build()

    @property
    def name(self):
        return (
            'WRN-{depth}-{widen_factor}-'
            '{label}-{size}x{size}x{channels}'
        ).format(
            depth=(self.total_block_number+4),
            widen_factor=self.widen_factor,
            size=self.input_size,
            channels=self.input_channels,
            label=self.label,
        )

    def layers(self):
        # define group configurations.
        blocks_per_group = self.total_block_number // self.group_number
        zipped_group_channels_and_strides = zip(
            self.widened_channels[:-1],
            self.widened_channels[1:],
            self.baseline_strides[1:]
        )

        # convolution layer.
        conv = nn.Conv2d(
            self.input_channels, self.widened_channels[0],
            kernel_size=3, stride=1, padding=1, bias=False
        )

        # residual block groups.
        residual_block_groups = [
            ResidualBlockGroup(blocks_per_group, i, o, s) for
            i, o, s in zipped_group_channels_and_strides
        ]

        # batchnorm & nonlinearity & pooling.
        bn = nn.BatchNorm2d(self.widened_channels[self.group_number])
        relu = nn.ReLU(inplace=True)
        pool = nn.AvgPool2d(
            self.input_size //
            functools.reduce(operator.mul, self.baseline_strides)
        )

        # classification scores from linear combinations of features.
        view = LambdaModule(lambda x: x.view(-1, self.widened_channels[-1]))
        fc = nn.Linear(self.widened_channels[self.group_number], self.classes)
        
        # softmax
        softmax = LambdaModule(lambda x: nn.functional.softmax(x, dim=1))

        # the final model structure.
        return [conv, *residual_block_groups, pool, bn, relu, view, fc, softmax]

