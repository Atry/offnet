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
class Deform2d(torch.nn.Module):
    def __init__(self, number_in_channels, number_of_filters,
                 kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            number_in_channels,
            2 * number_of_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.conv.weight, 0.0)
        torch.nn.init.normal_(self.conv.weight)

    @functools.lru_cache(maxsize=None)
    def identity_grid(device_index, height, width):
        return torch.nn.functional.affine_grid(torch.tensor((((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),), device=torch.device(device_index)), torch.Size((1, 1, height, width)))

    def forward(self, x):
        batch_size, number_in_channels, height, width = x.size()
        deformation = self.conv(x) * (2 / max(height, width))
        batch_size, uv, height, width = deformation.size()
        number_of_filters = uv // 2
        deformation.view(batch_size * number_of_filters, 2, height, width).permute(0, 2, 3, 1)
        
        expanded_input = x.unsqueeze(1).expand(batch_size, number_of_filters, number_in_channels, height, width).contiguous().view(batch_size * number_of_filters, number_in_channels, height, width)
        
        grid = Deform2d.identity_grid(x.device.index, height, width) + deformation \
            .view(batch_size * number_of_filters, 2, height, width) \
            .permute(0, 2, 3, 1)

        return torch.nn.functional.grid_sample(
            input=expanded_input,
            grid=grid.detach()
        ).view(batch_size, number_of_filters * number_in_channels, height, width)

class FakeConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        kernel_height, kernel_width = torch.nn.modules.utils._pair(kernel_size)
        self.deform = Deform2d(in_channels, kernel_height * kernel_width)
        self.conv = torch.nn.Conv2d(in_channels * kernel_height * kernel_width, out_channels, 1, stride=stride, bias=bias)
        def generate_offset():
            half_kernel_height = kernel_height // 2
            half_kernel_width = kernel_width // 2
            kernel_y_range = range(-half_kernel_height, kernel_height - half_kernel_height)
            kernel_x_range = range(-half_kernel_width, kernel_width - half_kernel_width)
            for y in kernel_y_range:
                for x in kernel_x_range:
                    yield x
            for y in kernel_y_range:
                for x in kernel_x_range:
                    yield y
        with torch.no_grad():
            torch.nn.init.constant_(self.deform.conv.weight, 0.0)
            self.deform.conv.bias[:] = torch.tensor(tuple(generate_offset()))

    def forward(self, x):
        return self.conv(self.deform(x))

class OffBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        # 1
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = FakeConv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, bias=False
        )
        # 2
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = FakeConv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, bias=False
        )
        # transformation
        self.need_transform = in_channels != out_channels
        self.conv_transform = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=stride, padding=0, bias=False
        ) if self.need_transform else None

    def forward(self, x):
        x_nonlinearity_applied = self.relu1(self.bn1(x))
        y = self.conv1(x_nonlinearity_applied)
        y = self.conv2(self.relu2(self.bn2(y)))
        return y.add_(self.conv_transform(x) if self.need_transform else x)


class OffBlockGroup(SequentialModule):
    def __init__(self, block_number, in_channels, out_channels, stride):
        super().__init__()
        self.block_number = block_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.build()

    def layers(self):
        return [
            OffBlock(
                self.in_channels if i == 0 else self.out_channels,
                self.out_channels,
                self.stride if i == 0 else 1
            ) for i in range(self.block_number)
        ]


class Offnet(SequentialModule):
    def __init__(self, label, input_size, in_channels, classes,
                 total_block_number, widen_factor=1,
                 baseline_strides=None,
                 baseline_channels=None):
        super().__init__()
        
        # model name label.
        self.label = label
        
        # data specific hyperparameters.
        self.input_size = input_size
        self.in_channels = in_channels
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
            channels=self.in_channels,
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
            self.in_channels, self.widened_channels[0],
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
        
        # the final model structure.
        return [conv, *residual_block_groups, pool, bn, relu, view, fc]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        # 1
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        # 2
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        # transformation
        self.need_transform = in_channels != out_channels
        self.conv_transform = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=stride, padding=0, bias=False
        ) if self.need_transform else None

    def forward(self, x):
        x_nonlinearity_applied = self.relu1(self.bn1(x))
        y = self.conv1(x_nonlinearity_applied)
        y = self.conv2(self.relu2(self.bn2(y)))
        return y.add_(self.conv_transform(x) if self.need_transform else x)


class ResidualBlockGroup(SequentialModule):
    def __init__(self, block_number, in_channels, out_channels, stride):
        super().__init__()
        self.block_number = block_number
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.build()

    def layers(self):
        return [
            ResidualBlock(
                self.in_channels if i == 0 else self.out_channels,
                self.out_channels,
                self.stride if i == 0 else 1
            ) for i in range(self.block_number)
        ]


class WideResNet(SequentialModule):
    def __init__(self, label, input_size, in_channels, classes,
                 total_block_number, widen_factor=1,
                 baseline_strides=None,
                 baseline_channels=None):
        super().__init__()

        # model name label.
        self.label = label

        # data specific hyperparameters.
        self.input_size = input_size
        self.in_channels = in_channels
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
            channels=self.in_channels,
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
            self.in_channels, self.widened_channels[0],
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
        
        # the final model structure.
        return [conv, *residual_block_groups, pool, bn, relu, view, fc]

