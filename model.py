"""
В качестве практики реализуем базовую линейную свёрточную сеть VGG-16 или версию D из статьи Karen Simonyan и Andrew
 Zisserman (https://arxiv.org/abs/1409.1556), которая в соревнованиях ILSVRC 2014 показала отличные результаты и стала
 основной для множества сетей в будущем.
Добавим batch normalization для более быстрого обучения сети (https://arxiv.org/abs/1502.03167).
"""
from torch import flatten
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self, num_classes, init_weights=True):
        super(VGG16, self).__init__()
        self.features = VGG16.get_features()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = VGG16.get_classifier(num_classes)

        if init_weights:
            self._initialize_weights()

    @staticmethod
    def conv_relu_bn(in_channels: int, out_channels: int, use_bn: bool):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        if use_bn:
            return [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        else:
            return [conv2d, nn.ReLU(inplace=True)]

    @staticmethod
    def get_features():
        layers = []
        layers += VGG16.conv_relu_bn(3, 64, True)
        layers += VGG16.conv_relu_bn(64, 64, True)
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers += VGG16.conv_relu_bn(64, 128, True)  # 112x112x128
        layers += VGG16.conv_relu_bn(128, 128, True)  # 112x112x128
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers += VGG16.conv_relu_bn(128, 256, True)  # 56x56x256
        layers += VGG16.conv_relu_bn(256, 256, True)
        layers += VGG16.conv_relu_bn(256, 256, True)
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers += VGG16.conv_relu_bn(256, 512, True)
        layers += VGG16.conv_relu_bn(512, 512, True)
        layers += VGG16.conv_relu_bn(512, 512, True)
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers += VGG16.conv_relu_bn(512, 512, True)
        layers += VGG16.conv_relu_bn(512, 512, True)
        layers += VGG16.conv_relu_bn(512, 512, True)
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)

    @staticmethod
    def get_classifier(num_classes):
        return nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)  # получаем тензор высокоуровневых признаков
        x = self.avgpool(x)  # объединяем все каналы в один
        x = flatten(x, 1)  # вытягиваем матрицу в вектор для отдачи данных в полносвязные слои
        x = self.classifier(x)  # классифицируем признаки
        return x

    def _initialize_weights(self):
        # Переиспользуем подход из PyTorch
        # Веса свёрточных слоёв инициализируем, как описано в статье https://arxiv.org/abs/1502.01852, где авторы
        # доказывали, что их подход лучше обычного нормального распределения с фиксированным cреднеквадратичным
        # отклонением.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)




