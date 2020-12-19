"""
Домашняя работа по глубоким нейронным сетям Ерасовой Елены
"""
import os  # для работы с файловой системой
import sys  # для максимального значения
import argparse  # нужен для чтения параметров запуска программы обучения
import torch  # основной фреймворк обучения
import torchvision  # нужен для работы с CIFAR-10
import torchvision.transforms as transforms  # предобработка изображений
from pytorch_model_summary import summary
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter  # запись хода эксперимента
from utils import imshow
from model import VGG16


def show_random_images(trainloader, classes: tuple, num_images: int):
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(' '.join('%5s' % classes[labels[j]] for j in range(num_images)))
    imshow(torchvision.utils.make_grid(images))


def get_data(dataset: str, num_workers: int, batch_size: int):
    if dataset == 'cifar-10':
        # Эксперименты показали, что свёрточные сети лучше обучаются, если данные подаются в нормализованном виде

        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, 4),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        transform_val = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                  num_workers=num_workers, pin_memory=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                                 num_workers=num_workers, pin_memory=True)
        # Все классы из CIFAR-10
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        return trainloader, testloader, classes
    elif dataset == 'cifar_superclass':
        raise Exception('Not implemented')


def evaluate(net, criterion, testloader, device):
    net.eval()
    running_loss = 0.0
    num_minibatches = 0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            # прямой проход
            outputs = net(inputs)
            # функция потерь
            loss = criterion(outputs, labels)



            running_loss += loss.item()
            num_minibatches += 1
    net.train()
    return running_loss / num_minibatches


def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    confusion_matrix = [tp, tn, fp, fn]

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    result = {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy,
              'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}}
    return result


def train_cifar10(args, test_only=False):
    # Настройка tensorboard
    writer = SummaryWriter(os.path.join(args.log_dir, 'tensorboard_log'))
    model_save_path = os.path.join(args.log_dir, 'cifar10_VGG16_best.pth')

    # Выбор устройства для обучения
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} for training')
    using_cuda = False
    if 'cuda' in str(device):
        cudnn.benchmark = True  # CuDNN ищет самые оптимальные алгоритмы для текущего железа, что может ускорить работу
        using_cuda = True

    # Создаём сеть

    net = models.vgg16_bn(pretrained=False, num_classes=10)
    # net = models.vg
    # net = models.resnet18(pretrained=False, num_classes=10)
    # net = VGG16(10, True)
    print(summary(net, torch.zeros((1, 3, 32, 32)), show_input=True, show_hierarchical=True))
    net.to(device)
    net.train()

    # Загружаем данные
    trainloader, testloader, classes = get_data('cifar-10', args.num_workers, args.batch_size)

    if args.debug:
        # Показываем изображения из первого батча, чтобы удостовериться, что есть верный доступ к данным
        show_random_images(trainloader, classes, args.batch_size)

    criterion = nn.CrossEntropyLoss()
    if using_cuda:
        criterion.cuda()
    # Используем amsgrad версию оптимизатора, т. к. в статье https://arxiv.org/abs/1904.09237 показана более
    # стабильная работа этого варианта Adam.
    optimizer = optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=True)

    # Для более точного приближения к минимуму цели оптимизации, следует уменьшать learning rate в ходе обучения
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2)

    # Цикл обучения
    if not test_only:
        print(f'Starting training for {args.epochs} epochs')
    best_val_loss = sys.maxsize
    best_accuracy = -1
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}')
        for param_group in optimizer.param_groups:
            print(f'Current learning rate = {param_group["lr"]}')

        if test_only:
            break
        running_loss = 0.0
        epoch_loss = 0.0
        num_minibatches = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            # обнуляем градиенты
            optimizer.zero_grad()

            # прямой проход
            outputs = net(inputs)
            # функция потерь
            loss = criterion(outputs, labels)
            # обратный проход
            loss.backward()
            # обновление весов сети
            optimizer.step()
            # точность
            _, predicted = torch.max(outputs.data, 1)
            metrics = f1_loss(predicted, labels, True)
            # correct += (predicted == labels).sum().item()
            precision = metrics['precision']
            recall = metrics['recall']
            f1 = metrics['f1']
            accuracy = metrics['accuracy']
            confusion_matrix = metrics['confusion_matrix']

            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()
            num_minibatches += 1
            if i % 200 == 199 and args.debug:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
        val_loss = evaluate(net, criterion, testloader, device)
        lr_scheduler.step(val_loss)
        epoch_loss = epoch_loss / num_minibatches
        print(f'Epoch {epoch + 1} training loss = {epoch_loss}')
        print(f'Epoch {epoch + 1} accuracy = {accuracy}')
        print(f'Epoch {epoch + 1} validation loss = {val_loss}')
        print(f'Epoch {epoch + 1} precision = {precision}')
        print(f'Epoch {epoch + 1} recall = {recall}')
        print(f'Epoch {epoch + 1} f1 = {f1}')
        for key in confusion_matrix:
            print(f'Epoch {epoch + 1} confusion matrix {key} = {confusion_matrix[key]}')

        writer.add_scalar('training_loss',
                          epoch_loss,
                          epoch + 1)
        writer.add_scalar('validation_loss',
                          val_loss,
                          epoch + 1)
        writer.add_scalar('accuracy',
                          accuracy,
                          epoch + 1)

        if best_val_loss > val_loss:
            print(f'Loss improved from {best_val_loss} to {val_loss}. Saving model to {model_save_path}.')
            best_val_loss = val_loss
            torch.save(net.state_dict(), model_save_path)

        if best_accuracy < accuracy:
            print(f'Accuracy improved from {best_accuracy} to {accuracy}.')
            best_accuracy = accuracy
            # torch.save(net.state_dict(), model_save_path)

    if not test_only:
        print('Finished Training')
        torch.save(net.state_dict(), os.path.join(args.log_dir, 'cifar10_VGG16_last.pth'))

    print('Testing the final model on the validation set')
    # net = VGG16(10, True)
    # net.to(device)
    net.load_state_dict(torch.load(model_save_path))
    net.eval()

    correct = 0
    total = 0
    accuracy_final = 0
    temp = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            metrics = f1_loss(predicted, labels, True)
            accuracy_final += metrics['accuracy']
            temp += 1

    print(f'Accuracy of the network on the test images: {(100 * correct / total)}%')
    # print(f'Accuracy of the network on the test images (conf. matrix): {accuracy_final/temp}%')

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    # Описание параметров запуска скрипта принято делать на английском, чтобы с серверах с Linux не было проблем с
    # отображением текста
    parser = argparse.ArgumentParser(description='CNN training parameters')
    parser.add_argument('-n', '--num_workers', type=int, default=5,
                        help='Number of CPU workers')
    parser.add_argument('-b', '--batch_size', type=int, default=96,
                        help='Minibatch size')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Show debug information and images')
    parser.add_argument('-s', '--stage', type=int, required=True,
                        help='Homework stage')
    parser.add_argument('-e', '--epochs', type=int, default=1000,
                        help='Number of epochs for training')
    parser.add_argument('-l', '--log_dir', type=str, default='log',
                        help='Directory to save results in')
    args = parser.parse_args()

    # Cоздаём папку, куда будем сохранять результаты работы сети. Если папка существует, не пересоздаём.
    os.makedirs(args.log_dir, exist_ok=True)

    if args.stage == 1:
        train_cifar10(args)
    elif args.stage == 2:
        raise Exception('Not implemented')
    else:
        print(f'Stage {args.stage} does not exist in the task')
