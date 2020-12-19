"""
Тестирование обученное модели
"""
import argparse
import torch
import torch.backends.cudnn as cudnn
from train import get_data
from model import VGG16
from pytorch_model_summary import summary
import torchvision.models as models

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


def test_cifar10(args):
    # Выбор устройства для теста
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')
    using_cuda = False
    if 'cuda' in str(device):
        cudnn.benchmark = True
        using_cuda = True

    # Загружаем данные
    trainloader, testloader, classes = get_data('cifar-10', args.num_workers, args.batch_size)

    # net = models.resnet18(pretrained=False, num_classes=10)
    net = models.vgg16_bn(pretrained=False, num_classes=10)
    # net = VGG16(10, True)
    print(summary(net, torch.zeros((1, 3, 32, 32)), show_input=True, show_hierarchical=True))
    net.load_state_dict(torch.load(args.model))
    net.to(device)
    net.eval()

    correct = 0
    total = 0
    accuracy_final = 0
    temp = 0
    precision = 0
    recall = 0
    f1 = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            metrics = f1_loss(predicted, labels, True)
            accuracy_final += metrics['accuracy']
            precision += metrics['precision']
            recall += metrics['recall']
            f1 += metrics['f1']
            temp += 1

    print(f'Accuracy of the network on the test images: {(100 * correct / total)}%')
    print(f'precision of the network on the test images: {(precision / temp)}%')
    print(f'recall of the network on the test images: {(recall / temp)}%')
    print(f'f1 of the network on the test images: {(f1 / temp)}%')

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
    parser = argparse.ArgumentParser(description='CNN test parameters')
    parser.add_argument('-n', '--num_workers', type=int, default=4,
                        help='Number of CPU workers')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Minibatch size')
    parser.add_argument('-s', '--stage', type=int,
                        help='Homework stage')
    parser.add_argument('-m', '--model', type=str, default='log/cifar10_best.pth',
                        help='Path to the model to load')
    args = parser.parse_args()

    if args.stage == 1:
        test_cifar10(args)
