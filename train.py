from __future__ import print_function
import torch.nn.functional as F
from torch.autograd import Variable


def train(loader, model, optimizer, epoch, cuda, log_interval, verbose=True):
    '''
    #############################################################################
    train the model, you can write this function partly refer to the "test" below
    Args:
        loader: torch dataloader
        model: model to train
        optimizer: torch optimizer
        epoch: number of epochs to train
        cuda: whether to use gpu
        log_interval: how many batches to wait before logging training status
        verbose: whether to print training log(such as epoch and loss)
    Return:
        the average loss of this epoch
    #############################################################################
    '''
    

    model.train()      # 设置为trainning模式
    for batch_idx, (data, target) in enumerate(loader):
        if cuda:  # 如果要调用GPU模式，就把数据转存到GPU
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)  # 把数据转换成Variable
        optimizer.zero_grad()  # 优化器梯度初始化为零
#         print(type(data))
#         print(data)
#         print(data.shape)
        output = model(data)   # 把数据输入网络并得到输出，即进行前向传播
#         print(output.shape)
#         print(output)
#         print(target.shape)
        loss = F.nll_loss(output, target)               # 计算损失函数  
        loss.backward()        # 反向传播梯度
        optimizer.step()       # 结束一次前传+反传之后，更新优化器参数
#         print(batch_idx)
#         print(log_interval)
#         print(epoch)
#         print(loss.data[1])
#         print(type(loss.data))
        if batch_idx % log_interval == 0:          # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                100. * batch_idx / len(loader), loss.item()))








def test(loader, model, cuda, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data.item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(loader.dataset)
    if verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(loader.dataset), 100. * correct / len(loader.dataset)))
    return test_loss
