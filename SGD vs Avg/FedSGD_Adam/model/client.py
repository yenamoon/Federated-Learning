import torch
from torch import nn, optim
from torch.autograd import Variable
from model.lenet import lenet5


class client(object):
    def __init__(self, rank, data_loader, dataset='cifar10'):
        seed = 19201077 + 19950920 + rank
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.rank = rank
        self.dataset = dataset
        self.train_loader = data_loader[0]
        self.test_loader = data_loader[1]

    def load_global_model_from_memory(self, global_model_state):
        model = lenet5(self.dataset).cuda()
        model.load_state_dict(global_model_state)
        return model

    def __train(self, model):
        train_loss = 0
        train_correct = 0
        model.train()

        optimizer = optim.SGD(model.parameters(), lr=0.0)
        optimizer.zero_grad()

        for data, target in self.train_loader:
            data, target = Variable(data).cuda(), Variable(target).cuda()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            train_loss += loss.item()
            loss.backward()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()

        avg_loss = train_loss / len(self.train_loader)

        grads = {
            'n_samples': len(self.train_loader.dataset),
            'named_grads': {},
            'loss': avg_loss
        }
        for name, param in model.named_parameters():
            # ✅ 배치 수로 나눠 평균 gradient (정규화)
            grads['named_grads'][name] = param.grad.cpu() / len(self.train_loader) if param.grad is not None else None

        print('[Rank {:>2}]  Loss: {:>4.6f},  Accuracy: {:>.4f}'.format(
            self.rank,
            avg_loss,
            train_correct / len(self.train_loader.dataset)
        ))
        return grads

    def run_network_mode(self, global_model_state):
        model = self.load_global_model_from_memory(global_model_state)
        grads = self.__train(model=model)
        return grads