import torch
import os
from torch import optim
from torch.autograd import Variable
from model.lenet import lenet5


class server(object):
    def __init__(self, size, data_loader, dataset='cifar10'):
        self.size = size
        self.dataset = dataset
        self.test_loader = data_loader[1]
        self.path = './cache/global_model_state.pkl'

        os.makedirs('./cache', exist_ok=True)

        self.model = self.__init_server()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.accuracy = []
        self.loss = []

    def __init_server(self):
        model = lenet5(self.dataset).cuda()
        torch.save(model.state_dict(), self.path)
        return model

    @staticmethod
    def __average_grads(grads_info):
        total_grads = {}
        n_total_samples = 0
        for info in grads_info:
            n_samples = info['n_samples']
            for k, v in info['named_grads'].items():
                v_cuda = v.cuda()
                if k not in total_grads:
                    total_grads[k] = v_cuda * n_samples
                else:
                    total_grads[k] += v_cuda * n_samples
            n_total_samples += n_samples

        gradients = {}
        for k, v in total_grads.items():
            gradients[k] = torch.div(v, n_total_samples)
        return gradients

    def __step(self, gradients):
        self.model.train()
        self.optimizer.zero_grad()
        for k, v in self.model.named_parameters():
            v.grad = gradients[k]
        self.optimizer.step()

    def __test(self):
        test_correct = 0
        self.model.eval()
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = Variable(data).cuda(), Variable(target).cuda()
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
        return test_correct / len(self.test_loader.dataset)

    def aggregate_network_mode(self, grads_info):
        gradients = self.__average_grads(grads_info)
        self.__step(gradients)
        torch.save(self.model.state_dict(), self.path)

        avg_loss = sum(info['loss'] for info in grads_info) / len(grads_info)
        self.loss.append(avg_loss)
        torch.save(self.loss, './cache/loss.pkl')
        print('[Global Model]  Avg Client Loss: {:.6f}'.format(avg_loss))

        test_accuracy = self.__test()
        self.accuracy.append(test_accuracy)
        torch.save(self.accuracy, './cache/accuracy.pkl')
        print('[Global Model]  Test Accuracy: {:.2f}%\n'.format(test_accuracy * 100.))