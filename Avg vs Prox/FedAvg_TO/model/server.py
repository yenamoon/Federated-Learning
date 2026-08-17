import torch
import os
from torch.autograd import Variable
from model.lenet import lenet5


class server(object):
    def __init__(self, size, data_loader):
        self.size = size
        self.dataset = 'mnist'
        self.test_loader = data_loader[1]
        self.path = './cache/global_model_state.pkl'

        os.makedirs('./cache', exist_ok=True)

        self.model = self.__init_server()
        self.accuracy = []
        self.loss = []
        self.participated = []  # ✅ 라운드별 참여 클라이언트 수 기록

    def __init_server(self):
        model = lenet5(self.dataset).cuda()
        torch.save(model.state_dict(), self.path)
        return model

    @staticmethod
    def __average_weights(clients_info):
        total_weights = {}
        n_total_samples = 0

        for info in clients_info:
            n_samples = info['n_samples']
            for k, v in info['state_dict'].items():
                v_cuda = v.cuda()
                if k not in total_weights:
                    total_weights[k] = v_cuda * n_samples
                else:
                    total_weights[k] += v_cuda * n_samples
            n_total_samples += n_samples

        averaged_weights = {}
        for k, v in total_weights.items():
            averaged_weights[k] = torch.div(v, n_total_samples)
        return averaged_weights

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

    def aggregate_network_mode(self, clients_info):
        averaged_weights = self.__average_weights(clients_info)
        self.model.load_state_dict(averaged_weights)
        torch.save(self.model.state_dict(), self.path)

        avg_loss = sum(info['loss'] for info in clients_info) / len(clients_info)
        self.loss.append(avg_loss)
        torch.save(self.loss, './cache/loss.pkl')

        # ✅ 참여 클라이언트 수 기록
        self.participated.append(len(clients_info))
        torch.save(self.participated, './cache/participated.pkl')
        print('[Global Model]  Avg Client Loss: {:.6f}  참여 클라이언트: {}/{}'.format(
            avg_loss, len(clients_info), self.size))

        test_accuracy = self.__test()
        self.accuracy.append(test_accuracy)
        torch.save(self.accuracy, './cache/accuracy.pkl')
        print('[Global Model]  Test Accuracy: {:.2f}%\n'.format(test_accuracy * 100.))