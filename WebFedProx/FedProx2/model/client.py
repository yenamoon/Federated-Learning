import copy
import torch
import time
from torch import nn, optim
from torch.autograd import Variable
from model.lenet import lenet5


class client(object):
    def __init__(self, rank, data_loader, local_epoch=5, mu=1.0):
        seed = 19201077 + 19950920 + rank
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.rank = rank
        self.local_epoch = local_epoch
        self.mu = mu
        self.dataset = 'mnist'
        self.train_loader = data_loader[0]
        self.test_loader = data_loader[1]

    def load_global_model_from_memory(self, global_model_state):
        model = lenet5(self.dataset).cuda()
        model.load_state_dict(global_model_state)
        return model

    def __train(self, model, global_model, start_time, timeout):
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        avg_loss = 0
        completed_epochs = 0
        timeout_triggered = False

        for e in range(self.local_epoch):
            train_loss = 0
            train_correct = 0
            model.train()

            for data, target in self.train_loader:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    print(f'[Rank {self.rank}] timeout 감지 ({elapsed:.1f}초 경과, '
                          f'epoch {e+1} 진행 중 중단)')
                    timeout_triggered = True
                    break

                data, target = Variable(data).cuda(), Variable(target).cuda()
                optimizer.zero_grad()
                output = model(data)

                proximal_term = 0.0
                for w, w_t in zip(model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)

                loss = nn.CrossEntropyLoss()(output, target) + (self.mu / 2) * proximal_term
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()

            if timeout_triggered:
                break

            avg_loss = train_loss / len(self.train_loader)
            completed_epochs = e + 1
            print('[Rank {:>2}] Local Epoch {:>2}  Loss: {:>4.6f},  Accuracy: {:>.4f}'.format(
                self.rank, completed_epochs, avg_loss,
                train_correct / len(self.train_loader.dataset)
            ))

            elapsed = time.time() - start_time
            if elapsed >= timeout:
                break

        updated_state = {
            'n_samples': len(self.train_loader.dataset),
            'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
            'loss': avg_loss,
            'completed_epochs': completed_epochs,
            'local_epoch': self.local_epoch
        }
        return updated_state

    def run_network_mode(self, global_model_state, timeout):
        model = self.load_global_model_from_memory(global_model_state)

        global_model = copy.deepcopy(model)
        global_model.eval()
        for param in global_model.parameters():
            param.requires_grad = False

        start_time = time.time()
        result = self.__train(model=model, global_model=global_model,
                              start_time=start_time, timeout=timeout)
        return result
