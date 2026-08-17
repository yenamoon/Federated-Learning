import torch
from torch import nn, optim
from torch.autograd import Variable
from model.lenet import lenet5


class client(object):
    def __init__(self, rank, data_loader, local_epoch=5):
        seed = 19201077 + 19950920 + rank
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.rank = rank
        self.local_epoch = local_epoch
        self.dataset = 'mnist'
        self.train_loader = data_loader[0]
        self.test_loader = data_loader[1]

    def load_global_model_from_memory(self, global_model_state):
        model = lenet5(self.dataset).cuda()
        model.load_state_dict(global_model_state)
        return model

    def __train(self, model):
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        avg_loss = 0  # ✅ 마지막 epoch의 loss 저장용

        for e in range(self.local_epoch):
            train_loss = 0
            train_correct = 0
            model.train()

            for data, target in self.train_loader:
                data, target = Variable(data).cuda(), Variable(target).cuda()
                optimizer.zero_grad()
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()

            avg_loss = train_loss / len(self.train_loader)  # ✅ 매 epoch마다 갱신, 마지막 값이 남음
            print('[Rank {:>2}] Local Epoch {:>2}  Loss: {:>4.6f},  Accuracy: {:>.4f}'.format(
                self.rank,
                e + 1,
                avg_loss,
                train_correct / len(self.train_loader.dataset)
            ))

        updated_state = {
            'n_samples': len(self.train_loader.dataset),
            'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
            'loss': avg_loss  # ✅ 추가
        }
        return updated_state

    def run_network_mode(self, global_model_state):
        model = self.load_global_model_from_memory(global_model_state)
        updated_state = self.__train(model=model)
        return updated_state