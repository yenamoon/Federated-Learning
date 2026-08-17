import torch
from matplotlib import pyplot as plt


def plot():
    accuracy = torch.load('./cache/accuracy.pkl', weights_only=True)  # ✅ weights_only 추가
    plt.plot([e for e in range(1, len(accuracy) + 1)], accuracy, label='FedAvg')  # ✅ 라벨 FedAvg로 변경

    plt.title("Test Accuracy")
    plt.xlabel("round")  # ✅ epoch → round
    plt.ylabel("accuracy")

    plt.ylim(0, 1)
    plt.xlim(1, len(accuracy))
    plt.legend(loc=4)

    plt.show()