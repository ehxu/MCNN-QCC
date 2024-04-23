# -*- coding: utf-8 -*-
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt


# model trainer
def model_trainer(num_epochs, train_dataloader, test_dataloader, model, criterion, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # 将每次训练的训练误差和测试误差存储在这两个列表里，后面绘制误差变化折线图用
    train_losses, test_losses, train_accs, test_accs = [], [], [], []

    print('Start training')
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # for inputs, labels in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        # 每次学完一遍数据集，都进行以下测试操作
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            # for inputs, labels in tqdm(test_dataloader, desc='Test', leave=False):
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_test += labels.size(0)
                correct_test += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_dataloader.dataset)
        train_acc = 100. * correct_train / total_train
        test_loss = test_loss / len(test_dataloader.dataset)
        test_acc = 100. * correct_test / total_test

        # 恢复Dropout
        model.train()

        # 将训练误差和测试误差存在两个列表里，后面绘制误差变化折线图用
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print("Epoch: {}/{}  ".format(epoch + 1, num_epochs),
              "Train Loss: {:.4f}  ".format(train_loss),
              "Test Loss: {:.4f}  ".format(test_loss),
              "Model Test Accuracy: {:.3f}%".format(test_acc))

    print('Finished Training')

    return train_losses, test_losses, train_accs, test_accs


# Training and Test Loss
def draw_loss(train_losses, test_losses):
    # 训练次数
    epochs = range(1, len(train_losses) + 1)

    # 绘制训练误差和测试误差随学习次数增加的变化曲线
    # plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue', marker='o')
    plt.plot(epochs, test_losses, label='Test Loss', color='orange', marker='s')

    # 添加标签和标题
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)

    # 设置 x 轴刻度为每个 epoch 单位
    # plt.xticks(range(1, len(fc_train_losses) + 1))

    plt.show()


# Accuracy Curve
def draw_acc(train_accs, test_accs):
    # 训练次数
    epochs = range(1, len(train_accs) + 1)

    # 绘制准确率曲线图
    plt.plot(epochs, train_accs, label='Train Accuracy', color='blue')
    plt.plot(epochs, test_accs, label='Test Accuracy', color='orange')

    # 添加标签和标题
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)

    # 显示图像
    plt.show()


# Confusion Matrix
def draw_confusion(conf_matrix_percent, label_dic):
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_percent, annot=False, cmap='Purples', fmt=".2f", xticklabels=label_dic.values(),
                yticklabels=label_dic.values())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


# Classification accuracy
def draw_class_acc(class_accuracy, label_dic):
    plt.figure(figsize=(8, 6))
    bars = plt.bar(label_dic.values(), class_accuracy, color='skyblue')

    # 在每个条形上方添加文本标签
    for i, acc in enumerate(class_accuracy):
        plt.text(i, acc + 1, f'{acc:.2f}%', ha='center', va='bottom')

    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Classification accuracy', pad=20)
    plt.xticks(rotation=30)
    plt.ylim(0, 100)

    plt.show()


# Precision, Recall and F1 Score for each Class
def draw_prf1(precision, recall, f1, label_dic):
    # 绘制精确率、召回率和 F1 值的条形图
    x = np.arange(len(label_dic))
    width = 0.2
    label_dic = {0: 'separable states', 1: 'two-separable states', 2: 'entangled states'}
    fig, ax = plt.subplots(figsize=(10, 6))
    bars_precision = ax.bar(x - width, precision, width, label='Precision')
    bars_recall = ax.bar(x, recall, width, label='Recall')
    bars_f1 = ax.bar(x + width, f1, width, label='F1 Score')

    ax.set_xlabel('Class')
    ax.set_ylabel('Value')
    ax.set_title('Precision, Recall and F1 Score for each Class')
    ax.set_xticks(x)
    ax.set_xticklabels(label_dic.values(), rotation=30)

    # 添加数值标签
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate('%.2f' % height,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(bars_precision)
    autolabel(bars_recall)
    autolabel(bars_f1)
    # 移动图例到右边
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()
