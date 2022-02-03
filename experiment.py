import os

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt


from utils.seed_func import fix_seed
from dataset import load_inner_speech_dataset, InnerSpeechDataset
from model import EEGNet

try:
    SEED = 42
    fix_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sub_num = 1

    # create datasets and dataloaders (4-folds)
    data_paths = []
    for ses_num in range(1, 4):
        sub_str = str(sub_num).zfill(2)
        ses_str = str(ses_num).zfill(2)
        data_path = 'dataset/derivatives/sub-{}/ses-{}/sub-{}_ses-{}_eeg-epo.fif'.format(
            sub_str, ses_str, sub_str, ses_str
        )
        data_paths.append(data_path)

    data, labels = load_inner_speech_dataset(data_paths)

    dataset_dict = {'train': [], 'valid': [], 'test': []}
    dataloader_dict = {'train': [], 'valid': [], 'test': []}

    phases = ['train', 'valid', 'test']
    ids = np.array([i for i in range(len(data))])
    kf = KFold(n_splits = 4, shuffle = True, random_state=SEED)
    for ids_train_valid, ids_test in kf.split(ids):
        np.random.shuffle(ids_train_valid)
        # train : valid = 85 : 15
        ids_train = ids_train_valid[:int(len(ids_train_valid)*17/20//1)]
        ids_valid = ids_train_valid[int(len(ids_train_valid)*17/20//1):]
        ids_dict = {'train': ids_train, 'valid': ids_valid, 'test': ids_test}

        for phase in phases:
            dataset = InnerSpeechDataset(data[ids_dict[phase]], labels[ids_dict[phase]])
            dataset_dict[phase].append(dataset)

            if phase == 'train':
                batch_size = 8
            else:
                batch_size = 1
            dataloader_dict[phase].append(DataLoader(dataset, batch_size=batch_size))


    # experiment
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 200

    test_acc_sum = 0
    os.makedirs('result', exist_ok=True)
    text_name = 'result/output_sub{}.txt'.format(str(sub_num).zfill(2))
    for fold_i in range(4):
        model = EEGNet()
        model = model.to(device)
        model_path = 'model.pth'

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        train_acc_list = []
        train_loss_list = []
        valid_acc_list = []
        valid_loss_list = []
        max_valid_acc = 0
        for epoch in range(epochs):
            # train
            model.train()
            train_acc = 0
            train_loss_epoch = 0
            for _, (inputs, labels) in enumerate(dataloader_dict['train'][fold_i]):
                optimizer.zero_grad()
                inputs = inputs.type(torch.FloatTensor).to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                train_loss = criterion(outputs, labels)
                train_loss_epoch += train_loss.item()
                for i in range(len(labels)):
                    if torch.argmax(outputs[i]) == torch.argmax(labels[i]):
                        train_acc += 1
                train_loss.backward()
                optimizer.step()

            train_acc_list.append(train_acc/len(dataloader_dict['train'][fold_i]))
            train_loss_list.append(train_loss_epoch/len(dataloader_dict['train'][fold_i]))

            # valid
            model.eval()
            valid_acc = 0
            valid_loss_epoch = 0
            with torch.no_grad():
                for _, (inputs, labels) in enumerate(dataloader_dict['valid'][fold_i]):
                    inputs = inputs.type(torch.FloatTensor).to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    valid_loss = criterion(outputs, labels)
                    valid_loss_epoch += valid_loss
                    for i in range(len(labels)):
                        if torch.argmax(outputs[i]) == torch.argmax(labels[i]):
                            valid_acc += 1
            if valid_acc > max_valid_acc:
                max_valid_acc = valid_acc
                torch.save(model.state_dict(), model_path)

            print('valid_acc:', valid_acc/len(dataloader_dict['valid'][fold_i]))
            valid_acc_list.append(valid_acc/len(dataloader_dict['valid'][fold_i]))
            valid_loss_list.append(valid_loss_epoch/len(dataloader_dict['valid'][fold_i]))


        # test
        model.load_state_dict(torch.load(model_path))
        model.eval()
        test_acc = 0
        with torch.no_grad():
            for _, (inputs, labels) in enumerate(dataloader_dict['test'][fold_i]):
                inputs = inputs.type(torch.FloatTensor).to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                for i in range(labels.shape[0]):
                    if torch.argmax(outputs[i]) == torch.argmax(labels[i]):
                        test_acc += 1
        test_acc_sum += test_acc/len(dataloader_dict['test'][fold_i])
        with open(text_name, 'a') as f:
            f.write('test: ' + str(test_acc/len(dataloader_dict['test'][fold_i])) + '\n')

        # plot training and validation accuracy and loss
        plt.plot(range(epochs), train_acc_list, color = "blue", linestyle = "solid", label = 'train acc')
        plt.plot(range(epochs), valid_acc_list, color = "green", linestyle = "solid", label= 'valid acc')
        plt.title('training and Validation accuracy')
        plt.legend()
        plt.savefig('result/acc_fold{}.png'.format(fold_i))
        plt.close()

        plt.plot(range(epochs), train_loss_list, color = "red", linestyle = "solid" ,label = 'train loss')
        plt.plot(range(epochs), valid_loss_list, color = "orange", linestyle = "solid" , label= 'valid loss')
        plt.title('Training and Validation loss')
        plt.legend()

        plt.savefig('result/loss_fold{}.png'.format(fold_i))
        plt.close()

    # calculate average test accuracy of 4-folds
    with open(text_name, 'a') as f:
        f.write('test_average: ' + str(test_acc_sum/4) + '\n')

except Exception as e:
    with open('error.txt', 'a') as f:
        f.write('error: ' + str(e.args))
