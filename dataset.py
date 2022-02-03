from torch.utils.data import Dataset
import mne
import numpy as np


def load_inner_speech_dataset(data_paths):
    data = []
    labels = []

    label_names = ['Arriba', 'Abajo', 'Derecha', 'Izquierda']
    for data_path in data_paths:
        epochs = mne.read_epochs(data_path)
        for label_i in range(len(label_names)):
            label_name = label_names[label_i]

            for epoch in epochs[label_name]:
                data.append(epoch)
                label = np.array([0, 0, 0, 0]).astype(float)
                label[label_i] = 1
                labels.append(label)

    return np.array(data), np.array(labels)


class InnerSpeechDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
