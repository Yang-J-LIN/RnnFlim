import os
import numpy as np
import pandas as pd
import torch
import seaborn as sns

from tqdm import tqdm

from torch.utils.data import random_split, DataLoader
import torch.optim as optim

from dataset import SimulationDataset
from quantized_gru import QuantizedGRU, QuantizedGRUCell, QuantizedFCNN
from utils import read_state_dict, loss_func

# check available GPUs
if torch.cuda.is_available() is True:
    for i in range(torch.cuda.device_count()):
        print('Device {}:'.format(i), torch.cuda.get_device_name(i))

device = 'cuda:0'   # use Nvidia RTX A4500 GPU

dataset_path = 'dataset'

simulation_dataset = SimulationDataset(
    data_path = dataset_path + 'data.npy',
    label_path = dataset_path + 'label.npy',
    device = device
)

train_set_size = int(0.8 * len(simulation_dataset))
# train_set_size = 2000
dev_set_size = int(0.1 * len(simulation_dataset))
test_set_size = len(simulation_dataset) - train_set_size - dev_set_size
print('train/dev/test size is {}/{}/{}.'.format(
    train_set_size,
    dev_set_size,
    test_set_size
))
train_set, dev_set, test_set = \
    random_split(simulation_dataset, [train_set_size, dev_set_size, test_set_size], generator=torch.Generator().manual_seed(42))

batch_size = 128

train_loader = DataLoader(train_set, batch_size=batch_size)
dev_loader = DataLoader(dev_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

state_dict_path = 'checkpoints/20221006_gru_12.pt'
rnn_state_dict, fcnn_state_dict = read_state_dict(torch.load(
    state_dict_path)
)

model = QuantizedGRU(hidden_size=12, device=device, weights={
    **rnn_state_dict,
    **fcnn_state_dict
})

# train the model

learning_rate = 0.0002

optimizer = optim.AdamW(lr=learning_rate, params=model.parameters())

scheduler = optim.lr_scheduler.StepLR(
    optimizer=optimizer,
    step_size=5,
    gamma=0.9
)

epoch = 5

weights = torch.sigmoid(torch.tensor((torch.arange(1024) - 256) / 256)).reshape(1, -1).to(device)
criterion = loss_func

for i in range(epoch):
    loss_total = 0
    for j, (data, label) in tqdm(enumerate(train_loader)):

        output = model(data.T.unsqueeze(2))

        loss = criterion(output.squeeze(2).T, label, weights=weights)

        loss_total += loss.item()

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

    scheduler.step()

    torch.save(model.state_dict(), os.path.join('checkpoints', 'QuGRU_12' + '.pt'))

    print('Finished training epoch {}. Average loss is {:.8f}'.format(i + 1, loss_total / len(train_loader)))