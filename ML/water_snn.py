import snntorch as snn
from snntorch import surrogate
import torch
from torch.optim.lr_scheduler import StepLR
import itertools
import warnings
import joblib
from torch.optim import AdamW, RAdam, Adagrad
from connector import connector_snn
warnings.filterwarnings('ignore')
# import os
import pandas as pd

def predict_water_level_snn(post_number, start_data, end_data):
        post_number= str(post_number)
        n_features = 6
        epochs = 10
        batch_size = 1
        hidden = 128
        num_samples = 1
        device = torch.device("cpu")

        dataset, clear_dataset, targets2, start_date, end_date, df_filtered = connector_snn(post_number)

        df_filtered = dataset[(dataset['Дата - время'] >= start_data) & (dataset['Дата - время'] <= end_data)]
        targets1 = []
        targets1.append(list(df_filtered['Уровень воды']))

        num_days_out = len(targets1[0])
        num_days_in = num_days_out * 2

        def make_inputs_and_targets(clear_dataset):
            inputs, targets = [], []

            for i in range(len(clear_dataset) - num_days_in - num_days_out):
                inputs.append(list(itertools.chain(*clear_dataset.iloc[i:i + num_days_in].values.tolist())))
                targets.append(list(clear_dataset['Уровень воды'].iloc[i:i + num_days_out]))
            # print(inputs)
            # print(targets)
            return inputs, targets

        class RegressionDataset(torch.utils.data.Dataset):
            def __init__(self, timesteps, num_samples, inputs, targets):
                self.num_samples = num_samples
                self.features = torch.stack([torch.tensor(inputs)], dim=1)
                # print(self.features)
                self.targets = torch.stack([torch.tensor(targets)], dim=1)

            def __len__(self):
                """Number of samples."""
                return self.num_samples

            def __getitem__(self, idx):
                """General implementation, but we only have one sample."""
                return self.features[:, idx, :], self.targets[:, idx, :]

        class Net(torch.nn.Module):
            """Simple spiking neural network in snntorch."""

            def __init__(self, timesteps, hidden):
                super().__init__()

                self.timesteps = timesteps  # number of time steps to simulate the network
                self.hidden = hidden  # number of hidden neurons
                spike_grad = surrogate.fast_sigmoid()  # surrogate gradient function

                # randomly initialize decay rate and threshold for layer 1
                beta_in = torch.rand(self.hidden)
                thr_in = torch.rand(self.hidden)

                # layer 1
                self.fc_in = torch.nn.Linear(in_features=n_features * num_days_in, out_features=self.hidden)
                self.lif_in = snn.Leaky(beta=beta_in, threshold=thr_in, learn_beta=True, spike_grad=spike_grad)

                # randomly initialize decay rate and threshold for layer 2
                beta_hidden = torch.rand(self.hidden)
                thr_hidden = torch.rand(self.hidden)

                # layer 2
                self.fc_hidden = torch.nn.Linear(in_features=self.hidden, out_features=self.hidden)
                self.lif_hidden = snn.Leaky(beta=beta_hidden, threshold=thr_hidden, learn_beta=True,
                                            spike_grad=spike_grad)

                # randomly initialize decay rate for output neuron
                beta_out = torch.rand(1)

                # layer 3: leaky integrator neuron. Note the reset mechanism is disabled and we will disregard output spikes.
                self.fc_out = torch.nn.Linear(in_features=self.hidden, out_features=num_days_out)
                self.li_out = snn.Leaky(beta=beta_out, threshold=1.0, learn_beta=True, spike_grad=spike_grad,
                                        reset_mechanism="none")

            def forward(self, x):
                """Forward pass for several time steps."""

                # Initalize membrane potential
                mem_1 = self.lif_in.init_leaky()
                mem_2 = self.lif_hidden.init_leaky()
                mem_3 = self.li_out.init_leaky()

                # Empty lists to record outputs
                mem_3_rec = []

                # Loop over
                for step in range(self.timesteps):
                    x_timestep = x[step, :, :]

                    cur_in = self.fc_in(x_timestep)
                    spk_in, mem_1 = self.lif_in(cur_in, mem_1)

                    cur_hidden = self.fc_hidden(spk_in)
                    spk_hidden, mem_2 = self.li_out(cur_hidden, mem_2)

                    cur_out = self.fc_out(spk_hidden)
                    _, mem_3 = self.li_out(cur_out, mem_3)

                    mem_3_rec.append(mem_3)

                return torch.stack(mem_3_rec)

        def train(model, dataloader, optimizer, scheduler, loss_function, index):
            loss_hist = []  # record loss

            for _ in range(epochs):
                train_batch = iter(dataloader)
                minibatch_counter = 0
                loss_epoch = []

            for feature, label in train_batch:
                # prepare data
                feature = torch.swapaxes(input=feature, axis0=0, axis1=1)
                label = torch.swapaxes(input=label, axis0=0, axis1=1)
                feature = feature.to(device)
                label = label.to(device)

                # forward pass
                mem = model(feature)
                loss_val = loss_function(mem, label)  # calculate loss
                optimizer.zero_grad()  # zero out gradients
                loss_val.backward()  # calculate gradients
                optimizer.step()  # update weights
                scheduler.step()

                # store loss
                loss_hist.append(loss_val.item())
                loss_epoch.append(loss_val.item())
                minibatch_counter += 1

                avg_batch_loss = sum(loss_epoch) / minibatch_counter  # calculate average loss p/epoch
            return mem[index]

        inputs, targets = make_inputs_and_targets(clear_dataset)
        index = dataset[dataset['Дата - время'] == start_data].index.item() - num_days_in

        num_steps = len(inputs)
        regression_dataset = RegressionDataset(timesteps=len(inputs), num_samples=num_samples, inputs=inputs, targets=targets)
        dataloader = torch.utils.data.DataLoader(dataset=regression_dataset, batch_size=batch_size, drop_last=True)

        model = Net(timesteps=num_steps, hidden=hidden).to(device)
        optimizer = Adagrad(params=model.parameters(), lr=0.997, weight_decay=0.01)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        loss_function = torch.nn.MSELoss()
        trained = train(model, dataloader, optimizer, scheduler, loss_function, index)
        predicted = trained[0]

        if len(predicted) > len(targets1[0]):
            predicted = predicted[:len(targets1[0])]


        h = lambda x: x * (max(targets1[0]) - min(targets1[0])) + min(targets1[0])
        avg_er = 0
        fact = []
        real = []

        for x, i in enumerate(predicted):
            fact.append(h(i.item()))
            real.append(targets1[0][x])
            print(f'Предсказанные: {h(i.item())}, Реальные: {targets1[0][x]}, Ошибка: {targets1[0][x] - h(i.item())}')
            avg_er += abs(targets1[0][x] - h(i.item()))
        print(f"Средняя ошибка: {avg_er / len(predicted)}")
        df_filtered['Дата - время'] = pd.to_datetime(df_filtered['Дата - время']).apply(lambda x: x.replace(hour=8, minute=0))

        result1 = pd.DataFrame({
            'Код поста': df_filtered['Код поста'],
            'Дата - время': df_filtered['Дата - время'],
            'Реальные уровни воды': real,
            'Прогнозируемые уровни воды': fact
        })

        result = pd.DataFrame(result1)
        return result



