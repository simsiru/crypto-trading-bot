import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

""" class LSTMPredictor(nn.Module):
    def __init__(self, n_hidden=51):
        super(LSTMPredictor, self).__init__()
        self.n_hidden = n_hidden
        self.lstm1 = nn.LSTMCell(1, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs """

class LSTMPredictor(nn.Module):
    def __init__(self, device, n_features=1, n_hidden=128, n_layers=3, n_classes=1):
        super(LSTMPredictor, self).__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.3
        )

        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.lin1 = nn.Linear(n_hidden, n_classes)

        self.device = device

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden).to(self.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden).to(self.device)

        _, (hn, _) = self.lstm(x, (h0, c0))

        hn = hn[-1]
        #hn = hn.view(-1, self.n_hidden)

        out = self.lin1(F.relu(hn))

        return out


#from torchinfo import summary
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = LSTMPredictor(device)#.to(device)
#model.forward(torch.zeros(32, 60, 2))

#batch_size = 64
#summary(model, input_size=(batch_size, 60, 2), input_data=torch.zeros(batch_size, 60, 2))


class HistoricalPriceDataset(Dataset):
    def __init__(self, btc_dataframe, seq_len=10) -> None:
        super(HistoricalPriceDataset, self).__init__()

        self.dataframe = btc_dataframe
        self.df_len = len(self.dataframe)
        self.seq_len = seq_len

        self.dataframe = self.dataframe[['Open','High','Low','Close','Volume_(Currency)']]
        self.dataframe.rename(columns = {'Open':'open','High':'high','Low':'low','Close':'close',
        'Volume_(Currency)':'vol'}, inplace = True)

        self.standard_scaler = StandardScaler()
        self.standard_scaler = self.standard_scaler.fit(self.dataframe)
        self.dataframe = self.standard_scaler.transform(self.dataframe)

    """ def min_max_vectorized_scaler(self, tensor):
        scale = 1.0 / (tensor.max() - tensor.min()) 
        tensor.mul_(scale).sub_(tensor.min())
        return tensor """

    def __len__(self):
        return self.df_len

    def __getitem__(self, idx):
        if idx+self.seq_len+1 > self.df_len:
            idx = self.df_len - self.seq_len - 1

        def get_seq_idx(idx):
            data = np.nan_to_num(self.dataframe[idx:idx+self.seq_len+1], nan=-1)
            if -1 in data:
                idx = int(np.random.rand() * (self.df_len - self.seq_len - 1))
                return get_seq_idx(idx)
            else:
                return data, idx

        data, idx = get_seq_idx(idx)

        label = torch.Tensor([data[-1][0]])
        #label = torch.Tensor(data[-1])

        #data = torch.Tensor(data[0:-1])
        data = torch.Tensor(data[0:-1,0]).unsqueeze(1)

        return data, label

btc_df = pd.read_csv("bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv")
print(btc_df[:20])
#aa = HistoricalPriceDataset(btc_df)
#ss, dd = aa.__getitem__(0)
#print(ss)#.size())
#print(dd)#.size())