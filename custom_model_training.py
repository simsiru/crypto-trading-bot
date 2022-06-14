import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from forcasting_models import LSTMPredictor, HistoricalPriceDataset

def training_main(epochs = 10, batch_size = 128, seq_len = 60):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    btc_df = pd.read_csv("bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv") # 4_857_377
    train_split_idx = int(len(btc_df) * 0.95)  # 4_371_639
    valid_split_idx = int(len(btc_df) * 0.975) # 4_614_508

    train_data = btc_df[0:train_split_idx:60]
    train_ds_len = len(train_data)
    valid_data = btc_df[train_split_idx:valid_split_idx:60]
    test_data = btc_df[valid_split_idx::60]

    train_dataset = HistoricalPriceDataset(train_data, seq_len=seq_len)
    valid_dataset = HistoricalPriceDataset(valid_data, seq_len=seq_len)
    test_dataset = HistoricalPriceDataset(test_data, seq_len=seq_len)

    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    #dataiter = iter(train_data_loader)
    #print(dataiter.next()[1].size())

    model = LSTMPredictor(device).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters())

    print('\nTraining in progress...')
    for i in range(epochs):
        print(f'\n-------------------------------\nEpoch {i+1}\n-------------------------------\n')
        for curr_batch, (train_data, train_target) in enumerate(train_data_loader):
            train_data = train_data.to(device)
            train_target = train_target.to(device)
            out = model(train_data)
            loss = criterion(out, train_target)
            train_loss = loss.item()
            print(f'\tLoss: {train_loss:.6f}  [{curr_batch*len(train_data):>5d} / {train_ds_len:>5d}]')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('\nEvaluating on validation dataset')
        with torch.no_grad():
            valid_loss = 0
            for valid_data, valid_target in valid_data_loader:
                valid_data = valid_data.to(device)
                valid_target = valid_target.to(device)
                pred = model(valid_data)
                valid_loss += criterion(pred, valid_target).item()
                #y = pred.detach().numpy()
            valid_loss /= len(valid_data_loader)
            print(f'\n\tAvg_valid_loss: {valid_loss:.6f}')
            torch.save(model.state_dict(), f'forcasting_models/9_lstm_model_loss_{train_loss:.4f}_valid_loss_{valid_loss:.4f}.pth')

    print('\n\t\tTraining Done !')
    print('\nEvaluating on test dataset')
    with torch.no_grad():
        test_loss = 0
        for test_data, test_target in test_data_loader:
            test_data = test_data.to(device)
            test_target = test_target.to(device)
            pred = model(test_data)
            test_loss += criterion(pred, test_target).item()
        test_loss /= len(test_data_loader)
        print(f'\n\tAvg_test_loss: \n{test_loss:.6f}')

    torch.save(model.state_dict(), f'forcasting_models/9_lstm_model_loss_{train_loss:.4f}_valid_loss_{valid_loss:.4f}_test_loss_{test_loss:.4f}.pth')

# h=128 l=6 f=5 c=1  sl=48
# h=128 l=6 f=1 c=1  sl=60
# h=128 l=3 f=1 c=1  sl=60
# h=256 l=3 f=1 c=1  sl=100
# h=64 l=3 f=1 c=1  sl=100
# h=128 l=2 f=1 c=1  sl=100
# h=128 l=4 f=1 c=1  sl=100
# h=128 l=3 f=1 c=1  sl=100

if __name__ == "__main__":
    training_main(epochs = 6, batch_size = 128, seq_len = 100)