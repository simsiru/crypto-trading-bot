import torch
from forcasting_models import LSTMPredictor, HistoricalPriceDataset
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def get_datetime_interval(start, end, delta):
    curr = start
    res = []
    while curr < end:
        res.append(curr)
        curr += delta
    return res

def forcasting(seq_len=30, future_steps=None, use_predicted=False, coin_to_predict="BTC-USD", price_period="6mo", price_interval="1d"):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    lstm_forcaster = LSTMPredictor(device)
    lstm_forcaster.load_state_dict(torch.load('forcasting_models/9_lstm_model_loss_0.0029_valid_loss_0.0059_test_loss_0.0489.pth',map_location="cuda:0"))
    lstm_forcaster.eval().to(device)

    dataframe = yf.download(coin_to_predict, period=price_period, interval=price_interval)
    #dataframe = dataframe[['Open','High','Low','Close','Volume']]
    dataframe = dataframe[['Open']]
    
    pred_len_hist_data = len(dataframe) - seq_len

    standard_scaler = StandardScaler()
    standard_scaler = standard_scaler.fit(dataframe)
    datatensor = standard_scaler.transform(dataframe)

    datatensor = torch.Tensor(datatensor).unsqueeze(0).to(device)

    if future_steps is not None:
        use_predicted=True
        for i in range(future_steps):
            pred = lstm_forcaster(datatensor[:,-seq_len:])

            if future_steps == 1:
                #datatensor = torch.concat((datatensor, torch.Tensor([[pred[0][0],0,0,0,0]]).unsqueeze(0).to(device)), dim=1)
                datatensor = torch.concat((datatensor, torch.Tensor([[pred[0][0]]]).unsqueeze(0).to(device)), dim=1)
            else:
                datatensor = torch.concat((datatensor, pred.unsqueeze(0).to(device)), dim=1)
    else:
        datatensor_ = datatensor.clone()
        for i in range(pred_len_hist_data):
            pred = lstm_forcaster(datatensor[:,i:i+seq_len])

            if (i < pred_len_hist_data) and use_predicted:
                datatensor[0][i+seq_len][0] = pred[0][0]
                #datatensor[0][i+seq_len] = pred[0]

            if not use_predicted:
                datatensor_[0][i+seq_len][0] = pred[0][0]
                #datatensor_[0][i+seq_len] = pred[0]

    if use_predicted:
        datatensor = datatensor.detach().cpu().numpy()
        datatensor = standard_scaler.inverse_transform(datatensor[0])

        if future_steps is not None:
            curr_date = datetime.now()
            date_interval = get_datetime_interval(curr_date - timedelta(days=len(dataframe)), curr_date + timedelta(days=future_steps), timedelta(days=1))
            plt.plot(date_interval, datatensor[:,0], color='black', label='Predicted')
        else:
            plt.plot(dataframe.index, dataframe['Open'], color='blue', label='Real', linewidth=3)
            plt.plot(dataframe.index, datatensor[:,0], color='black', label='Predicted')
    else:
        datatensor_ = datatensor_.detach().cpu().numpy()
        datatensor_ = standard_scaler.inverse_transform(datatensor_[0])

        plt.plot(dataframe.index, dataframe['Open'], color='blue', label='Real', linewidth=3)
        plt.plot(dataframe.index, datatensor_[:,0], color='black', label='Predicted')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("BTC-USD price prediction")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # use "period" instead of start/end
    # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    # (optional, default is '1mo')

    # fetch data by interval (including intraday if period < 60 days)
    # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    # (optional, default is '1d')

    forcasting(seq_len=100, future_steps=10, use_predicted=False, coin_to_predict="BTC-USD", price_period="3mo", price_interval="1d")