import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from sklearn.cluster import KMeans
from datetime import datetime,timedelta

#get ticker data
def get_data(ticker):
    ticker = ticker.upper()
    end_date = datetime.today()
    start_date = end_date-timedelta(days=2000) #arbitrary number of days
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    df = web.DataReader(ticker,'yahoo',start_date,end_date) #type=pandas DataFrame
    return df

#load data
aapl = get_data('aapl') #Loads AAPL data

#define plot
def plot_trends(x,y,lb):
    clist = ['r','orange','g']
    nclasses = len(np.unique(lb))
    for i in range(nclasses):
        xx = x[lb==i]
        yy = y[lb==i]
        plt.scatter(xx,yy,s=5,marker='.',c=clist[i],label=str(i))
    plt.legend(fontsize='x-large')
    plt.show()


#get trend based on ema 
def get_trends(c,lookback):
    '''
    :type c: ndarray
    :param c: Daily Close prices

    :type lookback: int
    :param lookback: Lookback window for smoothing
    '''
    cs = pd.Series(c)
    ema = cs.ewm(span=lookback).mean()
    ema = ema[::-1].ewm(span=lookback).mean()[::-1]
    ema = ema.values

    lr = np.diff(np.log(ema))        
    km = KMeans(3).fit(lr.reshape(-1,1))

    lb = km.labels_

    #Change labels to have some semblance of order
    cc = km.cluster_centers_.flatten()
    temp = [(cc[i],i) for i in range(3)]
    temp = sorted(temp,key=lambda x: x[0])

    labels = np.zeros(len(lb),dtype=int)
    for i in range(1,3):
        old_lb = temp[i][1]
        idx = np.where(lb==old_lb)[0]
        labels[idx] = i

    x = np.arange(len(labels))
    y = ema[1:]

    return x,y,labels

#initialize stock data
c = aapl.Close.values

#retreive trend info
x,y,lb = get_trends(c,20)
yy = c[1:]

#define each trend type
up_trend = y[lb==2]
downtrend = y[lb==0]
notrend = y[lb==1]
print(y[-1])

#When was the last point?
print("UpTrend: {}".format(up_trend[-1]))
print("DownTrend: {}".format(downtrend[-1]))
print("NoTrend: {}".format(notrend[-1]))
plot_trends(x,y,lb)

#...
#plot_trends(x,yy,lb)
plt.savefig('foo.png')
