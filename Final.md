```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jason
```


```python
from sklearn.cluster import DBSCAN
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```

# Preprocessing


```python
ks_lst_df = pd.read_csv('kospi_list.csv',encoding='CP949')
ks_lst_df = ks_lst_df[~((ks_lst_df['구분'] == 10) | (ks_lst_df['구분'] == 12) | (ks_lst_df['구분'] == 17))] 
```


```python
kdq_lst_df = pd.read_csv('kosdaq_list.csv',encoding='CP949')
ks_dict = pd.Series(ks_lst_df.종목명.values,index=ks_lst_df.종목코드).to_dict() 
kdq_dict = pd.Series(kdq_lst_df.종목명.values,index=kdq_lst_df.종목코드).to_dict() 
```


```python
ks_dict.update(kdq_dict)
total_dict = ks_dict
```


```python
from glob import glob
stk_lst = glob(r'/Users/eunbihan/Desktop/HY Univ/KORMS/alicia')

file_name_lst = []
for name in stk_lst:
    file_name = name.split('\\')[-1].split('.')[0]
    if file_name in list(total_dict.keys()):
        file_name_lst.append(file_name)
```


```python
base_path2 = "/Users/eunbihan/Desktop/HY Univ/KORMS/pair_trading/alicia"
```


```python
close_df = pd.DataFrame()#pd.DataFrame(columns = list(ks_dict.values()))

for file_name in file_name_lst:
    temp_df = pd.read_csv(base_path2 + str(file_name) +'.csv',encoding='CP949').set_index('날짜').rename(columns={'종가':file_name})[file_name]#.iloc[:,3:4]
    close_df = pd.concat([close_df,temp_df],axis = 1)
close_df.index = close_df.index.map(lambda x :str(x).split('.')[0])
close_df.index = pd.to_datetime(close_df.index)
close_df.to_csv('종가.csv')
```


```python
daily_ret = np.log(close_df/close_df.shift(1))
```


```python
m_ret = daily_ret.resample('1m').apply(lambda x: x.sum() if len(x[x.notna()])>=15 else np.nan)
bw_ret = daily_ret.resample('2w').apply(lambda x: x.sum() if len(x[x.notna()])>=5 else np.nan)
w_ret = daily_ret.resample('1w').apply(lambda x: x.sum() if len(x[x.notna()])>=2 else np.nan)
```


```python
m_ret.to_csv('m_ret.csv')
bw_ret.to_csv('bw_ret.csv')
w_ret.to_csv('w_ret.csv')
```

# Pair_trading_calculator


```python
close_df = pd.read_csv('종가.csv',index_col=0)
close_df = close_df[close_df.index>'2000-01-01']
close_df.index = pd.DatetimeIndex(close_df.index)
daily_ret = np.log(close_df/close_df.shift(1))

```


```python
m_ret = pd.read_csv('m_ret.csv',index_col= 0 )
m_ret.index = pd.DatetimeIndex(m_ret.index)
bw_ret = pd.read_csv('bw_ret.csv',index_col= 0 )
bw_ret.index = pd.DatetimeIndex(bw_ret.index)
w_ret = pd.read_csv('w_ret.csv',index_col= 0 )
w_ret.index = pd.DatetimeIndex(w_ret.index)
```

## Data Pre-processing

- according to rebalancing interval 
- input : weekly/bi-weekly/monthly log-return data 


```python
def momentum(ret,max_momentum):
    if ret.shape[0]<max_momentum:
        return "Length of data is shorter than max_momentum"
    else:
        ret = ret.apply(lambda x:x if len(x[x.notna()])>=max_momentum else np.nan)
        ret = ret.iloc[-max_momentum:,:].dropna(axis=1)
        
        col_name = list(pd.Series(['mom']*max_momentum)+pd.Series(list(range(1,max_momentum+1))).astype(str))
        etf_list = list(ret.columns)
        momentem_df = pd.DataFrame(index = etf_list,columns = col_name)
        for etf in etf_list:
            momentum_df.loc[etf,'mom1']=ret[etf].iloc[-1]
            for j in range(2,max_momentum+1):
                mom='mom'+str(j)
                momentum_df.loc[etf,mom]=ret[etf].iloc[-j:-1].sum()
        return momentum_df
```


```python
# input : momentum data from momentum function

def PCA_df(df):
    normalized_df = StandardScaler().fit(df)
    normalized_df = normalized_df.transform(df)
    
    pca = PCA()
    PCA_df = pca.fit_transform(noramlized_df)
    explained_var = pca.explained_variance_ratio_.cumsum()
    num_PC = len(explained_var[explained_var<0.99])+1
    
    final_df = PCA_df[:,:num_PC]
    final_df = pd.DataFrame(final_df,index=df.index)
    
    return final_df
```

## Clustering

### 1. K-Means


```python
def kmeans_w_pct_ol(dataframe,K,percentile):
    km = KMeeans(n_clusters=K,max_iter=1000)
    km = km.fit(dataframe)
    
    result = pd.DataFrame(index=dataframe.index)
    result['label']=km.labels_
    
    nearest_dis = pd.DataFrame(euclidean_distances(dataframe)).apply(lambda x:x[x>0].min())
    eps = np.percentile(nearest_dis.sort_values(),percentile)
    centroids = km.cluster_centers_
    
    for i in range(K):
        results.loc[results['label']==i,'central_dis']=cdist(dataframe.iloc[km.labels_==i],
                                                            centroids[i].reshape(1,centroids.shape[1]),'euclidean')
    result['OL']=((results['central_dis']-eps)>=0).astype(float)
    
    return result[['label','OL']]
```

### 2. Agglomerative Clustering


```python
def AG_cluster(dataframe,percentile):
    dis = pd.DataFrame(manhattan_distances(dataframe)).apply(lambda x:x[x>0].min()).sort_values()
    eps = np.percentile(dis,percentile)
    ag_cluster = AgglomerativeClustering(n_clusters=None,affinity = 'l1',
                                         linkage='average',distance_threshold=eps).fit(dataframe)
    result = pd.DataFrame(ag_cluster.labels_,index=dataframe.index,columns=['label'])
    
    return result
```

### 3. DBscan


```python
def dbscan(dataframe,percentile):
    minpts = round(np.log(dataframe.shape[0]))
    dis = pd.DataFrame(manhattan_distances(dataframe)).apply(lambda x:x[x>0].sort_values()[:minpts].mean()).sort_values()
    eps = np.percentile(dis,percentile)
    db_cluster = DBSCAN(eps = eps, min_samples=minpts,
                       metric = 'l1').fit(dataframe)
    result = pd.DataFrame(db_cluster.labels_,index=dataframe.index,
                         columns=['label'])
    result.loc[result['label']<0,'OL']=1
    result.loc[results['label']>=0,'OL']=0
    
    return result
```

#### Stocks in Portfolio


```python
#stocks in portfolio

def portfolio_generation(momentum_df, cluster_df):


    if cluster_df.shape[1]==2:
        non_outlier = cluster_df[cluster_df['OL']==0]
    else:
        non_outlier = cluster_df
    
    cluster_list = list(non_outlier['label'].unique())    
    K = len(cluster_list)
    LONG = []
    SHORT = []
    diff_ = []
    for i in range(K):
        cur_cluster = cluster_list[i]
        
        temp_df_ = momentum_df.loc[list(non_outlier[non_outlier['label']==cur_cluster].index),'mom1']
        if len(temp_df_)==1:
            continue
        temp_df_ = temp_df_.sort_values()
        

        temp_long_ = temp_df_.iloc[:int(temp_df_.shape[0]/2)]
        temp_short_ = temp_df_.iloc[-int(temp_df_.shape[0]/2):]

        for j in range(len(temp_long_)):
            diff_ = diff_ + [temp_short_.iloc[-(j+1)]-temp_long_.iloc[j]]
    
    if len(diff_)==0:
        return LONG,SHORT

    diff_cut = pd.Series(diff_).std()

    for i in range(K):
        cur_cluster = cluster_list[i]
        
        temp_df = momentum_df.loc[list(non_outlier[non_outlier['label']==cur_cluster].index),'mom1']
        if len(temp_df)==1:
            continue
        temp_df = temp_df.sort_values()
        

        temp_long = temp_df.iloc[:int(temp_df.shape[0]/2)]
        temp_short = temp_df.iloc[-int(temp_df.shape[0]/2):]

        for j in range(len(temp_long)):
            if (temp_short.iloc[-(j+1)]-temp_long.iloc[j])>diff_cut:
                LONG = LONG + [temp_long.index[j]]
                SHORT = SHORT + [temp_short.index[-(j+1)]]

    return LONG,SHORT
```

### Calcualte Performance


```python
#extract performance
def performance(daily_ret,ret,max_momentum,K,percentile, cluster):

    performance_df = pd.DataFrame()
    portfolio_weight = pd.DataFrame(columns = ret.columns)

    for i in range(max_momentum,ret.shape[0]-1):
        reb_date = ret.index[i]
        next_reb_date = ret.index[i+1]
        temp_ret = ret[ret.index<=reb_date]
        momentum_df = momentum(temp_ret, max_momentum)
        final_df = PCA_df(momentum_df)
        if cluster =='km':
            cluster_df = kmeans_w_pct_ol(final_df, K, percentile)
        elif cluster =='db':
            cluster_df = dbscan(final_df, percentile)
        elif cluster =='kmd':
            cluster_df = Kmedoid_o(final_df , K ,percentile)
        else:
            cluster_df = AG_cluster(final_df, percentile)
        long_short = portfolio_generation(momentum_df,cluster_df)
        long = long_short[0]
        short = long_short[1]
        temp_return = (daily_ret.loc[(daily_ret.index<=next_reb_date)&(daily_ret.index>reb_date),long].sum(axis=1)\
                        -daily_ret.loc[(daily_ret.index<=next_reb_date)&(daily_ret.index>reb_date),short].sum(axis=1))/len(long)

        temp_return = temp_return.fillna(0)

        performance_df = pd.concat([performance_df,temp_return])
        if len(long)>0:
            portfolio_weight.loc[reb_date,long] = 1/len(long)
            portfolio_weight.loc[reb_date,short] = -1/len(short)
        else:
            portfolio_weight.loc[reb_date] = 0
  
    return performance_df,portfolio_weight.fillna(0)
```


```python
iter_dir = 'result\\monthly_db\\'

i=0
#Kmeans
#cluster_list = [5,10,20,25]
mom_list = [12,24,48]
percentile_list = [10,20,30,40,50,60,70,80,90]
cl=0
all_per = pd.DataFrame()
for mom in mom_list:
    #for cl in cluster_list:
        for pct in percentile_list:
            temp = performance(daily_ret,m_ret,mom,cl,pct,'db')
            temp_per = pd.DataFrame(temp[0])
            
            temp_per.columns=[str(mom)+'-'+str(pct)]
            temp_dir = iter_dit + str(mom)+'-'+str(pct)+'.csv'
            
            temp[1].to_csv(temp_dir)
            all_per = pd.concat([all_per,temp_per],axis=1)
            all_per.to_csv('monthly_db_per.csv') #weekly_km_per24.csv / weekly ~~
```


```python

```
