#!/usr/bin/env python
# coding: utf-8

# In[24]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


Apple = yf.download("AAPL", start = "2010-01-01",end ="2023-01-01")


# In[3]:


Apple


# In[4]:


ticker = ["SPY","AAPL","KO"]


# In[5]:


stocks = yf.download(ticker, start = "2010-01-01",end ="2023-01-01")


# In[6]:


stocks


# In[7]:


stocks.head()


# In[8]:


stocks.tail()


# In[9]:


stocks.to_csv("stocksyt.csv")


# In[10]:


stocks = pd.read_csv("stocksyt.csv")


# In[11]:


stocks


# In[14]:


stocks = pd.read_csv("stocksyt.csv",header=[0,1])
stocks


# In[15]:


stocks = pd.read_csv("stocksyt.csv",header=[0,1], index_col=[0])
stocks


# In[16]:


stocks = pd.read_csv("stocksyt.csv",header=[0,1], index_col=[0], parse_dates=[0])
stocks


# In[17]:


stocks.columns


# In[18]:


stocks.describe()


# In[20]:


close=stocks.loc[:,"Close"].copy()


# In[21]:


close


# In[25]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use("seaborn")


# In[26]:


close.plot(figsize=(15,8),fontsize=12)


# In[27]:


close.plot(figsize=(15,8),fontsize=12)
plt.legend(fontsize=12)
plt.show()


# In[28]:


close.iloc[0,0]


# In[29]:


close.AAPL


# In[30]:


close.AAPL.div(close.iloc[0,0])


# In[32]:


close.AAPL.div(close.iloc[0,0]).mul(100)


# In[33]:


close.iloc[0]


# In[34]:


normclose = close.div(close.iloc[0]).mul(100)


# In[35]:


normclose.plot(figsize=(15,8),fontsize=12)
plt.legend(fontsize=12)
plt.show()


# In[36]:


close.iloc[0,0]


# In[37]:


aapl= close.AAPL.copy().to_frame()


# In[38]:


aapl


# In[40]:


aapl.shift(periods=1)


# In[41]:


aapl["lag1"]=aapl.shift(periods=1)


# In[42]:


aapl


# In[44]:


aapl.AAPL.sub(aapl.lag1)


# In[45]:


aapl


# In[46]:


aapl["Diff"]=aapl.AAPL.sub(aapl.lag1)


# In[47]:


aapl


# In[48]:


aapl.AAPL.div(aapl.lag1)


# In[49]:


aapl["% change"]=aapl.AAPL.div(aapl.lag1)


# In[50]:


aapl


# In[51]:


del aapl["lag1"]


# In[52]:


del aapl["Diff"]
aapl


# In[53]:


aapl.rename(columns = {'% change': 'Change'}, inplace =True)


# In[54]:


aapl


# In[56]:


aapl.AAPL.resample("M").last()


# In[57]:


aapl.AAPL.resample("M").last()


# In[58]:


aapl.AAPL.resample("M").last()


# In[59]:


aapl.AAPL.resample("M")


# In[60]:


aapl.AAPL.resample("BM").last()


# In[61]:


aapl.rename(columns = {'% change': 'Change'}, inplace =True)
aapl


# In[62]:


aapl.AAPL.resample("M").last()


# In[63]:


weekly_summary['weekly']=df.story_point.resample('W').transform('sum')


# In[65]:


stocks = pd.read_csv('stocksyt.csv', parse_dates=['Date'])


# In[67]:


del aapl["Change"]


# In[68]:


aapl


# In[71]:


ret = aapl.pct_change().dropna()


# In[72]:


ret


# In[73]:


ret.info()


# In[75]:


ret.plot(kind="hist",figsize=(12,8), bins=100)
plt.show()


# In[77]:


daily_mean_ret=ret.mean()
daily_mean_ret


# In[78]:


var_daily=ret.var()
var_daily


# In[79]:


std_daily=np.sqrt(var_daily)


# In[80]:


std_daily


# In[81]:


ret.std()


# In[87]:


annual_mean_ret=daily_mean_ret*252
annual_mean_ret


# In[88]:


annual_var_ret=var_daily*252
annual_var_ret


# In[90]:


annual_std_returns=np.sqrt(annual_var_ret)
annual_std_returns


# In[92]:


ret.std()*np.sqrt(252)


# In[93]:


ticker = ["SPY","AAPL","KO","IBM","DIS","MSFT"]


# In[94]:


stocks = yf.download(ticker, start = "2010-01-01",end ="2023-01-01")


# In[95]:


close=stocks.loc[:,"Close"].copy()


# In[96]:


normclose=close.div(close.iloc[0]).mul(100)


# In[97]:


normclose.plot(figsize=(15,8),fontsize=12)
plt.legend(fontsize=12)
plt.show()


# In[98]:


ret=close.pct_change().dropna()
ret.head()


# In[99]:


ret.describe().T


# In[101]:


summary=ret.describe().T.loc[:,["mean","std"]]
summary


# In[103]:


summary["mean"]=summary["mean"]*252
summary["std"]=summary["std"]*np.sqrt(252)
summary


# In[108]:


summary.plot.scatter(x="std",y="mean",figsize=(12,8),s=50,fontsize=15)
for i in summary.index:
    plt.annotate(i,xy=(summary.loc[i,"std"]+0.002,summary.loc[i,"mean"]+0.002),size=15)
    plt.xlabel("Annual risk(std)",fontsize=15)
    plt.ylabel("Annual return",fontsize=15)
    plt.title("Risk/return",fontsize=15)


# In[109]:


ret.cov()


# In[110]:


ret.corr()


# In[111]:


import seaborn as sns


# In[114]:


plt.figure(figsize=(12,8))
sns.set(font_scale=1.4)
sns.heatmap(ret.corr(),cmap="Reds",annot=True,annot_kws={"size":15},vmax=0.6)
plt.show()


# In[115]:


df = pd.DataFrame(index=[2016,2017,2018],data=[100,50,95],columns=["Price"])


# In[116]:


df


# In[117]:


simplereturns=df.pct_change().dropna()


# In[118]:


simplereturns


# In[119]:


simplereturns.mean()


# In[122]:


logreturns=np.log(df/df.shift(1)).dropna()
logreturns


# In[123]:


logreturns.mean()


# In[124]:


100*np.exp(logreturns.mean()*2)


# In[125]:


SPY = yf.download("SPY")


# In[ ]:




