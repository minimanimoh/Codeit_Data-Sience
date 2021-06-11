#!/usr/bin/env python
# coding: utf-8

# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
df = pd.read_csv("downloads/broadcast.csv", index_col=0)
df


# # if omits index_col = 0 ..?

# In[9]:


old_df


# In[12]:


df.plot()


# In[14]:


#default is line graph
df.plot(kind='line') 


# In[15]:


#if wants to choose one of parameters..
df.plot(y='KBS')


# In[16]:


#if choose more than one, make a list
df.plot(y=['KBS', 'JTBC'])


# In[20]:


#another way
#choose two columns from dataframe and make a plot graph from pandas
df[['KBS', 'JTBC']].plot()


# In[ ]:




