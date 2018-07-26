
# coding: utf-8

# In[ ]:


import seaborn as sns


# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


tips=sns.load_dataset('tips')


# In[14]:


tips.head()


# In[13]:


sns.distplot(tips['total_bill'])


# In[18]:


sns.pairplot(tips,hue='smoker')


# In[28]:


sns.violinplot(x=tips['day'],y=tips['total_bill'])
sns.stripplot(x=tips['day'],y=tips['total_bill'])

