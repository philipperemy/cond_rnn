#!/usr/bin/env python
# coding: utf-8

# In[1]:


import temp


# We read in a dataset with daily recorded temperatures for major cities around the world.

# In[2]:


df = temp.read_data()
print(df.shape)
df.head()


# Input-output pairs are created for supervised machine learning. The output is y and the input are the previous 100 temperature values
# denoted as 'x-i' for i in [1,100]. The location is one-hot encoded so it can also serve as an input.

# In[3]:


df_temp = temp.generate_set(df)
df_temp.head()


# In[15]:


from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df_temp, test_size=0.2)


# In[17]:


temp.fitmodel(df_train, conditional=False)


# # Convert notebook to python

# In[3]:


get_ipython().system('jupyter nbconvert --to script example.ipynb')


# # Points of improvement
# I see the following points for improvement;
#  * the time series signature could be used to create the input pair
#  * a comparison with respect to a classical method such as ARMAX should be added.
#  * The model can be made more complex using features like, distance between cities or temperatures of other cities at that time.