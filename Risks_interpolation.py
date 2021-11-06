#!/usr/bin/env python
# coding: utf-8

# In[92]:


import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
file=pd.read_excel('Interpolationexp2.xlsx',index_col=0,parse_dates=True)
df=pd.DataFrame(data=file)
file1=df.interpolate(method='time',limit_direction='both',downcast='infer')
print(file1)


# In[ ]:




