#!/usr/bin/env python
# coding: utf-8

# In[1]:


## for data
import pandas as pd
import numpy as np
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
## for statistical tests
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm
## for machine learning
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
## for explainer


# In[2]:


dfs = pd.read_html("https://www.pgatour.com/stats/stat.02567.html")
df_drive = pd.read_html("https://www.pgatour.com/stats/stat.101.html")
df_sg_atg = pd.read_html("https://www.pgatour.com/stats/stat.02568.html")
df_three_putt = pd.read_html("https://www.pgatour.com/stats/stat.426.html")
df_short_acc = pd.read_html("https://www.pgatour.com/stats/stat.374.html")
df_drive_acc = pd.read_html("https://www.pgatour.com/stats/stat.102.html")
df_results = pd.read_html("https://en.wikipedia.org/wiki/Masters_Tournament")
df_sg_putt = pd.read_html("https://www.pgatour.com/stats/stat.02564.html")
df_avg_scr = pd.read_html("https://www.pgatour.com/stats/stat.108.html")


# In[3]:


df_current_players = pd.read_csv('current_players.csv',encoding = "ISO-8859-1")


# In[4]:


df_current_players


# In[5]:


df_sg_off_the_tee = dfs[1]
df_drive = df_drive[1]
df_sg_atg = df_sg_atg[1]
df_three_putt = df_three_putt[1]
df_short_acc = df_short_acc[1]
df_drive_acc = df_drive_acc[1]
df_sg_putt = df_sg_putt[1]
df_results = df_results[4]
df_avg_scr =df_avg_scr[1]


# In[6]:


df_avg_scr


# In[7]:


df_three_putt = df_three_putt[['PLAYER NAME','%']].copy()
df_drive_acc = df_drive_acc[['PLAYER NAME','%']].copy()
df_results = df_results[['Champion','Year']].copy()
df_sg_putt = df_sg_putt[['PLAYER NAME','AVERAGE']].copy()
df_avg_scr = df_avg_scr[['PLAYER NAME','AVG']].copy()
df_three_putt.columns = ['PLAYER NAME', '3 PUTT %']
df_drive_acc.columns = ['PLAYER NAME', 'Fairway %']
df_results.columns = ['PLAYER NAME', 'Year']
df_sg_putt.columns = ['PLAYER NAME', 'Sg_Putt']
df_avg_scr.columns = ['PLAYER NAME', 'AVG_SCR']


# In[8]:


df_sg_off_the_tee = df_sg_off_the_tee[['PLAYER NAME','AVERAGE']].copy()
df_drive = df_drive[['PLAYER NAME','AVG.']].copy()
df_sg_atg = df_sg_atg[['PLAYER NAME','AVERAGE']].copy()
df_short_acc = df_short_acc[['PLAYER NAME','AVG DTP']].copy()


# In[9]:


df_drive_acc.head(10)


# In[10]:


data_frames = [df_drive_acc,df_short_acc,df_three_putt,df_sg_atg,df_sg_off_the_tee,df_drive,df_sg_putt,df_avg_scr ]


# In[11]:


from functools import reduce
df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['PLAYER NAME'],
                                            how='inner'), data_frames)


# In[ ]:





# In[12]:


df_losers = df_merged


# In[13]:


df_losers.tail(50)


# In[ ]:





# In[ ]:





# In[14]:


dfs = [df_merged, df_results]


# In[15]:


dffs = [df_merged, df_current_players]


# In[16]:


df_merged_results = reduce(lambda  left,right: pd.merge(left,right,on=['PLAYER NAME'],
                                            how='inner'), dfs)


# In[17]:


df_merged_players = reduce(lambda  left,right: pd.merge(left,right,on=['PLAYER NAME'],
                                            how='inner'), dffs)


# In[18]:


df_merged_results = df_merged_results.sort_values(by="Year")


# In[ ]:





# In[19]:


df_merged_results


# In[20]:


df_merged_results['Year'] = 1


# In[21]:


df_merged_results.columns = ['PLAYER NAME','Fairway %','AVG DTP','3 PUTT %','AVERAGE_x','AVERAGE_y','AVG.','Sg_Putt','AVG_SCR','Winner']


# In[22]:


df_merged_results.tail(50)


# In[23]:


df_losers.tail(50)


# In[24]:


df_losers['Winner'] = 0


# In[25]:


df_losers.iloc[:50]


# In[26]:


df_losers.iloc[50:100]


# In[27]:


df_losers.iloc[150:200]


# In[28]:


df_losers.iloc[100:150]


# In[29]:


column = df_merged_results['PLAYER NAME'].values


# In[30]:


column


# In[31]:


indexs = [51,202,82,142,106,176,195,129,119,85]


# In[32]:


df_losers = df_losers.drop(index=[51,202,82,142,106,176,195,129,119,85])


# In[33]:


df_losers.tail(50)


# In[34]:


df_losers.reset_index(drop=True)


# In[35]:


df_merged_results


# In[36]:


#df_merged_results = df_merged_results.rename(columns={'Year':'Winner'})


# In[37]:


df_losers


# In[38]:


df4 = pd.concat([df_losers,df_merged_results], axis=0)


# In[39]:


df4


# In[40]:


df4.reset_index(drop=True)


# In[41]:


df4 = df4.set_index('PLAYER NAME')


# In[42]:


df_merged_players = df_merged_players.set_index('PLAYER NAME')


# In[43]:


df4.head(10)


# In[44]:


df4 =df4.drop(columns='AVG DTP')


# In[45]:


df_merged_players =df_merged_players.drop(columns='AVG DTP')


# In[46]:


df4


# In[47]:


## split data
dtf_train, dtf_test = model_selection.train_test_split(df4, 
                      test_size=0.1)
## print info
print("X_train shape:", dtf_train.drop("Winner",axis=1).shape, "| X_test shape:", dtf_test.drop("Winner",axis=1).shape)
print("y_train mean:", round(np.mean(dtf_train["Winner"]),2), "| y_test mean:", round(np.mean(dtf_test["Winner"]),2))
print(dtf_train.shape[1], "features:", dtf_train.drop("Winner",axis=1).columns.to_list())


# In[48]:


#dtf_train = df4
#dtf_test = df_merged_players


# In[49]:


X_train = dtf_train.drop(columns='Winner').values
y_train = dtf_train["Winner"].values
#X_test = dtf_test.drop(columns='Winner').values
#y_test = dtf_test["Winner"].values


# In[50]:


#X_train = dtf_train.drop(columns='Winner').values
#y_train = dtf_train["Winner"].values
X_test = dtf_test.values


# In[51]:


df4['AVG_SCR'].dtypes


# In[ ]:





# In[52]:


scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(dtf_train.drop("Winner", axis=1))
dtf_scaled= pd.DataFrame(X, columns=dtf_train.drop("Winner", axis=1).columns, index=dtf_train.index)
dtf_scaled["Winner"] = dtf_train["Winner"]
dtf_scaled.head(5)


# In[53]:


X = dtf_train.drop("Winner", axis=1).values
y = dtf_train["Winner"].values
feature_names = dtf_train.drop("Winner", axis=1).columns
## Anova
selector = feature_selection.SelectKBest(score_func=  
               feature_selection.f_classif, k='all').fit(X,y)
anova_selected_features = feature_names[selector.get_support()]

## Lasso regularization
selector = feature_selection.SelectFromModel(estimator= 
              linear_model.LogisticRegression(C=1, penalty="l1", 
              solver='liblinear'), max_features=5).fit(X,y)
lasso_selected_features = feature_names[selector.get_support()]
 
## Plot
dtf_features = pd.DataFrame({"features":feature_names})
dtf_features["anova"] = dtf_features["features"].apply(lambda x: "anova" if x in anova_selected_features else "")
dtf_features["num1"] = dtf_features["features"].apply(lambda x: 1 if x in anova_selected_features else 0)
dtf_features["lasso"] = dtf_features["features"].apply(lambda x: "lasso" if x in lasso_selected_features else "")
dtf_features["num2"] = dtf_features["features"].apply(lambda x: 1 if x in lasso_selected_features else 0)
dtf_features["method"] = dtf_features[["anova","lasso"]].apply(lambda x: (x[0]+" "+x[1]).strip(), axis=1)
dtf_features["selection"] = dtf_features["num1"] + dtf_features["num2"]
sns.barplot(y="features", x="selection", hue="method", data=dtf_features.sort_values("selection", ascending=False), dodge=False)


# In[54]:


X_train = dtf_train.drop(columns='Winner').values
y_train = dtf_train["Winner"].values
#X_test = dtf_test.drop(columns='Winner').values
#y_test = dtf_test["Winner"].values


# In[55]:


#X_train = dtf_train.drop(columns='Winner').values
#y_train = dtf_train["Winner"].values
X_test = df_merged_players.values


# In[56]:


X_test


# In[57]:


from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[58]:


y_pred


# In[59]:


df_merged_players['Winner'] = y_pred


# In[60]:


df_merged_players.head(50)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




