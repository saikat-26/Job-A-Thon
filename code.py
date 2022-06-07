#importing necessary packages and libraries

import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#importing data from csv file

rawData=pd.read_csv(r'../input/jobathon-june-22/train_wn75k28.csv')
rawTestData=pd.read_csv(r'../input/jobathon-june-22/test_Wf7sxXF.csv')

#setting up the data into a data frame

finalTest=pd.DataFrame(rawTestData, columns=['campaign_var_1', 'campaign_var_2', 'products_purchased', 'user_activity_var_1', 'user_activity_var_2', 'user_activity_var_3', 'user_activity_var_4', 'user_activity_var_5', 'user_activity_var_6', 'user_activity_var_7', 'user_activity_var_8', 'user_activity_var_9', 'user_activity_var_10', 'user_activity_var_11', 'user_activity_var_12' ])
X=pd.DataFrame(rawData, columns=['campaign_var_1', 'campaign_var_2', 'products_purchased', 'user_activity_var_1', 'user_activity_var_2', 'user_activity_var_3', 'user_activity_var_4', 'user_activity_var_5', 'user_activity_var_6', 'user_activity_var_7', 'user_activity_var_8', 'user_activity_var_9', 'user_activity_var_10', 'user_activity_var_11', 'user_activity_var_12' ])
Y=pd.DataFrame(rawData, columns=['buy'])

#handling missing data

for x in X.index:
  if math.isnan(X.loc[x, "products_purchased"]):
    X.loc[x, "products_purchased"] = 0
for x in finalTest.index:
  if math.isnan(finalTest.loc[x, "products_purchased"]):
    finalTest.loc[x, "products_purchased"] = 0

rf=RandomForestClassifier(n_estimators=400, bootstrap=True, max_depth=70, min_samples_leaf=4, min_samples_split=10)

#fitting the given data into the model

rf.fit(X,Y)

#predicting results from given test data

Y_pred=rf.predict(finalTest)

#processing the results into a csv file

arr=[]
for i in range(39162,52346):
    arr.append(str(i))
output=pd.DataFrame(Y_pred, columns=['buy'], index=arr)
output.to_csv('SubmissionFinal.csv')