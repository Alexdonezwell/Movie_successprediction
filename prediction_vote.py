import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn import preprocessing
from Preprocessing import drop_cols
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
# from feature_selector import FeatureSelector
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import tree

df = pd.read_csv('/Users/alexdonezwell/Desktop/cs534-master/final_pj/Meta_cleaned_genres_2001_2011.csv')
df_img = pd.read_pickle('/Users/alexdonezwell/Desktop/cs534-master/final_pj/poster_features')
df_twiter = pd.read_csv('/Users/alexdonezwell/Desktop/cs534-master/final_pj/my_data_matrix.csv')


df = df.drop_duplicates(subset='imdb_id', keep="last")
# df_img = df_img.drop_duplicates(subset='imdb_id', keep="last")
# df_twiter = df_twiter.drop_duplicates(subset='imdb_id', keep="last")

# df_new = df.merge(df_img, on="imdb_id", how = 'inner')
# print('1',df_new.shape)
# df_new= df_new.merge(df_twiter, on="imdb_id", how = 'inner')
# print('2',df_new.shape)


# df_new.to_csv(r'/Users/alexdonezwell/Desktop/cs534-master/final_pj/merged_meta_img.csv')
df = df.dropna()
# df =pd.con

# y = df.revenue
lab_enc = preprocessing.LabelEncoder()
y = df.vote_average
y = lab_enc.fit_transform(y)
print(df.shape)
X = drop_cols(df,['revenue','imdb_id','vote_average','Unnamed: 0'])
X = StandardScaler().fit_transform(X)


print(X.shape,y.shape)
print(np.where(pd.isnull(df)))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state=1)



def rmsle(y,y0): 
	return np.sqrt(np.mean(np.square(np.log(y+1)-np.log(y0+1)))) 
def rms(y,y0):
	return np.sqrt(np.mean(np.square(y-y0))) 


#Linear model
reg = LinearRegression()
lin_model = reg.fit(X_train, y_train)
y_pred = reg.predict(X_val)

# #Lasso model
reg = Lasso()
lin_model = reg.fit(X_train, y_train)
lasso_pred = reg.predict(X_val)

# #KNN model
knn = KNeighborsRegressor(n_neighbors = 5)
knn_model = knn.fit(X_train, y_train)
knn_y_pred = knn.predict(X_val)

#Random forest
rf = RandomForestRegressor()
rf_model = rf.fit(X_train, y_train)
rf_y_pred = rf.predict(X_val)

#Decision tree
dt = tree.DecisionTreeRegressor()
dt_model = dt.fit(X_train, y_train)
DT_pred = dt_model.predict(X_val)

#importance analysis
model = ExtraTreesClassifier(n_estimators=10)
model.fit(X, y) # use entire dataset
print(list(model.feature_importances_))
importance_idx = sorted(range(len(list(model.feature_importances_))), key=lambda k: list(model.feature_importances_)[k])
print(importance_idx)

# sorted(range(len(list(model.feature_importances_))), key=lambda k: list(model.feature_importances_)[k])

print('RMSE score for linear model is {}'.format(rms(y_val, y_pred)))
print('RMSE score for Lasso model is {}'.format(rms(y_val, lasso_pred)))
print('RMSE score for k-NN model is {}'.format(rms(y_val, knn_y_pred)))
print('RMSE score for Random Forest model is {}'.format(rms(y_val, rf_y_pred)))
print('RMSE score for Decision tree model is {}'.format(rms(y_val, DT_pred)))






