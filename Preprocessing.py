import pandas as pd
import numpy as np

def drop_cols(df,col_names):

	if len(col_names) != 0: 
		df.drop(col_names, inplace = True, axis = 1)
	else:
		print('col_names is empty')	
	return df


def drop_rows(df,col,condition,year_filter):

	if year_filter == None:	

		n_,m_ = df.shape

		if   type(condition) == int:
			df.drop(df[df[col]==condition].index,inplace = True, axis = 0)
		elif type(condition) == str:	
			df.drop(df[df[col]==str(condition)].index,inplace = True, axis = 0)

		n,m = df.shape
		print('The number of rows dropped by ',col,condition,'is', n_ - n)
		print('----                ----                ----                ----')
		return df

	else:
		n_,m_ = df.shape

		if   type(year_filter) == str:	
			df.drop(df[df[col] <= str(year_filter)].index,inplace = True, axis = 0)

		n,m = df.shape
		print('The number of rows dropped by year',year_filter,'is', n_ - n)
		print('----                ----                ----                ----')
		return df

def drop_rows_B(df,col,condition,year_filter):

	if year_filter == None:	

		n_,m_ = df.shape

		if   type(condition) == int:
			df.drop(df[df[col]==condition].index,inplace = True, axis = 0)
		elif type(condition) == str:	
			df.drop(df[df[col]==str(condition)].index,inplace = True, axis = 0)

		n,m = df.shape
		print('The number of rows dropped by ',col,condition,'is', n_ - n)
		print('----                ----                ----                ----')
		return df

	else:
		n_,m_ = df.shape

		if   type(year_filter) == str:	
			df.drop(df[df[col] >= str(year_filter)].index,inplace = True, axis = 0)

		n,m = df.shape
		print('The number of rows dropped by year',year_filter,'is', n_ - n)
		print('----                ----                ----                ----')
		return df

def zero_one_replacement(df,col):

	df[col] = df[col].notnull().astype('int')	
	return df



df = pd.read_csv('/Users/alexdonezwell/Desktop/cs534-master/final_pj/movies_metadata.csv')
print('Original data shape is',df.shape)

col_to_drop =  ['video',
				# 'poster_path',
				# 'imdb_id',
				'belongs_to_collection']

df = drop_cols(df,col_to_drop)
# df = drop_rows(df,'budget',str(0),None)
# print('Data shape after drop rows with 0 budget',df.shape)
df = drop_rows(df,'revenue',0,None)
df = df.dropna(subset=['revenue']) 
# print('Data shape after drop rows with 0 revenue',df.shape)
df = drop_rows(df,'vote_count',0,None)
# print('vote',df.shape)
# df = drop_rows(df,'release_date', None, str(2011))
# df = drop_rows(df,'release_date', None, str(2001))
df = drop_rows_B(df,'release_date',None,str(2011))
# df = df_temp.merge(df, on="imdb_id", how = 'inner')
print('Data shape after drop movie earlier than',2012,df.shape)
df = zero_one_replacement(df,'homepage')
df.to_csv(r'/Users/alexdonezwell/Desktop/cs534-master/final_pj/Meta_cleaned_2001_2011.csv')



