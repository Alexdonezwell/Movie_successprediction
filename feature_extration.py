import pandas as pd
import numpy as np
from ast import literal_eval
from Preprocessing import drop_cols

def Unique_Val(df):
	emp_list = []
	n = df.shape[0]
	for row in range(n):
		emp_list+= df[row]
	return list(set(emp_list))

def Unique_Val_modified(df):
	emp_list = []
	n = df.shape[0]
	for row in range(n):
		emp_list.append(df[row])
	return list(set(emp_list))	

def Create_cols_from_col(df, col_name, list_of_col_names):
	for j in list_of_col_names:
		df[j] = 0
	for i in range(df.shape[0]):
		for j in list_of_col_names:
			if(j in df[col_name].iloc[i]):
				df.loc[i,j] = 1 	
	return df			

def TF_converter(df,old_col_name,new_col_name,termA, termB, termC):
	n = df.shape[0]
	# df[new_col_name] = 0
	for row in range(n):
		# if df[row].contains(termA):
		if df[old_col_name].loc[row] == termA:
			df[old_col_name].loc[row] = 1
		# elif df[row].contains(termB):
		elif df[old_col_name].loc[row] == termB:
			df[old_col_name].loc[row] = 0
		elif df[old_col_name].loc[row] == termC:
			df[old_col_name].loc[row] = 0	
		else:
			print('Exception arise, there is a third type of element in this column')
	return df			

def Big_six_filter(df,col_name,new_col_name,list_of_big_six):
	n = df.shape[0]
	zeros = np.zeros(n)
	df[new_col_name] = zeros
	for row in range(n):
		for ele in list_of_big_six:
			if (ele in df[col_name].loc[row]):
				df[new_col_name].loc[row] = 1
	df = drop_cols(df,[col_name])			
	return df			


df = pd.read_csv('/Users/alexdonezwell/Desktop/cs534-master/final_pj/Meta_cleaned_2001_2011.csv')

df['genres'] = df['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
df['production_companies'] = df['production_companies'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
df['production_countries'] = df['production_countries'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
df['spoken_languages'] = df['spoken_languages'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

genres = Unique_Val(df['genres'])
spoken_languages = Unique_Val(df['spoken_languages'])
production_countries = Unique_Val(df['production_countries'])
original_language = Unique_Val_modified(df['original_language'])
# production_companies = Unique_Val(df['production_companies'])

df = Create_cols_from_col(df,'genres',genres)
df = Create_cols_from_col(df,'spoken_languages',spoken_languages)
df = Create_cols_from_col(df,'production_countries',spoken_languages)
df = Create_cols_from_col(df,'genres',genres)
df = Create_cols_from_col(df,'original_language',original_language)
# df = Create_cols_from_col(df,'production_companies',production_companies)

split_values = df['release_date'].str.split("-", n = 1, expand = True)
df['release_year'] = split_values[0]

list_of_ele_already_modified = ['vote_count',#'imdb_id',
								# 'vote_average',
								'poster_path',
								'Unnamed: 0',
								'adult',
								'id',
								'genres',
								'release_date',
								'spoken_languages',
								'production_countries',
								'original_language']
list_of_big_six              = ['Universal Pictures',
								'Twentieth Century Fox Film Corporation',
								'Columbia Pictures',
								'Warner Bros.',
								'Paramount Pictures',
								'Walt Disney Pictures']

df = drop_cols(df,list_of_ele_already_modified)
df = TF_converter(df,'status','movie_status','Released','Post Production','Rumored')  # Note: there r only two rows in the df that has Post Production
df = Big_six_filter(df,'production_companies','Big_six_bool',list_of_big_six)


# drop    original_title,
	    # overview, 
	    # tagline,
	    # title  for temporary

Drop_temp = ['original_title','overview','tagline','title']	 
df = drop_cols(df,Drop_temp)   



df.to_csv(r'/Users/alexdonezwell/Desktop/cs534-master/final_pj/Meta_cleaned_genres_2001_2011.csv')
print('work done')


### Note: the cols need to deal with: original_title,
								    # overview, 
								    # tagline,title


