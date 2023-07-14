#!/usr/bin/env python
# coding: utf-8

# # # Movie Recommendation System
# This is a ML-based approach to filtering or predicting movies to the user based on their previous past choices and preferences.
# Types of Recommendation Systems
# A) Content-Based Movie Recommendation Systems
# B) Collaborative Filtering Movie Recommendation Systems
# C) Popularity based Recommendation Systems
# Here, I am using cosine similarity.

# In[3]:


import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# TfidfVectorizer - conerts text to numerical value

# ### Data-Collection and Pre-Processing

# In[4]:


#loading the data from the csv file to apnadas dataframe
movies_data = pd.read_csv("movies.csv")


# In[5]:


#printing the firts 5 rows of the dataframe
movies_data.head()


# In[6]:


#number of rows nad columns in the data frame
movies_data.shape


# In[7]:


# selecting the relevant features for recommendation

selected_features = ['genres','keywords','tagline','cast','director']
print(selected_features)


# In[8]:


#Cleaning the Data- replacing the null valuess with null string

for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')


# In[9]:


# combining all the 5 selected features

combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']


# In[11]:


print(combined_features)


# In[12]:


# converting the text data to feature vectors

vectorizer = TfidfVectorizer()


# In[13]:


feature_vectors = vectorizer.fit_transform(combined_features)


# In[14]:


print(feature_vectors)


# In[15]:


# getting the similarity scores using cosine similarity

similarity = cosine_similarity(feature_vectors)


# In[16]:


print(similarity)


# In[17]:


print(similarity.shape)


# ## Getting the Movie Name from the User

# In[18]:


movie_name = input(' Enter your favourite movie name : ')


# In[19]:


# creating a list with all the movie names given in the dataset

list_of_all_titles = movies_data['title'].tolist()
print(list_of_all_titles)


# In[20]:


# finding the close match for the movie name given by the user

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
print(find_close_match)


# In[21]:


close_match = find_close_match[0]
print(close_match)


# In[22]:


# finding the index of the movie with title

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
print(index_of_the_movie)


# In[23]:


# getting a list of similar movies

similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)


# In[24]:


len(similarity_score)


# In[25]:


# sorting the movies based on their similarity score

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
print(sorted_similar_movies)


# In[26]:


# print the name of similar movies based on the index

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1


# ## MOVIE RECCOMENDATION SYSTEM

# In[27]:


movie_name = input(' Enter your favourite movie name : ')

list_of_all_titles = movies_data['title'].tolist()

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

close_match = find_close_match[0]

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

print('Movies suggested for you : \n')

i = 1

for movie in sorted_similar_movies:
  index = movie[0]
  title_from_index = movies_data[movies_data.index==index]['title'].values[0]
  if (i<30):
    print(i, '.',title_from_index)
    i+=1


# ## Thank You!
