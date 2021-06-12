<div class="cell markdown">

# MOVIE RECOMMENDATION

</div>

<div class="cell markdown">

# Introduction

</div>

<div class="cell markdown">

The main goal of this machine learning project is to build a
recommendation engine that recommends movies to users.This python
project is designed to help you understand the functioning of how a
recommendation system works.

</div>

<div class="cell markdown">

# What is a Recommendation System?

</div>

<div class="cell markdown">

A recommendation system provides suggestions to the users through a
filtering process that is based on user preferences and browsing
history. A recommendation system is a platform that provides its users
with various contents based on their preferences and likings. A
recommendation system takes the information about the user as an input.

</div>

<div class="cell markdown">

# Contents

</div>

<div class="cell markdown" data-_cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0" data-_uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a" data-collapsed="true">

1.  Introduction
2.  What is a Recommendation System?
3.  Importing important libraries
4.  Reading dataset
5.  Missing values imputation
6.  Feature Engineering
7.  Visualisation
8.  Recommending Movies Based on Languages
9.  Recommending Movies Based on Actor
10. Recommending movies on similar genres
11. Recommending Similar Movies
12. Conclusion

</div>

<div class="cell markdown">

# Importing Important Libraries

</div>

<div class="cell code" data-execution_count="1">

``` python
# basic libraries
import pandas as pd
import numpy as np

# basic libraries for data visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# libraries for widgets
import ipywidgets as widgets
from ipywidgets import interact
from ipywidgets import interact_manual

# for interactive shells
from IPython.display import display

# for setting background for graph
plt.rcParams["figure.figsize"] = (16,8)
plt.style.use("fivethirtyeight")

# for removing error
import warnings
warnings.filterwarnings('ignore')
```

</div>

<div class="cell markdown">

# Reading Dataset

</div>

<div class="cell code" data-execution_count="2">

``` python
# reading dataset
data = pd.read_csv("movie_metadata.csv")
```

</div>

<div class="cell code" data-execution_count="3">

``` python
# printing first five line 
data.head(5)
```

<div class="output execute_result" data-execution_count="3">

``` 
   color      director_name  num_critic_for_reviews  duration  \
0  Color      James Cameron                   723.0     178.0   
1  Color     Gore Verbinski                   302.0     169.0   
2  Color         Sam Mendes                   602.0     148.0   
3  Color  Christopher Nolan                   813.0     164.0   
4    NaN        Doug Walker                     NaN       NaN   

   director_facebook_likes  actor_3_facebook_likes      actor_2_name  \
0                      0.0                   855.0  Joel David Moore   
1                    563.0                  1000.0     Orlando Bloom   
2                      0.0                   161.0      Rory Kinnear   
3                  22000.0                 23000.0    Christian Bale   
4                    131.0                     NaN        Rob Walker   

   actor_1_facebook_likes        gross                           genres  ...  \
0                  1000.0  760505847.0  Action|Adventure|Fantasy|Sci-Fi  ...   
1                 40000.0  309404152.0         Action|Adventure|Fantasy  ...   
2                 11000.0  200074175.0        Action|Adventure|Thriller  ...   
3                 27000.0  448130642.0                  Action|Thriller  ...   
4                   131.0          NaN                      Documentary  ...   

  num_user_for_reviews language  country  content_rating       budget  \
0               3054.0  English      USA           PG-13  237000000.0   
1               1238.0  English      USA           PG-13  300000000.0   
2                994.0  English       UK           PG-13  245000000.0   
3               2701.0  English      USA           PG-13  250000000.0   
4                  NaN      NaN      NaN             NaN          NaN   

   title_year actor_2_facebook_likes imdb_score  aspect_ratio  \
0      2009.0                  936.0        7.9          1.78   
1      2007.0                 5000.0        7.1          2.35   
2      2015.0                  393.0        6.8          2.35   
3      2012.0                23000.0        8.5          2.35   
4         NaN                   12.0        7.1           NaN   

  movie_facebook_likes  
0                33000  
1                    0  
2                85000  
3               164000  
4                    0  

[5 rows x 28 columns]
```

</div>

</div>

<div class="cell code" data-execution_count="4">

``` python
# checking shape of our dataset
print(data.shape)
```

<div class="output stream stdout">

    (5043, 28)

</div>

</div>

<div class="cell code" data-execution_count="5">

``` python
# printing inforamtion of our dataset
data.info()
```

<div class="output stream stdout">

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5043 entries, 0 to 5042
    Data columns (total 28 columns):
     #   Column                     Non-Null Count  Dtype  
    ---  ------                     --------------  -----  
     0   color                      5024 non-null   object 
     1   director_name              4939 non-null   object 
     2   num_critic_for_reviews     4993 non-null   float64
     3   duration                   5028 non-null   float64
     4   director_facebook_likes    4939 non-null   float64
     5   actor_3_facebook_likes     5020 non-null   float64
     6   actor_2_name               5030 non-null   object 
     7   actor_1_facebook_likes     5036 non-null   float64
     8   gross                      4159 non-null   float64
     9   genres                     5043 non-null   object 
     10  actor_1_name               5036 non-null   object 
     11  movie_title                5043 non-null   object 
     12  num_voted_users            5043 non-null   int64  
     13  cast_total_facebook_likes  5043 non-null   int64  
     14  actor_3_name               5020 non-null   object 
     15  facenumber_in_poster       5030 non-null   float64
     16  plot_keywords              4890 non-null   object 
     17  movie_imdb_link            5043 non-null   object 
     18  num_user_for_reviews       5022 non-null   float64
     19  language                   5031 non-null   object 
     20  country                    5038 non-null   object 
     21  content_rating             4740 non-null   object 
     22  budget                     4551 non-null   float64
     23  title_year                 4935 non-null   float64
     24  actor_2_facebook_likes     5030 non-null   float64
     25  imdb_score                 5043 non-null   float64
     26  aspect_ratio               4714 non-null   float64
     27  movie_facebook_likes       5043 non-null   int64  
    dtypes: float64(13), int64(3), object(12)
    memory usage: 1.1+ MB

</div>

</div>

<div class="cell code" data-execution_count="6">

``` python
# removing unnecassary columns form our dataset
data = data.drop(["color","director_facebook_likes","actor_3_facebook_likes","cast_total_facebook_likes",
                 "facenumber_in_poster","content_rating","country","movie_imdb_link","aspect_ratio","plot_keywords"],
                axis=1)
data.columns
```

<div class="output execute_result" data-execution_count="6">

    Index(['director_name', 'num_critic_for_reviews', 'duration', 'actor_2_name',
           'actor_1_facebook_likes', 'gross', 'genres', 'actor_1_name',
           'movie_title', 'num_voted_users', 'actor_3_name',
           'num_user_for_reviews', 'language', 'budget', 'title_year',
           'actor_2_facebook_likes', 'imdb_score', 'movie_facebook_likes'],
          dtype='object')

</div>

</div>

<div class="cell markdown">

# Missing values imputation

</div>

<div class="cell code" data-execution_count="7">

``` python
# checking the rows having high percentage of missing values in our dataset
round(100*(data.isnull().sum()/len(data.index)),2)
```

<div class="output execute_result" data-execution_count="7">

    director_name              2.06
    num_critic_for_reviews     0.99
    duration                   0.30
    actor_2_name               0.26
    actor_1_facebook_likes     0.14
    gross                     17.53
    genres                     0.00
    actor_1_name               0.14
    movie_title                0.00
    num_voted_users            0.00
    actor_3_name               0.46
    num_user_for_reviews       0.42
    language                   0.24
    budget                     9.76
    title_year                 2.14
    actor_2_facebook_likes     0.26
    imdb_score                 0.00
    movie_facebook_likes       0.00
    dtype: float64

</div>

</div>

<div class="cell code" data-execution_count="8">

``` python
# since gross and budget columns having large number of nan value so we will remove these two columns from our dataset
data=data[~np.isnan(data["gross"])]
data = data[~np.isnan(data["budget"])]
```

</div>

<div class="cell code" data-execution_count="9">

``` python
# again checking the nan values
data.isnull().sum()
```

<div class="output execute_result" data-execution_count="9">

    director_name              0
    num_critic_for_reviews     1
    duration                   1
    actor_2_name               5
    actor_1_facebook_likes     3
    gross                      0
    genres                     0
    actor_1_name               3
    movie_title                0
    num_voted_users            0
    actor_3_name              10
    num_user_for_reviews       0
    language                   3
    budget                     0
    title_year                 0
    actor_2_facebook_likes     5
    imdb_score                 0
    movie_facebook_likes       0
    dtype: int64

</div>

</div>

<div class="cell code" data-execution_count="10">

``` python
# the rows foe which the sum of nan is less than two are retained
data = data[data.isnull().sum(axis=1) <=2]
data.isnull().sum()
```

<div class="output execute_result" data-execution_count="10">

    director_name             0
    num_critic_for_reviews    1
    duration                  1
    actor_2_name              0
    actor_1_facebook_likes    0
    gross                     0
    genres                    0
    actor_1_name              0
    movie_title               0
    num_voted_users           0
    actor_3_name              5
    num_user_for_reviews      0
    language                  3
    budget                    0
    title_year                0
    actor_2_facebook_likes    0
    imdb_score                0
    movie_facebook_likes      0
    dtype: int64

</div>

</div>

<div class="cell code" data-execution_count="11">

``` python
# lets fill the nan values
# we are remvoving nan value of numerical columns by mean
data["num_critic_for_reviews"].fillna(data["num_critic_for_reviews"].mean(),inplace=True)
# we are remvoving nan value of categorical columns by mode
data["duration"].fillna(data["duration"].mean(),inplace=True)
# we know that we can not use statistical values for removing the missing 
data["language"].fillna(data["language"].mode()[0],inplace =True)
data["actor_3_name"].fillna("Unknown Actor",inplace = True)
data.isnull().sum().sum()
```

<div class="output execute_result" data-execution_count="11">

``` 
0
```

</div>

</div>

<div class="cell markdown">

# Feature Engineering

</div>

<div class="cell code" data-execution_count="12">

``` python
# here we are convert the gross and budget from $ to million $ to make our analysis easier
data["gross"] = data["gross"]/1000000
data["budget"] = data["budget"]/1000000
```

</div>

<div class="cell code" data-execution_count="13">

``` python
# here we are creating a profit column using thr bydget and gross
data["profit"] = data["gross"] - data["budget"]
# here we are checking the name of top 10 profitable movies
data[["profit","movie_title"]].sort_values(by = "profit",ascending=False).head(10)
```

<div class="output execute_result" data-execution_count="13">

``` 
          profit                                 movie_title
0     523.505847                                     Avatar 
29    502.177271                             Jurassic World 
26    458.672302                                    Titanic 
3024  449.935665         Star Wars: Episode IV - A New Hope 
3080  424.449459                 E.T. the Extra-Terrestrial 
794   403.279547                               The Avengers 
17    403.279547                               The Avengers 
509   377.783777                              The Lion King 
240   359.544677  Star Wars: Episode I - The Phantom Menace 
66    348.316061                            The Dark Knight 
```

</div>

</div>

<div class="cell code" data-execution_count="14">

``` python
# by looking at the above result we can easily analyze that there are some duplicate
# here we are printing the no.of rows before removing duplicates
print("No of rows before removing duplicates:", data.shape[0])
# here we are removing all the duplicates from the data
data.drop_duplicates(subset = None,keep="first",inplace=True)
# here we are printing the no.of rows after removing duplicates
print("No of rows after removing duplicates:", data.shape[0])
```

<div class="output stream stdout">

    No of rows before removing duplicates: 3886
    No of rows after removing duplicates: 3851

</div>

</div>

<div class="cell code" data-execution_count="15">

``` python
# here we are checking top 10 profitable movies again
data[["movie_title","profit"]].sort_values(by ="profit",ascending = False).head(10)
```

<div class="output execute_result" data-execution_count="15">

``` 
                                     movie_title      profit
0                                        Avatar   523.505847
29                               Jurassic World   502.177271
26                                      Titanic   458.672302
3024         Star Wars: Episode IV - A New Hope   449.935665
3080                 E.T. the Extra-Terrestrial   424.449459
17                                 The Avengers   403.279547
509                               The Lion King   377.783777
240   Star Wars: Episode I - The Phantom Menace   359.544677
66                              The Dark Knight   348.316061
439                            The Hunger Games   329.999255
```

</div>

</div>

<div class="cell code" data-execution_count="16">

``` python
# here we are checking the values in the language column
data["language"].value_counts()
```

<div class="output execute_result" data-execution_count="16">

    English       3672
    French          37
    Spanish         26
    Mandarin        14
    German          13
    Japanese        12
    Hindi           10
    Cantonese        8
    Italian          7
    Portuguese       5
    Korean           5
    Norwegian        4
    Dutch            3
    Danish           3
    Thai             3
    Persian          3
    Hebrew           2
    Aboriginal       2
    Dari             2
    Indonesian       2
    Vietnamese       1
    Telugu           1
    Arabic           1
    Kazakh           1
    Hungarian        1
    Maya             1
    Czech            1
    Russian          1
    Bosnian          1
    Romanian         1
    None             1
    Icelandic        1
    Zulu             1
    Aramaic          1
    Mongolian        1
    Swedish          1
    Dzongkha         1
    Filipino         1
    Name: language, dtype: int64

</div>

</div>

<div class="cell code" data-execution_count="17">

``` python
# looking at the above output we can easily observe that out of 3500 movies only 150 movies are of other than english
# so it is better to keep only two languages that is english and foregin
def language(x):
    if x == "English":
        return "English"
    else:
        return "Foregin"
# here we are applying the function on the language column
data['language'] = data["language"].apply(language)
# here we checking values again
data["language"].value_counts()
```

<div class="output execute_result" data-execution_count="17">

    English    3672
    Foregin     179
    Name: language, dtype: int64

</div>

</div>

<div class="cell code" data-execution_count="18">

``` python
# the duration of movies is not varying a lot but we know that most of the users either like watching long or short duration movie.
# duration movies we can categorise the movie in two part i.e. short and long
# here we are defining a function for categorizing duration of movies
def duration(x):
    if x <= 120:
        return "short"
    else:
        return "Long"
# here we are applying this function on the duration column
data["duration"] = data["duration"].apply(duration)
# here er are checking the values of duration column
data['duration'].value_counts()
```

<div class="output execute_result" data-execution_count="18">

    short    2934
    Long      917
    Name: duration, dtype: int64

</div>

</div>

<div class="cell code" data-execution_count="19">

``` python
# here we are spliting genres
data["genres"].str.split("|")[0]
```

<div class="output execute_result" data-execution_count="19">

    ['Action', 'Adventure', 'Fantasy', 'Sci-Fi']

</div>

</div>

<div class="cell code" data-execution_count="20">

``` python
# we can see from the above cell that most of the movies are having a lot of genres
# also a movie can have so many genres so lets keep four genres
data["Moviegenres"] = data["genres"].str.split("|")
data['genres1'] = data["Moviegenres"].apply(lambda x: x[0])
# some of the movies have only one genre in such case assign the some genres to genres_2 as well
data['genres2'] = data["Moviegenres"].apply(lambda x: x[1] if len(x) > 1 else x[0])
data['genres3'] = data["Moviegenres"].apply(lambda x:x[2] if len(x) > 2 else x[0])
data['genres4'] = data["Moviegenres"].apply(lambda x:x[3] if len(x) > 3 else x[0])
# here we are checking the head of the data
data[["genres","genres1","genres2","genres3","genres4"]].head(5)
```

<div class="output execute_result" data-execution_count="20">

``` 
                            genres genres1    genres2   genres3 genres4
0  Action|Adventure|Fantasy|Sci-Fi  Action  Adventure   Fantasy  Sci-Fi
1         Action|Adventure|Fantasy  Action  Adventure   Fantasy  Action
2        Action|Adventure|Thriller  Action  Adventure  Thriller  Action
3                  Action|Thriller  Action   Thriller    Action  Action
5          Action|Adventure|Sci-Fi  Action  Adventure    Sci-Fi  Action
```

</div>

</div>

<div class="cell markdown">

# Data Visualisation

</div>

<div class="cell code" data-execution_count="21">

``` python
# here we are calculating the social media popularity of movie
# to calculate popularity of a movie we can aggregate no of voted users no of users for reviews and facebook
data["Social_Media_Popularity"] = (data["num_user_for_reviews"]/
                                   data["num_voted_users"])*data["movie_facebook_likes"]
# here we are checking top 10 most popular movies on social media
x = data[["movie_title",'Social_Media_Popularity']].sort_values(by = "Social_Media_Popularity",
                                                               ascending=False).head(10).reset_index()
print(x)
sns.barplot(x["movie_title"],x["Social_Media_Popularity"],palette="magma")
plt.title("top 10 most popular movies on social medai",fontsize=20)
plt.xticks(rotation=90,fontsize=14)
plt.xlabel(" ")
plt.show()
```

<div class="output stream stdout">

``` 
   index                          movie_title  Social_Media_Popularity
0     10  Batman v Superman: Dawn of Justice               1599.794424
1    150                        Ghostbusters               1076.336425
2   1582                        Ghostbusters               1075.827482
3     96                        Interstellar               1024.560802
4   3015               10 Days in a Madhouse                828.025478
5    945                      Into the Woods                692.937200
6     73                       Suicide Squad                652.816996
7   1190                Fifty Shades of Grey                624.306881
8    108                            Warcraft                622.790277
9     92        Independence Day: Resurgence                599.274128
```

</div>

<div class="output display_data">

![](bc47288d6fb7064753e900ad12ab838fc6175f97.png)

</div>

</div>

<div class="cell code" data-execution_count="22">

``` python
# here we are comparing the gross with genres
# first group the genres and get max,min and avg gross of the movies of that genres
display(data[["genres1","gross",]].groupby(["genres1"]).agg(["max",'mean',"min"]).style.background_gradient(cmap="Blues"))
# here we are ploting these values using lineplot
data[["genres1","gross",]].groupby(["genres1"]).agg(["max",'mean',"min"]).plot(kind="line",color =["red","black","blue"])
plt.title("Which Genre is most Bankable?", fontsize=20)
plt.xticks(np.arange(17),["Action","Adventure","Animation","Biography","comedy",'crime',
                          "Documentry","Drama","Family","Fantasy","Horror","Musical",
                          "Mystery","Romance","Sci-Fi","Thriller","Western"],rotation=90,fontsize=15)
plt.ylabel("Gross",fontsize=15)
plt.xlabel(" ")
plt.show()
print("The most profitable movie from each genre")
display(data.loc[data.groupby(data["genres1"])["profit"].idxmax()][["genres1",
                                                                  "movie_title","gross"]].style.background_gradient(cmap="copper"))
```

<div class="output display_data">

    <pandas.io.formats.style.Styler at 0x1eb64ee9e50>

</div>

<div class="output display_data">

![](fc301055967d51bdb980642011691146dbd74c36.png)

</div>

<div class="output stream stdout">

    The most profitable movie from each genre

</div>

<div class="output display_data">

    <pandas.io.formats.style.Styler at 0x1eb64ee9160>

</div>

</div>

<div class="cell code" data-execution_count="23">

``` python
# here we are converting year into integer
data['title_year'] = data["title_year"].astype("int")
print('most profitable years in box office')
display(data[["title_year",'language','profit']].groupby(["language",
                                                         "title_year"]).agg("sum").sort_values(by="profit",
                                                          ascending = False).head(10).style.background_gradient(cmap='Greens'))
# here we are plotting them
sns.lineplot(data["title_year"],data['profit'],hue=data["language"])
plt.title("time series for box office profit for english vs forigen language",fontsize=20)
plt.xticks(fontsize=18)
plt.xlabel(" ")
plt.show()
print("movies that msde huge losses")
display(data[data["profit"]< -2000][["movie_title",
                                    "language","profit"]].style.background_gradient(cmap="Reds"))
```

<div class="output stream stdout">

    most profitable years in box office

</div>

<div class="output display_data">

    <pandas.io.formats.style.Styler at 0x1eb65c31fa0>

</div>

<div class="output display_data">

![](8e5a64393d02c6fb1d503a96cf81e3847c559f1f.png)

</div>

<div class="output stream stdout">

    movies that msde huge losses

</div>

<div class="output display_data">

    <pandas.io.formats.style.Styler at 0x1eb65c31c10>

</div>

</div>

<div class="cell code" data-execution_count="24">

``` python
display(data[data["duration"]=="Long"][["movie_title","duration","gross",
                                       "profit"]].sort_values(by="profit", ascending= False).head(5).style.background_gradient(cmap="spring"))
display(data[data["duration"]=="short"][["movie_title","duration","gross",
                                       "profit"]].sort_values(by="profit", ascending= False).head(5).style.background_gradient(cmap="spring"))
sns.barplot(data["duration"],data["gross"],hue= data["language"],palette="spring")
plt.title("gross comparsion")
```

<div class="output display_data">

    <pandas.io.formats.style.Styler at 0x1eb64e91190>

</div>

<div class="output display_data">

    <pandas.io.formats.style.Styler at 0x1eb65d1e280>

</div>

<div class="output execute_result" data-execution_count="24">

    Text(0.5, 1.0, 'gross comparsion')

</div>

<div class="output display_data">

![](50f9fe11eb94d714b475e49c650c718fee1d695b.png)

</div>

</div>

<div class="cell code" data-execution_count="25">

``` python
print("average IMDB score for long duration movie is {0:.2f}".format(data[data["duration"]=="Long"]["imdb_score"].mean()))
print("average TMDB score for short duration movie is {0:.2f}".format(data[data["duration"]== "short"]["imdb_score"].mean()))
print("\nhightest rated long duration movie\n",
     data[data["duration"]=="Long"][["movie_title","imdb_score"]].sort_values(by="imdb_score",ascending=False).head(1))
print("\nhightest rated short duration movie\n",
     data[data["duration"]=="short"][["movie_title",'imdb_score']].sort_values(by="imdb_score",ascending=False).head(1))
sns.boxplot(data["imdb_score"],data["duration"],palette="copper")
plt.title("imdb rating vs gross", fontsize=20)
plt.xticks(rotation=90)
plt.show()
```

<div class="output stream stdout">

    average IMDB score for long duration movie is 7.06
    average TMDB score for short duration movie is 6.28
    
    hightest rated long duration movie
                         movie_title  imdb_score
    1937  The Shawshank Redemption          9.3
    
    hightest rated short duration movie
                   movie_title  imdb_score
    3175  American History X          8.6

</div>

<div class="output display_data">

![](20e04c522fe83798004a638a09bd1cd8070965a6.png)

</div>

</div>

<div class="cell code" data-execution_count="26">

``` python
def query_actors(x):
    a = data[data["actor_1_name"]==x]
    b = data[data["actor_2_name"]==x]
    c = data[data["actor_3_name"]==x]
    x = a.append(b)
    y = x.append(c)
    y = y[["movie_title",
          "budget",
          "gross",
          "title_year",
          "genres",
          "language",
          "imdb_score",
          ]]
    return y
```

</div>

<div class="cell code" data-execution_count="27">

``` python
query_actors("Meryl Streep")
```

<div class="output execute_result" data-execution_count="27">

``` 
                         movie_title  budget       gross  title_year  \
410                It's Complicated     85.0  112.703470        2009   
1106                 The River Wild     45.0   46.815748        1994   
1204                  Julie & Julia     40.0   94.125426        2009   
1408          The Devil Wears Prada     35.0  124.732962        2006   
1483                Lions for Lambs     35.0   14.998070        2007   
1575                  Out of Africa     31.0   87.100000        1985   
1618                   Hope Springs     30.0   63.536011        2012   
1674                 One True Thing     30.0   23.209440        1998   
1925                      The Hours     25.0   41.597830        2002   
2781                  The Iron Lady     13.0   29.959436        2011   
3135       A Prairie Home Companion     10.0   20.338609        2006   
860               Death Becomes Her     55.0   58.422650        1992   
914                      Mamma Mia!     52.0  143.704210        2008   
945                  Into the Woods     50.0  127.997349        2014   
1111                  The Ant Bully     50.0   28.133159        2006   
1295              Fantastic Mr. Fox     40.0   20.999103        2009   
1920                      The Giver     25.0   45.089048        2014   
1941           August: Osage County     25.0   37.738400        2013   
2086  The Bridges of Madison County     35.0   70.960517        1995   
2206                          Doubt     20.0   33.422556        2008   
2386                    Adaptation.     19.0   22.245861        2002   
1801                      Rendition     27.5    9.664316        2007   
2067                  Marvin's Room     23.0   12.782508        1996   

                                         genres language  imdb_score  
410                        Comedy|Drama|Romance  English         6.6  
1106            Action|Adventure|Crime|Thriller  English         6.3  
1204                    Biography|Drama|Romance  English         7.0  
1408                       Comedy|Drama|Romance  English         6.8  
1483                         Drama|Thriller|War  English         6.2  
1575                    Biography|Drama|Romance  English         7.2  
1618                       Comedy|Drama|Romance  English         6.3  
1674                                      Drama  English         7.0  
1925                              Drama|Romance  English         7.6  
2781                    Biography|Drama|History  English         6.4  
3135                         Comedy|Drama|Music  English         6.8  
860                       Comedy|Fantasy|Horror  English         6.4  
914               Comedy|Family|Musical|Romance  English         6.3  
945      Adventure|Comedy|Drama|Fantasy|Musical  English         6.0  
1111  Adventure|Animation|Comedy|Family|Fantasy  English         5.9  
1295    Adventure|Animation|Comedy|Crime|Family  English         7.8  
1920                       Drama|Romance|Sci-Fi  English         6.5  
1941                                      Drama  English         7.3  
2086                              Drama|Romance  English         7.5  
2206                              Drama|Mystery  English         7.5  
2386                               Comedy|Drama  English         7.7  
1801                             Drama|Thriller  English         6.8  
2067                                      Drama  English         6.7  
```

</div>

</div>

<div class="cell code" data-execution_count="28">

``` python
def actors_report(x):
    a = data[data["actor_1_name"]==x]
    b = data[data["actor_2_name"]==x]
    c = data[data["actor_3_name"]==x]
    x = a.append(b)
    y = x.append(c)
    print("Time:",y["title_year"].min(),y["title_year"].max())
    print("Max Gross:{0:.2f} Millions".format(y["gross"].max()))
    print("Avg gross:{0:.2f} Millions".format(y["gross"].mean()))
    print("Min Gross:{0:.2f} Millions".format(y["gross"].min()))
    print("number of 100 Millions Movies:",y[y["gross"]>100].shape[0])
    print("Aug IMDB Score: {0:.2f}".format(y["imdb_score"].mean()))
    print("most common Genres:\n",y["genres1"].value_counts().head())
actors_report("Meryl Streep")
```

<div class="output stream stdout">

    Time: 1985 2014
    Max Gross:143.70 Millions
    Avg gross:55.23 Millions
    Min Gross:9.66 Millions
    number of 100 Millions Movies: 4
    Aug IMDB Score: 6.81
    most common Genres:
     Drama        9
    Comedy       7
    Biography    3
    Adventure    3
    Action       1
    Name: genres1, dtype: int64

</div>

</div>

<div class="cell code" data-execution_count="29">

``` python
# here we are comparing brad pitt, leonardo caprio, and Tom cruise
def critically_acclaimed_actors(x):
    a = data[data["actor_1_name"]==x]
    b = data[data["actor_2_name"]==x]
    c = data[data["actor_3_name"]==x]
    x = a.append(b)
    y = x.append(c)
    return y["num_critic_for_reviews"].sum().astype("int")
print("number of critic reviews for brad pitt")
display(critically_acclaimed_actors("Brad Pitt"))
print("number of critic reviews for Tom Cruise")
display(critically_acclaimed_actors("Tom Cruise"))
print("number of critic reviews for Leonardo DiCaprio")
display(critically_acclaimed_actors("Leonardo DiCaprio"))
```

<div class="output stream stdout">

    number of critic reviews for brad pitt

</div>

<div class="output display_data">

    7814

</div>

<div class="output stream stdout">

    number of critic reviews for Tom Cruise

</div>

<div class="output display_data">

    6740

</div>

<div class="output stream stdout">

    number of critic reviews for Leonardo DiCaprio

</div>

<div class="output display_data">

    7014

</div>

</div>

<div class="cell code" data-execution_count="30">

``` python
# here we are printing movie based of imdb score using jypter widgets
@interact
def show_movies_more_than(columns="imdb_score",score=9.0):
    x = data.loc[data[columns]>score][["title_year","movie_title",
                                    "director_name",
                                    "actor_1_name",
                                     "actor_2_name",
                                     "actor_3_name",
                                     "profit",
                                     "imdb_score",
                                     ]]
    x = x.sort_values(by = "imdb_score",ascending=False)
    x = x.drop_duplicates(keep="first")
    return x
```

<div class="output display_data">

``` json
{"model_id":"ecfc959c7ea14569b0bd9f334e54a19f","version_major":2,"version_minor":0}
```

</div>

</div>

<div class="cell code" data-execution_count="31">

``` python
pd.set_option("max_rows",30000)
@interact
def show_articles_more_than(column=["budget","gross"],x=1000):
    return data.loc[data[column]>x][["movie_title","duration","gross","profit","imdb_score"]]
```

<div class="output display_data">

``` json
{"model_id":"1c426471d2a7443c98d8b0d62e8a88c2","version_major":2,"version_minor":0}
```

</div>

</div>

<div class="cell markdown">

# Recommending Movies Based on Languages

</div>

<div class="cell code" data-execution_count="32">

``` python
def recommend_lang(x):
    y = data[["language",'movie_title',"imdb_score"]][data["language"] == x]
    y = y.sort_values(by="imdb_score",ascending=False)
    return y.head(15)
```

</div>

<div class="cell code" data-execution_count="33">

``` python
recommend_lang("Foregin")
```

<div class="output execute_result" data-execution_count="33">

``` 
     language                      movie_title  imdb_score
4498  Foregin  The Good, the Bad and the Ugly          8.9
4747  Foregin                   Seven Samurai          8.7
4029  Foregin                     City of God          8.7
2373  Foregin                   Spirited Away          8.6
4259  Foregin             The Lives of Others          8.5
4921  Foregin              Children of Heaven          8.5
3931  Foregin                         Samsara          8.5
1298  Foregin                          Amélie          8.4
2323  Foregin               Princess Mononoke          8.4
1329  Foregin        Baahubali: The Beginning          8.4
4105  Foregin                          Oldboy          8.4
2970  Foregin                        Das Boot          8.4
4659  Foregin                    A Separation          8.4
2734  Foregin                      Metropolis          8.3
4033  Foregin                        The Hunt          8.3
```

</div>

</div>

<div class="cell code" data-execution_count="34">

``` python
recommend_lang("English")
```

<div class="output execute_result" data-execution_count="34">

``` 
     language                                        movie_title  imdb_score
1937  English                          The Shawshank Redemption          9.3
3466  English                                     The Godfather          9.2
2837  English                            The Godfather: Part II          9.0
66    English                                   The Dark Knight          9.0
3355  English                                      Pulp Fiction          8.9
1874  English                                  Schindler's List          8.9
339   English     The Lord of the Rings: The Return of the King          8.9
683   English                                        Fight Club          8.8
270   English  The Lord of the Rings: The Fellowship of the R...         8.8
2051  English    Star Wars: Episode V - The Empire Strikes Back          8.8
97    English                                         Inception          8.8
836   English                                      Forrest Gump          8.8
3867  English                   One Flew Over the Cuckoo's Nest          8.7
340   English             The Lord of the Rings: The Two Towers          8.7
3024  English                Star Wars: Episode IV - A New Hope          8.7
```

</div>

</div>

<div class="cell markdown">

# Recommending Movies Based on Actor

</div>

<div class="cell code" data-execution_count="35">

``` python
def recommend_movies_on_actors(x):
    a = data[["movie_title","imdb_score"]][data["actor_1_name"]==x]
    b = data[["movie_title","imdb_score"]][data["actor_2_name"]==x]
    c = data[["movie_title","imdb_score"]][data["actor_3_name"]==x]
    a = a.append(b)
    a = a.append(c)
    a = a.sort_values(by = "imdb_score",ascending=False)
    return a.head(15)
```

</div>

<div class="cell code" data-execution_count="36">

``` python
recommend_movies_on_actors("Tom Cruise")
```

<div class="output execute_result" data-execution_count="36">

``` 
                                            movie_title  imdb_score
1868                                          Rain Man          8.0
75                                    Edge of Tomorrow          7.9
284                                    Minority Report          7.7
158                                   The Last Samurai          7.7
736                                         Collateral          7.6
1524                                    A Few Good Men          7.6
940   Interview with the Vampire: The Vampire Chroni...         7.6
155               Mission: Impossible - Ghost Protocol          7.4
135                 Mission: Impossible - Rogue Nation          7.4
671                                     Eyes Wide Shut          7.3
930                                      Jerry Maguire          7.3
3128                                     The Outsiders          7.2
2768                        Born on the Fourth of July          7.2
370                                           Valkyrie          7.1
438                                Mission: Impossible          7.1
```

</div>

</div>

<div class="cell markdown">

# Recommending movies on similar genres

</div>

<div class="cell code" data-execution_count="37">

``` python
from mlxtend.preprocessing import TransactionEncoder
x = data["genres"].str.split("|")
te = TransactionEncoder()
x = te.fit_transform(x)
x = pd.DataFrame(x,columns=te.columns_)
x.head()
```

<div class="output error" data-ename="ModuleNotFoundError" data-evalue="No module named &#39;mlxtend&#39;">

    ---------------------------------------------------------------------------
    ModuleNotFoundError                       Traceback (most recent call last)
    <ipython-input-37-3a8d1c0b9193> in <module>
    ----> 1 from mlxtend.preprocessing import TransactionEncoder
          2 x = data["genres"].str.split("|")
          3 te = TransactionEncoder()
          4 x = te.fit_transform(x)
          5 x = pd.DataFrame(x,columns=te.columns_)
    
    ModuleNotFoundError: No module named 'mlxtend'

</div>

</div>

<div class="cell code">

``` python
genres = x.astype("int")
genres.head()
```

</div>

<div class="cell code">

``` python
genres.insert(0, "movie_title",data["movie_title"])
genres.head(8)
```

</div>

<div class="cell code">

``` python
genres = genres.set_index("movie_title")
genres.head()
```

</div>

<div class="cell code">

``` python
def recommendation_genres(gen):
    gen = genres[gen]
    similar_genres = genres.corrwith(gen)
    similar_genres = similar_genres.sort_values(ascending=False)
    similar_genres = similar_genres.iloc[1:]
    return similar_genres.head(3)
```

</div>

<div class="cell code">

``` python
recommendation_genres("Action")
```

</div>

<div class="cell code">

``` python
x = genres.transpose()
x.head()
```

</div>

<div class="cell markdown">

# Recommending Similar Movies

</div>

<div class="cell code">

``` python
def recommendation_movie(movie):
    movie = x[movie+'\xa0']
    similar_movies = x.corrwith(movie)
    similar_movies = similar_movies.sort_values(ascending=False)
    similar_movies = similar_movies.iloc[1:]
    return similar_movies.head(20)
```

</div>

<div class="cell code">

``` python
recommendation_movie("The Expendables")
```

</div>

<div class="cell markdown">

# Conclusion

</div>

<div class="cell markdown">

Thanks for reading. I hope you like my recoomendation and visualization
found it to be helpful. If you have any questions or suggestions, feel
free to write them down in the comment section.

</div>
