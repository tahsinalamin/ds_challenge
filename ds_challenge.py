"""
Author: Sikder Tahsin Al Amin
Description: Eluvio DS challenge. 
How to Run: python3 ds_challenge.py dataset="/path_to_folder/datasetname.csv"
"""

#importing libraries
import numpy as np 
import pandas as pd 
import sys 
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sklearn.feature_extraction.text as text
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

#command line arguments
#sys.argv=["k_gamma.py","dataset=Eluvio_DS_challenge.csv"]

#reading dataset
if len(sys.argv)!=2:
    print("Error: Run the program as python3 ds_challenge.py dataset=datasetname.csv")
    sys.exit(0)

arg2 = sys.argv[1]
arg2 = arg2.split("=")
dataset = arg2[1] #get name of the dataset 
    
df = pd.read_csv(dataset,skiprows=0)

## doing simple analytics on the data set
print(df.head(5)) #take a look at the dataset
print(df.shape) # rows=509236 and columns=8
print(df.info()) #no null values;; int,bool, and obj
print(df.describe()) #mean, std_dev, count from int columns
print(df.describe(include=['object'])) #object types count, unique, freq. 

#check the down_vote and category column
print(len(df.down_votes.unique()))
print(len(df.category.unique()))


### Feature engineering #####

#drop the down_votes and category column as it has one unique value
df = df.drop("down_votes",axis=1)
df = df.drop("category",axis=1)

#convert the binary column to integer 0,1
df["over_18"] = df["over_18"].astype(int)

#extract year,month,day, day of week from date_created and make 4 new columns
df['date_created'] = pd.to_datetime(df['date_created'])
df['year'] = pd.DatetimeIndex(df['date_created']).year
df['month'] = pd.DatetimeIndex(df['date_created']).month
df['day'] = pd.DatetimeIndex(df['date_created']).day
df['day_of_week']=df['date_created'].dt.dayofweek
df = df.drop("date_created",axis=1)
print(df.head(5))

# extract hour from time_created and make a new column
df['time_created'] = pd.to_datetime(df['time_created'], unit='s')
df['hour']=pd.DatetimeIndex(df['time_created']).hour
df = df.drop("time_created",axis=1)
print(df.head(5))

# label encoding of the author column; unique author = 509236 .
# label encoding because one-hot encoding will introduce curse of dimensionality
df["author_id"] = LabelEncoder().fit_transform(df.author)
print(df.head(5))



"""
Problem 1: Predict the number of Up votes given the data set.
"""

#forking the previous dataframe 
df1 = df

# detect and remove outlier values in up_votes by computing Z-score
df1 = df1[(np.abs(stats.zscore(df1["up_votes"])) <= 4)] 
print (len(df1)) #rows=499545

df1 = df1.drop("author",axis=1)

####### Handling Text data (Title column) #####
title = df1["title"].str.lower()

#get the list of stopwords
stopwords = text.ENGLISH_STOP_WORDS

#TF-IDF vectorize text - highlight more interesting words
vectorizer = TfidfVectorizer(analyzer = 'word', stop_words=stopwords)
tfidf_matrix = vectorizer.fit_transform(title)
print(tfidf_matrix.shape)

# select 80% as upvote class 1, otherwise 0
up_vote_threshold = np.percentile(df.up_votes,80)
def up_vote_class(row):
    if row["up_votes"] >= up_vote_threshold:
        val = 1
    else:
        val = 0
    return val 

df1["up_votes"] = df1.apply(up_vote_class,axis=1)
y = np.array(df1.up_votes)

y = df1.up_votes
y = np.array(y)

##train test split
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, y, test_size=0.20, shuffle=True, random_state=42)


## multinominalNB model. As its best for text/document classifications
clf = MultinomialNB()
clf.fit(X_train,y_train)
y_predict = clf.predict(X_test)

print(classification_report(y_test,y_predict))

## RandomForest.
rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(X_train, y_train)
y_predict = rfc.predict(X_test)

print(classification_report(y_test, y_predict))

## Support vector machine; linear kernel is best suited for text classification
clf2 = svm.SVC(kernel="linear")
clf2.fit(X_train, y_train)
y_predict = clf2.predict(X_test)

print(classification_report(y_test, y_predict))


"""
Problem 2: Sentiment analysis
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#fork the original data set.
df2 = df

title = df2["title"].str.cat(sep=' ').lower() #lower case; otherwise string mismatch

tokens = word_tokenize(title)

stop_words = set(stopwords.words('english')) #get stopwords
#those who are not stopwords and is a word (removing punctuation symbols)
tokens = [w for w in tokens if not w in stop_words and w.isalpha()] 

vocabulary = set(tokens)
print(len(vocabulary)) #81265

frequency_dist = nltk.FreqDist(tokens)

#print(sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)[0:50])

wordcloud = WordCloud().generate_from_frequencies(frequency_dist)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()



"""
Problem 3: Business Analytics
"""
import seaborn as sns
sns.set()

# Number of posts vs Year plot
g=sns.countplot(x='year',data=df)
g.set_title('Number of posts vs Year')

# Number of posts vs Day of Week plot
g=sns.countplot(x='day_of_week',data=df)
g.set_title('Number of posts vs Day of Week')

# Number of posts vs Time of Day plot
g=sns.countplot(x='hour',data=df)
g.set_title('Number of posts vs Time of Day')


#top 10 authors with most up votes
author_most_up_votes = df.groupby(["author"])["up_votes"].sum()
author_most_up_votes = author_most_up_votes.sort_values(ascending=False).head(10)
#print(author_most_up_votes)
author_most_up_votes.columns = ["total"]
ax = author_most_up_votes.plot.bar( y='total')
ax.set_title("Top 10 authors with Up votes of all time")

#top 10 authors who publishes most on any arbitrary month ex: December
december_authors = df[df["month"]==12]
december_authors = december_authors.groupby("author").size()
december_authors = december_authors.sort_values(ascending=False).head(10)
december_authors.columns = ["total"]
ax = december_authors.plot.bar( y='total')
ax.set_title("Top authors with number of posts in month=December")


#top upvoted post and its author by each year
top_votes_by_year = df.groupby("year").agg({"up_votes":np.max}) #get top votes by year
posts_by_year = pd.merge(top_votes_by_year, df2, on=["up_votes","year"]) #join with original dataframe
print(posts_by_year[["year","title","author"]]) #filter the columns
