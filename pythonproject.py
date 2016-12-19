#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 01:04:40 2016

@author: rungsunan
"""

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import json
import matplotlib.pyplot as plt
import pandas as pd
#import logging #This is for debugging gensim
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import sklearn.naive_bayes as naive_bayes

from yelp.client import Client
from yelp.oauth1_authenticator import Oauth1Authenticator


bus_list = []
whole_dict = {}

print("Getting data ready to analyze")

#user_dict holds username and elite user years and elite status for easier lookup
user_dict = {}
for line in open('yelp_academic_dataset_user.json')    :
    temp_dict = {}    
    temp_dict.update(json.loads(line))
    userid = temp_dict["user_id"]
    elite = temp_dict["elite"]
    if len(elite) > 0:
        elite_status = 1
    else:
        elite_status = 0
    user_dict[userid] = {"elite":[temp_dict["elite"]],"elite_status":elite_status}

#from the user dict make a list of just elite user id's to compare

eliteuser_list = []
for k,v in user_dict.items():
    if v["elite_status"] == 1:
        eliteuser_list.append(k)
# restaurant dict is the business dict for just restaurants 
res_dict = {}

for line in open('yelp_academic_dataset_business.json')    :
    temp_dict = {}    
    temp_dict.update(json.loads(line))
    busid = temp_dict["business_id"]
    categories = temp_dict["categories"]
    longitude = temp_dict["longitude"]
    latitude = temp_dict["latitude"]
    whole_dict[busid] = categories
    for i in range(len(categories)):
        if categories[i]=="Restaurants":
            res_dict[busid]={"categories":categories[i],"longitude":[temp_dict["longitude"]],"latitude":[temp_dict["latitude"]]}
#with open('res_dict.json', 'w') as fp:
#    json.dump(res_dict, fp)

print("size of whole busid:categorylist -> " + str(len(whole_dict)))  
print("number of restaurants: " + str(len(res_dict)))

    
review_dict = {}
outcome_list = []

#review dict is a business indexed with review text, 
for line in open('10000_yelp_academic_dataset_review.json')    :
    temp_dict = {}    
    temp_dict.update(json.loads(line))
    review_business = temp_dict["business_id"]
    for business in res_dict.keys():
        if review_business == business:
            if review_business in review_dict:
                review_dict[review_business]["user_id"].append(temp_dict["user_id"])  
                review_dict[review_business]["categories"] = res_dict[business]["categories"]
                review_dict[review_business]["revtext"].append(temp_dict["text"])
                review_dict[review_business]["rating"].append(temp_dict["stars"])
                review_dict[review_business]["longitude"] = res_dict[business]["longitude"]
                review_dict[review_business]["latitude"] = res_dict[business]["latitude"]
                review_dict[review_business]["useful"].append(temp_dict["votes"]["useful"])       
            else:
                review_dict[review_business]={"user_id":[temp_dict["user_id"]],"categories":res_dict[business]["categories"],"revtext":[temp_dict["text"]],"rating":[temp_dict["stars"]],"useful":[temp_dict["votes"]["useful"]],"latitude":res_dict[business]["latitude"],"longitude":res_dict[business]["longitude"]}
print("number of businesses in review dictionary: " + str(len(review_dict)))

#This list is the useful users in reviews
useful_user = []
for k,review in review_dict.items():
    for i in range(len(review['useful'])):
        if review['useful'][i]>0:
            useful_user.append(review['user_id'][i])

#counting how many useful users are elite users
useful_elite_counter = 0
for j in range(len(useful_user)):
    for i in range(len(eliteuser_list)):
        if eliteuser_list[i]==useful_user[j]:
            useful_elite_counter = useful_elite_counter +1
#set(useful_user) & set(eliteuser_list)
print (str(useful_elite_counter/len(useful_user)*100) + "% of elite users are useful users")
#Creating sets of documents and outcomes for Naive Bayes and ldamodel training
doc_set = []
useful_doc_set = []
outcomes_list = []

for k in review_dict.keys():
    for i in range(len(review_dict[k]["revtext"])):
        doc_set.append(review_dict[k]["revtext"][i])
        if review_dict[k]["useful"][i] >= 3:
            useful_doc_set.append(review_dict[k]["revtext"][i])
        if review_dict[k]["useful"][i] < 3:
            outcomes_list.append(0)
        else:
            outcomes_list.append(1)
#            useful_doc_set.append(review_dict[k]["revtext"][i])
    
print("there are " + str(len(outcomes_list)) + " reviews in all")    
    
texts = []
tokenizer = RegexpTokenizer(r'\w+')
    
# create English stop words list
en_stop = get_stop_words('en')
   
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# loop through document list
for i in range(len(useful_doc_set)):
# clean and tokenize document string
    raw = useful_doc_set[i].lower()
    tokens = tokenizer.tokenize(raw)
    # remove stop words from tokens
    tokens = [word for word in tokens if len(word)>1]
    stopped_tokens = [k for k in tokens if not k in en_stop]

    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stemmed_tokens) 
       
while True:
    print("Welcome to yelp Text Analysis!")        
    print("1. make plots with API")
    print("2. Naive Bayes Classification")    
    print("3. LDA topic modeling")
    print("0. exit")
    print("What do you want to do:")
    choice = int(input("Your choice:"))
    if choice == 0:
        break
    elif choice == 1:
        plt.style.use("ggplot")
        auth = Oauth1Authenticator(
            consumer_key="xxxxxxxxxxxxxxxxxxxxxx",
            consumer_secret="xxxxxxxxxxxxxxxxxxxxxxxxxxx",
            token="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            token_secret="xxxxxxxxxxxxxxxxxxxxxxxxxxx"
        )
         
        client = Client(auth)
        
        location_zip = input("Please enter zip code: ")
         
        #ll = 40.439226, -80.0017249
        avgRating = {}
        reviewCt = {}
        #cuisines = []
        num_categories = int(input("Please enter number of categories(eg. 1-5): "))
        for i in range(num_categories):
            food_type = input("Please enter a category(eg. italian,chinese): ").lower()
            params = {
                      'category_filter': food_type,
                      'lang': 'en',
                      'radius': 100000
                      }
            #resp_search = client.search_by_coordinates(latitude,longitude, **params)
            resp_search=client.search(location_zip, **params)
            total = 0
            count = 0
            for j in range(len(resp_search.businesses)):
                print(str(resp_search.businesses[j].name.encode('utf-8').strip()) + " " +
                  " rating: " + str(resp_search.businesses[j].rating) +
                  " review_count: " + str(resp_search.businesses[j].review_count))
                total = total + resp_search.businesses[j].rating
                count= count + resp_search.businesses[j].review_count
            cat_avg = total/len(resp_search.businesses)
            #avgRating.append(cat_avg)
            #cuisines.append(food_type)
            avgRating[food_type]= cat_avg
            reviewCt[food_type] = count
        print(avgRating)
        print(reviewCt)

        s1 = pd.Series(avgRating,name="Average Rating")
        s2 = pd.Series(reviewCt,name ="Review Count")
        df = pd.concat([s1, s2], axis=1)
        df.index.name = "Cuisine"
        df.reset_index()
        
        df_fig=pd.DataFrame(df,index=df.index,columns=["Average Rating","Review Count"])
        fig, ax=plt.subplots(figsize=(12,7))
        ax2 = ax.twinx()
        df_fig["Review Count"].plot(kind = "bar", color = "red", ax = ax, align = "center",position = 0.3, alpha = 0..5, width = 0.5,  grid=False, title="Restaurants Near Your Searched Location")
        df_fig["Average Rating"].plot(kind = "line", color = "blue", marker="o", ms = 10, ax=ax2, grid=True)
        ax.yaxis.tick_left()
        ax2.yaxis.tick_right()
        ax.set_ylabel("Number of Reviews")
        ax2.set_ylabel("Average Rating", rotation = -90)
        plt.axis("tight")
        plt.tight_layout()
        ax2.set_ylim(0,5.0)
        
        for p in range(num_categories):
            ax2.annotate(str(df_fig["Average Rating"][p]),(df_fig.index.get_loc(df.index[p]),df_fig["Average Rating"][p]),xytext=(df_fig.index.get_loc(df.index[p]),df_fig["Average Rating"][p]+0.3), ha="center",va="top")
        plt.show()
    elif choice == 2:
        print("Creating Naive Bayes Classifier")
        #Naive Bayes classifier from sci-kit learn     
        from sklearn.feature_extraction.text import CountVectorizer
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(doc_set)
        from sklearn.feature_extraction.text import TfidfTransformer
        tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
        X_train_tf = tf_transformer.transform(X_train_counts)
        X_train_tf.shape
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        X_train_tfidf.shape
        clf = naive_bayes.MultinomialNB().fit(X_train_tfidf, outcomes_list)
        print("model created!")
       
        doc_typed_string = str(input("Please enter a review text to classify:"))
        doc_typed = [ doc_typed_string ]
        X_typed_counts = count_vect.transform(doc_typed)
        X_typed_tfidf = tfidf_transformer.transform(X_typed_counts)
        predicted_typed = clf.predict(X_typed_tfidf)
        if predicted_typed[0]==0:
            print("That's a bad review!")
        else:
            print("That's a useful review!")
        input('Press enter to continue: ')
        
        print("If we could classify with a larger review set, this would be an example of a useful review:")
        
        docs_new = [ "now called Tara Thai ~ http://www.yelp.com/biz/tara-thai-phoenix#hrid:jZ5ynxfXefkDBzm4vB1E4A" ]
        print("inputing: " + str(docs_new))
        X_new_counts = count_vect.transform(docs_new)
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)
        predicted = clf.predict(X_new_tfidf)
        doc_use_count = 0 

        for doc, category in zip(docs_new, predicted):
            if category == 1:
                print("That's a useful review!")
                #print('%r => %s' % (doc, category))
                doc_use_count=doc_use_count+1
                #print (doc_use_count)
            else:
                print("This is a bad review!")
        input('Press enter to continue: ')
    elif choice == 3:
        print("Creating an LDA model from useful reviews")
        dictionary = corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below=10, no_above=0.5)
        print("there are: " + str(len(texts)) + " review documents in the dictionary")
        print("dictionary done!")
        print("making ldamodel")
        # convert tokenized documents into a document-term matrix for p2
        corpus = [dictionary.doc2bow(text) for text in texts]
        ldamodel = models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary,  alpha='auto',passes=5)
        print("These are the most common words found in useful reviews")
        for i in ldamodel.show_topics(num_topics=2, num_words=50):
            print (i)
        input('Press enter to continue: ')
