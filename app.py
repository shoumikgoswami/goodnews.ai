from newsapi import NewsApiClient
import pandas as pd
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from IPython.display import HTML


apikey = '999b5c8772b244f5baffa0480f1656b2'
pd.set_option('display.max_colwidth', -1)


def sentiment_list(data):

    sent_list = []
    for i, row in data.iterrows():
        sent_dict={}
        sent_dict['URL'] = data.url[i]
        sent_dict['Title'] = data.title[i]
        sent_dict['Description'] = data.description[i]
        sent_dict['Source'] = data.author[i]
        sent_list.append(sent_dict)
    return sent_list

def analyze_sentiment_sentiwordnet_lexicon(sentence):
	
    text_tokens = nltk.word_tokenize(sentence)
    tagged_text = nltk.pos_tag(text_tokens)
    pos_score = neg_score = token_count = obj_score = 0
    for word, tag in tagged_text:
        ss_set = None
        if 'NN' in tag and list(swn.senti_synsets(word, 'n')):
            ss_set = list(swn.senti_synsets(word, 'n'))[0]
        elif 'VB' in tag and list(swn.senti_synsets(word, 'v')):
            ss_set = list(swn.senti_synsets(word, 'v'))[0]
        elif 'JJ' in tag and list(swn.senti_synsets(word, 'a')):
            ss_set = list(swn.senti_synsets(word, 'a'))[0]
        elif 'RB' in tag and list(swn.senti_synsets(word, 'r')):
            ss_set = list(swn.senti_synsets(word, 'r'))[0]   
        if ss_set:
            pos_score += ss_set.pos_score()
            neg_score += ss_set.neg_score()
            obj_score += ss_set.obj_score()
            token_count += 1
    final_score = pos_score - neg_score
    norm_final_score = round(float(final_score) / token_count, 2)
    final_sentiment = 'positive' if norm_final_score >= 0 else 'negative'
    
    return final_sentiment

def analyze_sentiment_vader_lexicon(sentence, 
                                    threshold=0.1):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(sentence)
    agg_score = scores['compound']
    final_sentiment = 'positive' if agg_score >= threshold\
                                   else 'negative'
    return final_sentiment    

def goodnews():
	newsapi = NewsApiClient(api_key=apikey)
	source_list = 'bbc-news,abc-news,cnbc,cnn,fox-news'
	top_headlines = newsapi.get_top_headlines(sources=source_list,
                                          language='en')
	data = pd.DataFrame.from_dict(top_headlines)
	data = pd.concat([data.drop(['articles'], axis=1), data['articles'].apply(pd.Series)], axis=1)
	data = data.replace('', np.nan, regex=True)
	data = data.dropna()
	sent_list = sentiment_list(data)
	news_data = pd.DataFrame.from_dict(sent_list)
	Sentiword_desc = []
	Vader_desc = []
	Sentiword_title = []
	Vader_title = []
	for i in range(len(news_data)):
	    Sentiword_desc.append(analyze_sentiment_sentiwordnet_lexicon(news_data['Description'][i]))
	    Vader_desc.append(analyze_sentiment_vader_lexicon(news_data['Description'][i]))
	    Sentiword_title.append(analyze_sentiment_sentiwordnet_lexicon(news_data['Title'][i]))
	    Vader_title.append(analyze_sentiment_vader_lexicon(news_data['Title'][i]))
	news_data['Sentiword_title'] = Sentiword_title
	news_data['Vader_title'] = Vader_title
	news_data['Sentiword_desc'] = Sentiword_desc
	news_data['Vader_desc'] = Vader_desc
	net_sentiment = []
	for i in range(len(news_data)):
	    if news_data['Sentiword_title'][i] == 'positive':
	        news_data['Sentiword_title'][i] = 1
	    else:
	        news_data['Sentiword_title'][i] = -1
	    if news_data['Vader_title'][i] == 'positive':
	        news_data['Vader_title'][i] = 1
	    else:
	        news_data['Vader_title'][i] = -1
	    if news_data['Sentiword_desc'][i] == 'positive':
	        news_data['Sentiword_desc'][i] = 1
	    else:
	        news_data['Sentiword_desc'][i] = -1
	    if news_data['Vader_desc'][i] == 'positive':
	        news_data['Vader_desc'][i] = 1
	    else:
	        news_data['Vader_desc'][i] = -1	
	    net_sentiment.append(news_data['Sentiword_title'][i] + news_data['Vader_title'][i] + news_data['Sentiword_desc'][i] + news_data['Vader_desc'][i])

	for i in range(len(net_sentiment)):
	    if net_sentiment[i] > 0:
	        net_sentiment[i] = 'positive'
	    elif net_sentiment[i] < 0:
	        net_sentiment[i] = 'negative'
	    else:
	        net_sentiment[i] = 'neutral'
	news_data['net_sentiment'] = net_sentiment
	news_data_pos = news_data[news_data['net_sentiment']!='negative']
	news_data_pos = news_data_pos.drop(['Sentiword_title', 'Vader_title','Sentiword_desc','Vader_desc', 'net_sentiment'], axis=1)
	news_data_pos = news_data_pos.reset_index()
	news_data_pos = news_data_pos.drop(['index'], axis = 1)
	cols = ['Source','Title','Description','URL']
	news_data_pos = news_data_pos[cols]
	news_data_pos['URL'] = news_data_pos['URL'].apply(lambda x: '<a href="{}">{}</a>'.format(x,x))
	return news_data_pos

APPNAME = "GoodNews"
STATIC_FOLDER = 'static'
TEMPLATE_FOLDER = 'templates'

app = Flask(__name__, static_folder=STATIC_FOLDER, template_folder = TEMPLATE_FOLDER)
app.config.update(
    APPNAME=APPNAME,
    )


@app.route('/')
def index():
    return render_template('main.html', title = 'Home')

@app.route('/get_news', methods=['POST'])
def get_news():
    get_goodnews = goodnews()

    return render_template('results.html', tables=[HTML(get_goodnews.to_html(classes='data', index = False, escape = False))], titles=get_goodnews.columns.values)
        
if __name__ == '__main__':
    app.run()


