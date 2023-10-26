import json
import nltk
import pickle
import random
import numpy as np
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Retrieve preprocessed data and model
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
tags = pickle.load(open('tags.pkl', 'rb'))
words = pickle.load(open('words.pkl', 'rb'))
intents = json.loads(open('intents.json').read())

def clean_query(query):
    """
    Clean input query by lemmatization.

    query: string
    """
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    word_lst = nltk.word_tokenize(query)
    word_lst = [lemmatizer.lemmatize(word) for word in word_lst if word not in punc]

    return word_lst

def bag_of_words(query):
    """
    Get occurrence of words in query within intents patterns.

    query: string
    """
    word_lst = clean_query(query)
    bag = [0]*len(words)

    for word in word_lst:
        for i, w in enumerate(words):
            if w == word:
                bag[i] = 1

    return np.array(bag)

def predict_tag(query):
    """
    Predict the category of query.

    query: string
    """
    bow = bag_of_words(query)
    res = model.predict(np.array([bow]))[0]
    
    # Get list of possible intents and their scores
    ERROR_THRESHOLD = 0.25
    results = [[i, p] for i, p in enumerate(res) if p>ERROR_THRESHOLD]
    results.sort(key=lambda x:x[1], reverse=True)
    
    tag_lst = []
    for r in results:
        tag_lst.append({'intent':tags[r[0]], 'probability':str(r[1])})

    return tag_lst

def get_response(intents_lst, intents_json):
    """
    Retrieve response that falls under certain tags
    after prediction.

    intents_lst: list of possible intents
    intents_json: intents json file
    """
    intents_arr = intents_json['intents']
    tag = intents_lst[0]['intent'] # get best prediction
    
    # Find best prediction
    # Return random response from that tag
    for i in intents_arr:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break

    return result

print('Bot functioning. Hi, how may I help you? Enter quit or exit to leave.')

exit_conditions = ('quit', 'exit')

# Run bot
while True:
    query = input('>')
    ints = predict_tag(query)
    res = get_response(ints, intents)

    if query in exit_conditions:
        break
    else:
        print(res)
