import json
import nltk
import pickle
import random
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = []
tags = []
docs = []
punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

# Clean intents file
# Store tags and associate pattern words for training
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_lst = nltk.word_tokenize(pattern)
        words.extend(word_lst)
        docs.append((word_lst, intent['tag']))

        if intent['tag'] not in tags:
            tags.append(intent['tag'])

# Ignore punctation and lemmatize words
words = [lemmatizer.lemmatize(word) for word in words if word not in punc]
words = sorted(set(words))
tags = sorted(set(tags))

# Serialize and store processed words and tags
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(tags, open('tags.pkl', 'wb'))

train_data = []
empty_output = [0]*len(tags)

# Create randomized training data 
for doc in docs:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    # bag-of-words
    for word in words: bag.append(1) if word in word_patterns else bag.append(0)

    # one-hot-encoding of tags
    output = list(empty_output)
    output[tags.index(doc[1])] = 1    # vectorize
    train_data.append(bag + output)

random.shuffle(train_data)

train_data = np.array(train_data)
x_train = train_data[:, :len(words)] # words
y_train = train_data[:, len(words):] # tags

# Train tensorflow keras model
model = tf.keras.Sequential()
# Add linear stack of layers
model.add(tf.keras.layers.Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5)) # reduce overfitting
model.add(tf.keras.layers.Dense(64, activation='relu'))
# connect layers and ensure probabilty distribution over possible tags
model.add(tf.keras.layers.Dense(len(y_train[0]), activation='softmax')) 

# Optimize model using gradient descent (with momentum)
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Run model 200 times to retain accuracy
hist = model.fit(np.array(x_train), np.array(y_train), epochs=200, batch_size=5, verbose=1)

# Store model for chatbot
model.save('chatbot_model.h5', hist)

print('Executed')