# Building Therapy ChatBot Using NLP Artificial Neural Network (ANN)

## Files
* **bot_trainer.py** - functions to train a therapy chatbot model
* **bot.py** - functions to predict response to query
* **chatbot_model.h5** - hdf5 file that stores pre-trained model
* **intents.json** - json file with all possible patterns and their responses
* **tags.pkl** - pickle file of the tags of intents
* **words.pkl** - pickle file of the tokenized words of patterns
  
![chat-bot-preview](https://github.com/user-attachments/assets/25fab9be-fb2d-4a62-a18b-3f71f6ef213b)

## About ANN And ReLU
An Artificial Neural Network is a framework for machine learning that emulates the workings of the human brain. It consists of interconnected processing units, called neurons, organized into layers. These layers typically include an input layer, one or more hidden layers, and an output layer. 

ReLU, or Rectified Linear Unit, is an activation function commonly used in artificial neural networks. It's a simple function defined as f(x) = max(0, x), which means it returns zero for all negative input values and passes positive input values as is.

ReLU-based activation functions introduce non-linearity to the model, allowing it to learn complex relationships in the data. They are computationally efficient and have become the default choice for many deep learning applications (e.g. Chatbot).

## Dataset
The *Mental Health Conversational Data* dataset comprises fundamental dialogues encompassing mental health FAQs, traditional therapy interactions, and general advice given to individuals facing challenges related to anxiety and depression. This dataset serves as a valuable resource for training a chatbot model with therapeutic capabilities, enabling it to offer emotional support to those dealing with anxiety and depression.

Within this dataset, you will find a categorization of "intents." An "intent" represents the underlying purpose or motivation behind a user's message. For instance, when a user expresses, "I am feeling sad" to the chatbot, the associated intent would be "sad." Each intent is accompanied by a collection of Patterns, which are illustrative examples of user messages that align with the intent, and Responses, which are the chatbot's corresponding replies tailored to the specific intent. This diverse array of intents, patterns, and responses collectively constitutes the training data that equips the model to accurately identify and respond to users' particular intentions.

## Purpose of Project
I aspire to create a therapy chatbot dedicated to addressing people's mental health concerns and providing relief from stress. The concept behind this endeavor is to offer users a supportive and non-intimidating platform where they can engage in conversations without the potential anxiety associated with speaking to a psychologist. The prospect of building this chatbot not only aligns with my desire to make a meaningful impact on mental well-being but also empowers me to achieve a significant and beneficial goal. The act of constructing something so purposeful and potentially life-changing is not only fulfilling but also an opportunity to contribute to the well-being of individuals seeking assistance and understanding in their journey to better mental health.


## Reference
* Simpilearn.(2023). "Python Chatbot Tutorial | How to Create Chatbot Using Python | Python For Beginners | Simplilearn," Youtube, https://www.youtube.com/watch?v=t933Gh5fNrc.
* Elvis.(2022). "Mental Health Conversational Data," Kaggle, https://www.kaggle.com/datasets/elvis23/mental-health-conversational-data/data.
