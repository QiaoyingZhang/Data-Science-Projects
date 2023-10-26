# Classify Therapy Response Using NLP

## About Word Polarity
The process involves analyzing both flagged and unflagged text data from the training dataset to identify potential positive, neutral, and negative words. For classification purposes, the criteria are as follows: If a word exclusively appears in unflagged text or is significantly more frequent in unflagged text compared to flagged text, it is categorized as a positive word. Conversely, if a word is found exclusively in flagged text, it is categorized as a negative word. Additionally, if a word is equally or more prevalent in flagged text than in unflagged text, it is classified as a neutral word. This word classification approach is essential for training the model to recognize and differentiate between sentiment and context in responses, ultimately enabling the chatbot to make informed judgments regarding the mental health implications in user interactions.

## About AdaBoost (Best ML Algorithm Performance)
AdaBoost, short for Adaptive Boosting, is a machine learning ensemble method that is used for classification and regression tasks. It's a popular algorithm that combines multiple weak learners (usually decision trees) to create a strong learner. The main idea behind AdaBoost is to focus on the examples that are difficult to classify and give them more weight during the training process. Key characteristics of AdaBoost include its adaptability to focus on challenging examples and its ability to improve model accuracy by combining the strength of multiple weak learners. This makes it a robust and effective algorithm for a variety of classification and regression tasks.

## Dataset
The dataset within Sheet_1.csv comprises 80 user responses, located in the response_text column, given in response to a therapy chatbot prompt that asks, "Describe a time when you have acted as a resource for someone else." In this context, responses are categorized into two groups: those labeled as 'not flagged' indicate that the user can continue interacting with the chatbot, while those labeled as 'flagged' signify that the user should be directed to seek assistance. The objective is to develop a classifier capable of categorizing new responses as either 'flagged' or 'not flagged.'

## Purpose of Project
I aim to develop a therapy response classifier designed to categorize responses to a chatbot question that asks, "Describe a time when you have acted as a resource for someone else." The primary objective of this classifier is to discern whether the context implies that the friend being discussed may be experiencing mental health issues and requires a warning or assistance. This classifier operates by comparing textual responses to a preprocessed training dataset, evaluating them based on positive, neutral, and negative features. It ultimately makes predictions to flag responses as either "in need of help" or "able to continue the conversation." This tool will serve as a valuable addition to chatbot functionality, enabling it to proactively identify potential mental health concerns and provide appropriate support or resources, thereby contributing to user well-being.

The classifier's underlying mechanism involves the utilization of machine learning techniques to understand and categorize responses. By training on labeled data and analyzing the sentiment and context within responses, it equips the chatbot with the capability to not only engage in conversations but also to recognize and respond to situations where mental health issues might be a concern. This proactive approach to offering assistance is a testament to the potential of technology in promoting mental health awareness and support in a compassionate and timely manner, ultimately fostering a safer and more empathetic online environment.

## References
* SAMDEEPLEARNING.(2016). "Deep-NLP," Kaggle, https://www.kaggle.com/datasets/samdeeplearning/deepnlp/data.
* SHINIGAMI.(2021). "Sentimental Analysis," Kaggle, https://www.kaggle.com/code/gargmanas/sentimental-analysis/notebook.
* YEŞILÖZ, SERCAN.(2021). "Therapy Chatbot (NLP)," Kaggle, https://www.kaggle.com/code/sercanyesiloz/therapy-chatbot-nlp/notebook.






