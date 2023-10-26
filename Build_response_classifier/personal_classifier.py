import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

class ResponseClassifier:
    """
    Classify responses to be flagged(1) or unflagged(0).
    """
    def __init__(self):
        self.X = None
        self.y = None
        self.features = None
        self.pos_feat = None
        self.neu_feat = None
        self.neg_feat = None
    
    def get_feat_polarity(self, X, y):
        """
        Check for feature polarity(positive, neutral, negative)

        X: ndarray with 100 features in each row
        y: list of labels(flagged or unflagged)
        """
        ori_flag_txt = sum(sum([X[i] for i in list(np.where(y == 1))]))
        ori_flag_feat = self.get_occurance(ori_flag_txt)
        ori_unflag_txt = sum(sum([X[i] for i in list(np.where(y == 0))]))
        ori_unflag_feat = self.get_occurance(ori_unflag_txt)
        
        self.pos_feat = [k for k in ori_unflag_feat.keys() if k in ori_flag_feat.keys() and ori_unflag_feat[k] > ori_flag_feat[k]]
        self.pos_feat += [k for k in ori_unflag_feat.keys() if k not in ori_flag_feat.keys()]
        self.neu_feat = [k for k in ori_unflag_feat.keys() if k in ori_flag_feat.keys() and ori_unflag_feat[k] <= ori_flag_feat[k]]
        self.neg_feat = [k for k in ori_flag_feat.keys() if k not in ori_unflag_feat.keys()]
        
        return self.pos_feat, self.neu_feat, self.neg_feat
    
    def get_occurance(self, feat):
        """
        Get occurance count of words.
        Returns a dictionary of words:count

        feat: list of integers
        """
        res = {self.features[i]: feat[i] for i in range(len(self.features))}
        
        return {x:y for x, y in res.items() if y!=0}
                    
    def classify(self, feature):
        """
        Check if the response is flagged or unflagged by comparing polarity.
        If negative feature exists, flagged.
        If number of neutral feature greater than positive, flagged.

        feature: list of integers
        """
        pos_count = 0
        neu_count = 0
        neg_count = 0
        feat_occ = self.get_occurance(feature)
            
        for key, val in feat_occ.items():
            if key in self.pos_feat:
                pos_count += val
            elif key in self.neu_feat:
                neu_count += val
            elif key in self.neg_feat:
                neg_count += val
        
        if neg_count > 0 or neu_count >= pos_count:
            return 1
        else:
            return 0
            
    def train(self, data):
        """
        Store features then train model to get word polarity.

        data: dataframe
        """
        self.X, self.y, self.features = extract_info(data, 100)
        self.get_feat_polarity(self.X, self.y)
        
    def predict(self, corpus):
        """
        Predict if the corpus lines are flagged or unflagged.

        corpus: ndarray of strings
        """
        result = []
        test_vecotrs = get_vector(corpus, self.features)
        
        for i in range(len(test_vecotrs)):
            result.append(self.classify(test_vecotrs[i]))
            
        return result
    
def lemm_response(response, lower_case=True, lemm=True, stop_words=True):
    """
    Clean response and lemmatize words.

    response: String
    lower_case: boolean value indicate if the response should be lower-cased
    lemm: boolean value indicate if the response should be lemmatized
    stop_words: boolean value indicate if stop-words should be removed from reponse
    """
    if lower_case:
        response = re.sub('[^a-zA-Z]',' ', response).lower()
            
    words = word_tokenize(response)
        
    if stop_words:
        sw = stopwords.words('english')
        words = [w for w in words if w not in sw]
    if lemm:
        lemmatizer = WordNetLemmatizer()
        words = (lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(w, 'n'), pos='v'), pos='a') for w in words)
            
    return ' '.join(words)

def extract_info(data, max_features):
    """
    Extract feature vector, labels, and features from data for training.

    data: dataframe
    max_features: Integer
    """
    simple_txt = []

    for t in data['response_text']:
        simple_txt.append(lemm_response(t))
            
    vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(simple_txt).toarray()
    y = data['class']
    features = vectorizer.get_feature_names_out()
    
    return X, y, features

def get_vector(corpus, features):
    """
    Get vectorized form of features (size: max_feature).

    corpus: ndarray of String
    features: list of String
    """
    test_txt = []
    
    for t in corpus:
        test_txt.append(lemm_response(t))
        
    m = len(test_txt)
    test_vectors = np.zeros((m, len(features)))

    # Get feature occurance base on pre-trained features
    for i in range(m):
        for w in test_txt[i].split():
            if w in features:
                test_vectors[i][np.where(features == w)] += 1
                
    return test_vectors