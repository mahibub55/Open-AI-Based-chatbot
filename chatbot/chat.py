import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
import re -

# Download necessary NLTK resources if not already present
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# # Text preprocessing functions
def preprocess(text, remove_stopwords=True):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text.lower())
    if remove_stopwords:
        tokens = [token for token in tokens if token not in stopwords.words('english')]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)

class QASystem:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath, escapechar='\\')
        self.questions = self.df['Questions'].tolist()
        self.answers = self.df['Answers'].tolist()
        self.vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
        self.X = self.vectorizer.fit_transform([preprocess(q) for q in self.questions])

    def get_response(self, text):
        processed_text = preprocess(text)
        vectorized_text = self.vectorizer.transform([processed_text])
        similarities = cosine_similarity(vectorized_text, self.X)
        max_similarity = similarities.max()

        if max_similarity > 0.6:
            high_similarity_indices = similarities[0] > 0.6
            high_similarity_questions = [q for q, s in zip(self.questions, high_similarity_indices) if s]
            target_answers = [self.answers[self.questions.index(q)] for q in high_similarity_questions]

            Z = self.vectorizer.transform([preprocess(q) for q in high_similarity_questions])
            final_similarities = cosine_similarity(vectorized_text, Z)
            closest_index = final_similarities.argmax()

            return target_answers[closest_index]
        else:
            return "I can't answer this question."

def get_chatbot_response(input):
    qa_system = QASystem('test.csv')
    return qa_system.get_response(input)
