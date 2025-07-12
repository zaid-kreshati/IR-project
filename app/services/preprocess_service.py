import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textacy import preprocessing
from num2words import num2words
from nltk.stem import WordNetLemmatizer

# Custom imports
from app.utility.chat_words import chat_words
from app.utility.abbrevations import common_abbreviations
from app.utility.contractions import contractions
from app.utility.punct_mapping import punct, punct_mapping

# Initializations
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))


def preprocess_text(text: str) -> str:
    # Lowercase
    text = text.lower()

    # Basic cleaning
    text = preprocessing.remove.html_tags(text)
    text = preprocessing.remove.brackets(text)
    text = preprocessing.remove.accents(text)
    text = preprocessing.normalize.unicode(text)

    # Replace certain punctuation with space
    text = re.sub(r'[-./]', ' ', text)  # ensure dash/dot/slash are separators
    text = re.sub(r'[^\w\s]', '', text)  # remove other special characters

    # Normalize contractions, abbreviations, chat words
    text = normalize_contractions(text)
    text = normalize_abbreviations(text)
    text = chat_conversion(text)

    # Replace numbers with words
    text = replace_numbers_with_words(text)


    # Remove stopwords
    text = remove_stopwords(text)

    # Remove one-letter words
    text = remove_one_letter_words(text)

    # Optional stemming (you can replace with lem_words if needed)
    # text = stem_words(text)

    tokens = word_tokenize(text)

    return tokens


def preprocess_text_embeddings(text):
    text = text.lower()

    # text = clean_special_chars(text,punct,punct_mapping)
    text = preprocessing.remove.html_tags(text)
    text = preprocessing.remove.punctuation(text)
    text = preprocessing.remove.brackets(text)

    text = preprocessing.replace.emojis(text)
    text = preprocessing.replace.urls(text)
    text = preprocessing.remove.accents(text)
    return text

def preprocess_bm25(text):
    text = text.lower()

    # text = clean_special_chars(text,punct,punct_mapping)
    text = preprocessing.remove.html_tags(text)
    text = preprocessing.remove.punctuation(text)
    text = preprocessing.remove.brackets(text)

    text = preprocessing.replace.emojis(text)
    text = preprocessing.replace.urls(text)
    text = preprocessing.remove.accents(text)
    return text


# ========== Utility Functions ==========

def remove_stopwords(text: str) -> str:
    return " ".join([word for word in text.split() if word not in STOPWORDS])


def remove_one_letter_words(text: str) -> str:
    return " ".join([word for word in text.split() if len(word) > 1])


def stem_words(text: str) -> str:
    return " ".join([stemmer.stem(word) for word in text.split()])


def lem_words(text: str) -> str:
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


def replace_numbers_with_words(text: str) -> str:
    return re.sub(r'\b\d+\b', lambda m: num2words(int(m.group())), text)


def normalize_contractions(text: str) -> str:
    return " ".join([contractions.get(word, word) for word in text.split()])


def normalize_abbreviations(text: str) -> str:
    return " ".join([common_abbreviations.get(word.upper(), word) for word in text.split()])


def chat_conversion(text: str) -> str:
    return " ".join([chat_words.get(word.upper(), word) for word in text.split()])


