import unidecode
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from underthesea.word_tokenize import tokenize


class WordSetVectorizer(TransformerMixin):
    def __init__(self):
        self.words = {}

    def fit_transform(self, X):
        result = []
        for s in X:
            tokens = s.split()
            s_result = []
            for token in tokens:
                if token not in self.words:
                    self.words[token] = len(self.words)
                s_result.append(self.words[token])
            result.append(s_result)
        return result



WINDOW_SIZE = 4
count = 0
X = []
y = []
for line in open("tmp/vietnamese_tone_prediction/train.txt"):
    count += 1
    text = line.strip().lower()
    tokens = tokenize(text)
    tokens = ["bos"] * WINDOW_SIZE + tokens + ["eos"] * WINDOW_SIZE
    tokens_remove_tone = [unidecode.unidecode(token) for token in tokens]
    for i, token in enumerate(tokens[WINDOW_SIZE:-WINDOW_SIZE]):
        j = i + WINDOW_SIZE
        yi = token
        Xi = " ".join(tokens_remove_tone[j - WINDOW_SIZE:j + WINDOW_SIZE])
        X.append(Xi)
        y.append(yi)
    if count > 100000:
        break

x_encoder = WordSetVectorizer()
y_encoder = LabelEncoder()
X_train_dev = x_encoder.fit_transform(X)
y_train_dev = y_encoder.fit_transform(y)

X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=0.2, random_state=1024)
print(f"Train: {len(X_train)}, Dev: {len(X_dev)}")
clf = LinearSVC()
clf.fit(X_train, y_train)
score = clf.score(X_dev, y_dev)
print(score)
