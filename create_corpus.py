import unidecode
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from underthesea.word_tokenize import tokenize

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
        Xi = tokens_remove_tone[j-WINDOW_SIZE:j+WINDOW_SIZE]
        X.append(Xi)
        y.append(yi)
    if count > 100:
        break
x_encoder = LabelEncoder()
y_encoder = LabelEncoder()
y_train = y_encoder.fit_transform(y)
X_train = x_encoder.fit_transform(X)
clf = LinearSVC()
clf.fit(X, y_train)
print(0)