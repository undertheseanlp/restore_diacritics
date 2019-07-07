from sklearn.feature_extraction.text import HashingVectorizer

X = ["a b c", "a c", "b c"]
v = HashingVectorizer()
x = v.fit_transform(X)
print(0)