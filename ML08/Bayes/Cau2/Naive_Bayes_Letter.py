import numpy as np

class CategoricalNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha                      # Laplace smoothing
        self.class_prob = {}                    # P(class)
        self.feature_prob = {}                  # P(feature|class)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        # Tính P(class)
        for c in self.classes:
            count = np.sum(y == c)
            self.class_prob[c] = (count + self.alpha) / (n_samples + self.alpha * len(self.classes))

        # Tính P(feature_i = value | class) cho từng feature
        for c in self.classes:
            X_c = X[y == c]
            self.feature_prob[c] = []
            for i in range(n_features):
                counts = np.bincount(X_c[:, i], minlength=16) + self.alpha
                probs = counts / counts.sum()
                self.feature_prob[c].append(probs)

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                log_prob = np.log(self.class_prob[c])
                for i in range(len(x)):
                    log_prob += np.log(self.feature_prob[c][i][x[i]])
                posteriors.append(log_prob)
            predictions.append(self.classes[np.argmax(posteriors)])
        return predictions

    def score(self, X, y):
        pred = self.predict(X)
        return np.mean(pred == y)