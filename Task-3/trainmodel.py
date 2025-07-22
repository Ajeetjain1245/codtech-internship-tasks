# train_model.py

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save model
pickle.dump(model, open('Task-3/model.pkl', 'wb'))

print("Model saved!")
