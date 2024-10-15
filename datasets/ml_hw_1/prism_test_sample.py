import numpy as np
from prism import PRISMClassifier

X_sample = np.loadtxt("prism_X_sample.csv", dtype=str, delimiter=",")
y_sample = np.loadtxt("prism_y_sample.csv", dtype=str, delimiter=",")

# Test your PRISMClassifier
prism_classifier = PRISMClassifier()
prism_classifier.fit(X_train, y_train)
predictions = prism_classifier.predict(X_sample)

print("Predictions:", predictions)
print("Expected:   ", y_sample.tolist())
