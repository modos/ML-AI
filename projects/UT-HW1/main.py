import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mushroom_orig_data = pd.read_csv('dataset/mushrooms.csv')


data = mushroom_orig_data.copy()
data = data.dropna()


# data['stem-height'] = pd.cut(data['stem-height'].astype(int), bins=[0,1,2, float('inf')], labels=[1,2,3])
# data['season'] = data['season'].rank(method='dense').astype(int)
# data['cap-diameter'] = pd.cut(data['cap-diameter'].astype(int), bins=[0,500, 800, 1000, float('inf')], labels=[1,2,3, 4])
# data['stem-width'] = pd.cut(data['stem-width'].astype(int), bins=[0,1000,2000, 2500, 3000, float('inf')], labels=[1,2,3,4, 5])
X = data.drop(columns='class')
y = data['class']

class PrismClassifier:
    def __init__(self, min_support=0.1, min_accuracy=0.8):
        self.min_support = min_support
        self.min_accuracy = min_accuracy
        self.final_rules = []

    def fit(self, X, y):
        X_copy = X.reset_index(drop=True)
        y_copy = y.reset_index(drop=True)

        for class_label in y_copy.unique():
            perfect_rule_found = False
            while not perfect_rule_found:
                best_rule = None
                best_accuracy = 0

                for feature in X_copy.columns:
                    for value in X_copy[feature].unique():
                        rule = (feature, value)
                        covered_indices = X_copy[X_copy[feature] == value].index
                        support = len(covered_indices) / len(X_copy)

                        covered_y = y_copy[X_copy.index.isin(covered_indices)]

                        acc = covered_y.value_counts(normalize=True).get(class_label, 0)

                        if support >= self.min_support and acc > best_accuracy:
                            best_accuracy = acc
                            best_rule = rule

                if best_rule:
                    self.final_rules.append((best_rule, class_label))
                    X_copy = X_copy[X_copy[best_rule[0]] != best_rule[1]]
                    y_copy = y_copy[X_copy.index]
                    X_copy = X_copy.reset_index(drop=True)
                    y_copy = y_copy.reset_index(drop=True)


                    if len(X_copy) == 0:
                        perfect_rule_found = True
                else:
                    perfect_rule_found = True

    def predict(self, X):
        pred = []
        for _, instance in X.iterrows():
            predicted_class = self._classify_instance(instance)
            pred.append(predicted_class)
        return np.array(pred)

    def _classify_instance(self, instance):
        for rule, class_label in self.final_rules:
            if isinstance(instance[rule[0]], str):
                if instance[rule[0]] == rule[1]:
                    return class_label
            else:
                if instance[rule[0]] == rule[1]:
                    return class_label

        return 'unknown'

    def rules(self):
        class_1_rules = []
        class_0_rules = []
        for rule, class_label in self.final_rules:
            conditions = []
            for f in rule:
                conditions.append(f"{f}={rule[1]}")
            rule_string = " AND ".join(conditions)

            if class_label == 'p':
                class_1_rules.append(rule_string)
            else:
                class_0_rules.append(rule_string)
        print("IF", " AND " .join(class_1_rules) + " THEN class=p")

        print("IF", " AND " .join(class_0_rules) + " THEN class=e")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = PrismClassifier(min_support=0.1, min_accuracy=0.8)
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)



Prism = PrismClassifier()
Prism.fit(X_train, y_train)
Prism.rules()
accuracy = accuracy_score(y_test, predictions)
print(accuracy)



