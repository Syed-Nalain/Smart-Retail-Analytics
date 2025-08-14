from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def train_decision_tree(rfm_df):
    """
    Train a decision tree classifier on RFM data.
    """
    X = rfm_df[['Recency', 'Frequency', 'Monetary']]
    y = rfm_df['RFM_Score']  # Target is total RFM score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nDecision Tree Classifier Report:")
    print(classification_report(y_test, y_pred))

    return model
