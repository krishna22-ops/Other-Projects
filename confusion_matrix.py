import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

#Trained model and data from main.py
from Main import svm, vectorizer, X_test, y_test

# Create results folder
os.makedirs('results', exist_ok=True)

# -------------------------------
# CONFUSION MATRIX ONLY
# -------------------------------
y_pred = svm.predict(X_test)

# Create confusion matrix plot
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title('Confusion Matrix - SVM Sentiment Analysis')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("Confusion matrix saved to results/confusion_matrix.png")