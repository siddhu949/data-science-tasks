# Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Load dataset
df = pd.read_csv('creditcard.csv')

# Preprocessing: Normalize 'Amount' and drop 'Time' column
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df = df.drop(columns=['Time'])

# Features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Train-test split (before applying SMOTE or undersampling)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Choose a sampling method (Random undersampling or SMOTE)
# Option 1: Random Undersampling
undersampler = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

# Option 2: SMOTE (Oversampling the minority class)
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Model Training
# Option 1: Logistic Regression
# clf = LogisticRegression(random_state=42)

# Option 2: Random Forest Classifier
clf = RandomForestClassifier(random_state=42)

# Fit the model (using undersampled data in this case)
clf.fit(X_train_under, y_train_under)

# Predictions on the test set
y_pred = clf.predict(X_test)

# Evaluation: Confusion Matrix and Classification Report
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
