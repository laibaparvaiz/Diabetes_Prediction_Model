import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer 

#step 1: loads data and assigns coloumns

#the dataset is hosted by a repository
DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

#define column names as the dataset does not come with headers
column_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

try:
    df = pd.read_csv(DATA_URL, header=None, names=column_names)
except Exception as e:
    print(f"Error loading data: {e}")
    print("Please check your internet connection or the data URL.")
    exit()

print("--- Data Loading and Initial Inspection ---")
print(df.head())
print("\nData Info:")
df.info()


#step 2: Data Cleaning, replaces zeros with missing values

#in this dataset, values like 0 for Glucose, BloodPressure, SkinThickness,
#insulin, and BMI are physiologically impossible and represent missing data
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)

#imputation is Replacing missing(NaN) values with the mean of their respective column.
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df[cols_to_replace] = imputer.fit_transform(df[cols_to_replace])

print("\n--- Data Cleaning Complete (Zero values replaced by mean) ---")
print(df.describe().loc['mean', cols_to_replace].round(2))


#step 3: prepare data for modeling, splitting into training, testing sets

#define features(x) and target(y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

#split the data into 80% training and 20% testing sets
#we use 'stratify=y' to ensure both the training and test sets have the same proportion
#of diabetes(1) and non-diabetes(0) cases. This is crucial for classification!
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} | Test set size: {X_test.shape[0]}")


#step 4: creates logistic regression model, test and train
model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)

print("\n--- Training Logistic Regression Model... ---")
model.fit(X_train, y_train)
print("Training Complete.")


#Step 5: how well the model is performing
y_pred = model.predict(X_test)

#accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

#confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

#classification report
class_report = classification_report(y_test, y_pred)
print("\nClassification Report (Focus on F1-Score):")
print(class_report)
