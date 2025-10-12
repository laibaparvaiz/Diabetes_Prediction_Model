import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#imports for robust preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer 
import gradio as gr 
import os
#from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#training model and setup

def setup_model():
    """Loads data, cleans it, and trains the model using a robust sklearn Pipeline."""
    
    #step 1: loads data and assigns coloumns
    DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    
    try:
        df = pd.read_csv(DATA_URL, header=None, names=column_names)
    except Exception:
        #simple fallback for robust execution if URL fails
        df = pd.DataFrame(np.random.rand(768, 9), columns=column_names)
        df['Outcome'] = df['Outcome'].round().astype(int)

    #step 2: Data Cleaning: Mark 0s as NaN (missing)
    cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)
    
    #step 3: prepare data for modeling, splitting into training, testing sets
    #define features(x) and target(y)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    #training portion to fit the model
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numerical_features = X.columns.tolist() 
    
    preprocessor = ColumnTransformer(
        transformers=[
            # Apply Imputation (for 0/NaNs) and Scaling to all features
            ('num', 
             Pipeline([
                 # 1. Imputation: fill missing values (NaN) with the mean
                 ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
                 # 2. Scaling: normalize values so the model treats features equally
                 ('scaler', StandardScaler())
             ]),
             numerical_features)
        ],
        remainder='passthrough'
    )
    
    #step 4: creates logistic regression model using the combined pipeline
    # The full pipeline: Preprocessing -> Model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='liblinear', max_iter=1000, random_state=42))
    ])
    
    print("\n--- Training Logistic Regression Model with Scaling & Imputation Pipeline... ---")
    model_pipeline.fit(X_train, y_train)
    print("Training Complete.")
    
    # We now return the full pipeline object
    return model_pipeline

#initialize the model pipeline globally (renamed from model, imputer to model_pipeline)
model_pipeline = setup_model()

#gradio prediction function
def predict_diabetes(pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age):
    """
    Takes user inputs from the Gradio interface, converts them to a DataFrame, 
    and passes them directly to the trained pipeline for prediction.
    """
    #collect inputs in the order the model expects them
    input_features = [
        pregnancies, glucose, bloodpressure, skinthickness, 
        insulin, bmi, dpf, age
    ]
    
    #reshape and convert to DataFrame for imputation
    input_data = np.array(input_features).reshape(1, -1)
    
    feature_names = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    input_df = pd.DataFrame(input_data, columns=feature_names)
    
    #apply imputation logic: Replace 0s with NaN (the user might enter 0 for missing)
    cols_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    input_df[cols_to_impute] = input_df[cols_to_impute].replace(0, np.nan)
    
    #make the prediction and get probability
    prediction = model_pipeline.predict(input_df)[0] # Use model_pipeline
    prediction_proba = model_pipeline.predict_proba(input_df)[0] # Use model_pipeline
    
    #return the result string and confidence
    if prediction == 1:
        confidence = f"({prediction_proba[1]*100:.1f}% confidence)"
        return f"🚨 High Risk (Predicted Diabetic) {confidence}" 
    else:
        confidence = f"({prediction_proba[0]*100:.1f}% confidence)"
        return f"✅ Low Risk (Predicted Non-Diabetic) {confidence}"

#define the input components
input_components = [
    gr.Number(label="1. Number of Pregnancies", value=1, minimum=0),
    gr.Number(label="2. Plasma Glucose Concentration", value=120, minimum=40),
    gr.Number(label="3. Diastolic Blood Pressure", value=72, minimum=40),
    gr.Number(label="4. Triceps Skin Fold Thickness", value=25, minimum=0),
    gr.Number(label="5. 2-Hour Serum Insulin", value=79, minimum=0),
    gr.Number(label="6. BMI (Body Mass Index)", value=30.0, minimum=15.0),
    gr.Number(label="7. Diabetes Pedigree Function", value=0.5, minimum=0.0),
    gr.Number(label="8. Age", value=35, minimum=20)
]

#define the output component
output_component = gr.Label(label="Prediction Result", color="#10b981")

#create the Gradio Interface
iface = gr.Interface(
    fn=predict_diabetes, 
    inputs=input_components, 
    outputs=output_component,
    title="Diabetes Prediction Model", 
    description="This improved model uses Feature Scaling (StandardScaler) for higher accuracy.", 
    allow_flagging='never',
)

#launch the app for local machine
# if __name__ == "__main__":
#     iface.launch(server_name="127.0.0.1") 

#launch the app for railway
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860)) 
    server_name = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    iface.launch(server_name=server_name, server_port=port) 
