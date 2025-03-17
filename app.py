import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# Checking accuracy
from sklearn.metrics import accuracy_score
# Creating a sample dataset
data = {
    'Age': np.random.randint(18, 35, size=30),
    'Gender': np.random.choice(['Male', 'Female'], size=30),
    'Disease': np.random.choice(['Cough', 'Influenza', 'None'], size=30),
    'IQ': np.random.randint(90, 160, size=30),
    'CGPA': np.round(np.random.uniform(2.0, 4.0, size=30), 2),
    'Placement': np.random.choice(['Yes', 'No'], size=30)
}

# Creating DataFrame
df = pd.DataFrame(data)

# Display the dataframe
print("DataFrame:")
print(df)

# Applying preprocessing steps
df_info = df.info()
df_description = df.describe()
df_shape = df.shape
df_duplicated = df.duplicated().sum()



# Assuming df is already defined
# Splitting data into features and target
X = df.drop('Placement', axis=1)
y = df['Placement'].map({'Yes': 1, 'No': 0})  # Converting target to binary values

# Defining numerical and categorical columns
num_cols = ['Age', 'IQ', 'CGPA']
cat_cols = ['Gender', 'Disease']

# Creating column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),  # Scaling numerical features
        ('cat', OneHotEncoder(drop='first'), cat_cols)  # OneHotEncoding categorical features
    ]
)

# Creating the pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fitting the model
pipeline.fit(X_train, y_train)

# Predicting on test data
y_pred = pipeline.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

from sklearn.metrics import confusion_matrix, precision_score, recall_score

cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)


from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
