import pandas as pd
import numpy as np

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



