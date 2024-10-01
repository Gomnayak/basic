import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


# Load the dataset
file_path=r"C:\Users\gomat\Downloads\Telco-Customer-Churn.csv"

# Converting the dataset into Dataframe
df=pd.read_csv(file_path)

# using to_string to show entire DataFrame
(df.to_string())

# Display the first few rows of the dataset
(df.head(1000))

# display basic information about the dataset
(df.info())

# Display summary statistics of numerical columns
(df.describe())

# Check for missing values
(df.isnull().sum())

# Check the distribution of the target variable 'Churn'
print(df['Churn'].value_counts(normalize=True))

# Check for non-numeric values in "TotalCharges"
non_numeric_total_charges=df[df["TotalCharges"].str.strip()==""]
print(f"Number of non-numeric entries in \"TotalCharges\" : {len(non_numeric_total_charges)}")

# Replace spaces with NaN(missing values)
df ['TotalCharges']= df["TotalCharges"].replace(" ",pd.NA)

# Print the updated "TotalCharges" column
print(df["TotalCharges"])

# Convert 'TotalCharges' to numeric,coercing errors
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"], errors = "coerce")
print(df["TotalCharges"])

# Handle missing values by filling them with the median value of 'TotalCharges'
df["TotalCharges"]=df["TotalCharges"].fillna(df["TotalCharges"].median())
print(df["TotalCharges"])


# Encode categorical variables


label_encoders = {}
for column in df.select_dtypes(include=["object"]).columns:
    if column != "customerID":
       le = LabelEncoder()
       df[column] = le.fit_transform(df[column])
       label_encoders[column] = le



# Drop the 'customerID' column as it is not useful for prediction
print(df.drop(columns=['customerID'], inplace=True))

# Scale numerical features


scaler = StandardScaler()
numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
print(df[numerical_columns])
print(df.head(1000))


# Train-test-split
# Split the data into training and testing sets



# Define the target variable and features
X = df.drop(columns=['Churn'])
y = df['Churn']
print(X)
print(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(X_train, X_test, y_train, y_test)

print("X_train:\n", X_train)
print("\nX_test:\n", X_test)
print("\ny_train:\n", y_train)
print("\ny_test:\n", y_test)

# Model Building

# use a Logistic Regression model for prediction

# Initialize and train the model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
print(model.fit(X_train, y_train))

# Model Evaluation

# the model using accuracy, confusion matrix, and classification report

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Visualization
# Visualize the results using confusion matrix and ROC curve,auc

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='yellow', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()






