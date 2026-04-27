import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 1: Load dataset
data = pd.read_csv('house_price_regression_dataset.csv')

# Step 2: Print columns (VERY IMPORTANT)
print("Columns in dataset:", data.columns)

# Step 3: Set target column (TEMPORARY)
target_column = 'House_Price'   # we will fix if wrong

# Step 4: Split features and target
X = data.drop(target_column, axis=1)
y = data[target_column]

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 6: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Save model
pickle.dump(model, open('model.pkl', 'wb'))

print("✅ Model trained & saved as model.pkl")
