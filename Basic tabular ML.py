import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error, \
    mean_absolute_percentage_error, median_absolute_error

# Load dataset
file_path = "C:/Users/Rober/OneDrive/Documents/Research campus pc/Research/DRP/MERRA-2/Toolik Lake 1999_2020_test.csv"
df = pd.read_csv(file_path)

# Define input and target variables
input_vars = ['LWLAND', 'SNODP', 'TWLAND']
target_var = 'TSOIL1'

# Ensure selected columns exist in DataFrame
for col in input_vars + [target_var]:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataset.")

# Handle missing values (drop rows with NaN)
df = df.dropna(subset=input_vars + [target_var])

# Prepare data
X = df[input_vars]
y = df[target_var]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Models
models = {
    "Multiple Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
    "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")

    # Use scaled features for all models
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Performance Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    max_err_val = max_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        "MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape,
        "MedAE": medae, "Max Error": max_err_val, "R2 Score": r2
    }

    print(f"{name} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}, "
          f"MedAE: {medae:.4f}, Max Error: {max_err_val:.4f}, R2 Score: {r2:.4f}\n")

# Display results
results_df = pd.DataFrame(results).T
print("\nModel Performance:")
print(results_df)

# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x=results_df.index, y=results_df["R2 Score"], palette="coolwarm")
plt.title("Model Performance Comparison")
plt.ylabel("R2 Score")
plt.xticks(rotation=45)
plt.show()

print("Training completed!")
