import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, max_error, r2_score
import matplotlib.pyplot as plt

# Load dataset (Replace with actual file path)
file_path = "C:/Users/Rober/OneDrive/Documents/Research campus pc/Research/DRP/Papers written by group/air temp forecasting/Utqiagvik_air_temps_1979_2015.csv"
df = pd.read_csv(file_path)

# Function to clean and format the DATE column and extract features
def process_date_column(df, date_column='DATE'):
    """Processes the DATE column: formats it and extracts day, month, and year."""
    try:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce', dayfirst=True)
        df['Day'] = df[date_column].dt.day
        df['Month'] = df[date_column].dt.month
        df['Year'] = df[date_column].dt.year
        df.drop(columns=[date_column], inplace=True)  # Drop original date column
    except Exception as e:
        print(f"Error processing date column: {e}")
    return df


df = process_date_column(df, date_column='DATE')

# Handle missing or infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
if df.isnull().values.any():
    print("\nWarning: Missing values detected. Dropping missing values.")
    df.dropna(inplace=True)

# Define input and target variables
input_vars = ['Mean Temperature (degF)', 'Month']
target_var = 'Snow Depth (in)'

# Scale input features (Date values are also scaled for consistency)
scaler = MinMaxScaler()
df[input_vars] = scaler.fit_transform(df[input_vars])


# Convert dataframe to sequences for LSTM/GRU
def create_sequences(data, target, seq_length=10):
    """Creates sequences of length seq_length for LSTM/GRU training."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i + seq_length].values)
        y.append(target.iloc[i + seq_length])
    return np.array(X), np.array(y)


# Prepare data
seq_length = 10  # Number of time steps per sample
X, y = create_sequences(df[input_vars], df[target_var], seq_length)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)


# Define LSTM model
def create_lstm_model(input_shape, units=50):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        LSTM(units, return_sequences=False),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse')
    return model


# Define GRU model
def create_gru_model(input_shape, units=50):
    model = Sequential([
        GRU(units, return_sequences=True, input_shape=input_shape),
        GRU(units, return_sequences=False),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

# Define Transformer model
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Define a Transformer model
def create_transformer_model(input_shape, embed_dim=64, num_heads=4, ff_dim=128):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(embed_dim)(inputs)  # Embedding layer
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='linear')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


# Get input shape for models
input_shape = (seq_length, len(input_vars))

# Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train LSTM model
lstm_model = create_lstm_model(input_shape)
print("\nTraining LSTM Model...")
history_lstm = lstm_model.fit(X_train, y_train, epochs=50, batch_size=32,
                              validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

# Evaluate LSTM model
lstm_loss = lstm_model.evaluate(X_test, y_test)
y_pred_lstm = lstm_model.predict(X_test)

# Train GRU model
gru_model = create_gru_model(input_shape)
print("\nTraining GRU Model...")
history_gru = gru_model.fit(X_train, y_train, epochs=50, batch_size=32,
                            validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

# Evaluate GRU model
gru_loss = gru_model.evaluate(X_test, y_test)
y_pred_gru = gru_model.predict(X_test)


# Train Transformer model
transformer_model = create_transformer_model(input_shape)
print("\nTraining Transformer Model...")
history_transformer = transformer_model.fit(X_train, y_train, epochs=50, batch_size=32,
                                            validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

# Evaluate Transformer model
transformer_loss = transformer_model.evaluate(X_test, y_test)
y_pred_transformer = transformer_model.predict(X_test)

# Function to calculate and print evaluation metrics
def print_metrics(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    max_err = max_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name} Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Maximum Error: {max_err:.4f}")
    print(f"  R² Score: {r2:.4f}")

    return rmse, max_err, r2


# Print metrics for LSTM
rmse_lstm, max_err_lstm, r2_lstm = print_metrics(y_test, y_pred_lstm, "LSTM")

# Print metrics for GRU
rmse_gru, max_err_gru, r2_gru = print_metrics(y_test, y_pred_gru, "GRU")

# Print metrics for Transformer
rmse_transformer, max_err_transformer, r2_transformer = print_metrics(y_test, y_pred_transformer, "Transformer")


# Plot loss curves - LSTM
plt.figure(figsize=(10, 5))
plt.plot(history_lstm.history['loss'], label='LSTM Train Loss', linestyle='dashed')
plt.plot(history_lstm.history['val_loss'], label='LSTM Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.show()

# Plot loss curves - GRU
plt.figure(figsize=(10, 5))
plt.plot(history_gru.history['loss'], label='GRU Train Loss', linestyle='dashed')
plt.plot(history_gru.history['val_loss'], label='GRU Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.show()

# Plot loss curves - Transformer
plt.figure(figsize=(10, 5))
plt.plot(history_transformer.history['loss'], label='Transformer Train Loss', linestyle='dashed')
plt.plot(history_transformer.history['val_loss'], label='Transformer Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.show()

# Plot actual vs predicted values - LSTM
plt.figure(figsize=(12, 5))
plt.plot(y_test, label="Actual", marker='o', linestyle='dashed', alpha=0.7)
plt.plot(y_pred_lstm, label="LSTM Predicted", alpha=0.7)
plt.xlabel("Sample Index")
plt.ylabel("TSOIL1")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.show()

# Plot actual vs predicted values - GRU
plt.figure(figsize=(12, 5))
plt.plot(y_test, label="Actual", marker='o', linestyle='dashed', alpha=0.7)
plt.plot(y_pred_gru, label="GRU Predicted", alpha=0.7)
plt.xlabel("Sample Index")
plt.ylabel("TSOIL1")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.show()

# Plot actual vs predicted values - Transformer
plt.figure(figsize=(12, 5))
plt.plot(y_test, label="Actual", marker='o', linestyle='dashed', alpha=0.7)
plt.plot(y_pred_transformer, label="Transformer Predicted", alpha=0.7)
plt.xlabel("Sample Index")
plt.ylabel("TSOIL1")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.show()