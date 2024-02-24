import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

# Sample data
heights = [60, 62, 64, 66, 68, 70, 72, 74]
weights = [110, 120, 130, 140, 150, 160, 170, 180]

# Normalize the input data
scaler = MinMaxScaler()
heights_normalized = scaler.fit_transform(tf.expand_dims(heights, axis=1)).squeeze()
weights = tf.convert_to_tensor(weights, dtype=tf.float32)

# Define a simple model with a single neuron
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[1])
])

# Compile the model
model.compile(optimizer='sgd',  # Stochastic Gradient Descent
              loss='mean_squared_error')  # Mean Squared Error for regression problems

# Train the model
model.fit(heights_normalized, weights, epochs=500, verbose=1)

# Test the model with all input heights
all_heights_normalized = scaler.transform(tf.expand_dims(heights, axis=1)).squeeze()
all_predicted_weights = model.predict(all_heights_normalized)

# Print all predictions
for h, w_pred in zip(heights, all_predicted_weights):
    print(f"Height: {h} inches, Predicted Weight: {w_pred[0]:.2f} lbs")
