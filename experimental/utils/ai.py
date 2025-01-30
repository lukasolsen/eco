from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

def mutate_model(model):
    weights = model.get_weights()
    for i in range(len(weights)):
        if np.random.rand() < 0.2:  # Mutation rate
            weights[i] += np.random.normal(0, 0.1, size=weights[i].shape)
    model.set_weights(weights)

def create_model(input_dim, output_dim):
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        Dense(64, activation='relu'),
        Dense(output_dim, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model
