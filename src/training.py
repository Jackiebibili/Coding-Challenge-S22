from tensorflow import keras
from tensorflow.keras import layers


def train_binary_classification_model(X_train, y_train, X_valid, y_valid, input_dim):
    model = keras.Sequential([
        layers.BatchNormalization(input_shape=[input_dim]), #initial normalization
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),   #help correct overfitting
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),   #help correct overfitting
        layers.Dense(1, activation="sigmoid"),  #binary target; use sigmoid function to channel target to 0 or 1
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",     #make probability curve smooth as loss function prefers
        metrics=["binary_accuracy"],    #threshold probability
    )

    #prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        patience=20,
        min_delta=0.001,
        restore_best_weights=True,
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        batch_size=512,
        epochs=200,
        callbacks=[early_stopping],
    )
    return history
