import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

def main():
    # 1. Load MNIST dataset (60,000 train, 10,000 test)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 2. Normalize pixel values from [0, 255] to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # 3. Add channel dimension: (batch, 28, 28, 1)
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # 4. Build a simple CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")  # 10 classes: 0–9
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 5. Train the model
    model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1
    )

    # 6. Evaluate on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    # 7. Save the trained model
    model.save("mnist_cnn.h5")
    print("Model saved as mnist_cnn.h5")

if __name__ == "__main__":
    main()
