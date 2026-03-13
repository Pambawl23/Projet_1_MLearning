import dataset as dt
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam


def train_model(epochs: int = 300):
    donnees = dt.X.to_numpy(dtype="float32")
    reponses, classes = dt.y.iloc[:, 0].factorize(sort=True)
    n_classes = len(classes)

    model = Sequential([
        Input(shape=(31,)),
        Dense(16, activation="relu"),
        Dense(n_classes, activation="softmax")
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=0.01),
        metrics=["sparse_categorical_accuracy"],
    )

    history = model.fit(donnees, reponses, epochs=epochs, verbose=0)
    scores = model.evaluate(donnees, reponses, verbose=0)

    return {
        "model": model,
        "classes": classes,
        "accuracy": float(scores[1]),
        "loss": float(scores[0]),
        "history": history.history,
    }


if __name__ == "__main__":
    result = train_model()
    print("\nExactitude (accuracy) : %.2f%%" % (result["accuracy"] * 100))
