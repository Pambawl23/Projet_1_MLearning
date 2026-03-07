import dataset as dt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

donnees = dt.X.to_numpy(dtype="float32")
reponses, classes = dt.y.iloc[:, 0].factorize(sort=True)
n_classes = len(classes)
#training_data = np.array([ [0, 0],[0, 1], [1, 0],[1, 1]], 'float32')
#target_data = np.array([[0],[1], [1], [0]], 'float32')
model = Sequential()
model.add(Dense(16, input_dim=31, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(learning_rate=0.01),  # 0.001 → 0.01
    metrics=['sparse_categorical_accuracy']
)
model.fit(donnees, reponses, epochs=500)  # 100 → 500
scores = model.evaluate(donnees, reponses)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print(model.predict(donnees).argmax(axis=1).reshape(-1, 1))
model.summary()