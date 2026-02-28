import dataset as dt
from keras.models import Sequential
from keras.layers import Dense

donnees = dt.X
reponses = dt.y
#training_data = np.array([ [0, 0],[0, 1], [1, 0],[1, 1]], 'float32')
#target_data = np.array([[0],[1], [1], [0]], 'float32')
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['binary_accuracy'])
model.fit(donnees, reponses, epochs=1000)
scores = model.evaluate(donnees, reponses)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print (model.predict(donnees).round())
model.summary()