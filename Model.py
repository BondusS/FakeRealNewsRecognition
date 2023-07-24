import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

# Загрузка и подготовка данных
X_data = []
Y_data = []
dataset_fake = pd.read_csv("Fake.csv", skip_blank_lines=True)
for row in dataset_fake.itertuples():
    article = row[1] + ' ' + row[2]
    X_data.append(article.lower())
    Y_data.append(np.array([1, 0]))
del dataset_fake
dataset_true = pd.read_csv("True.csv", skip_blank_lines=True)
for row in dataset_true.itertuples():
    article = row[1] + ' ' + row[2]
    X_data.append(article.lower())
    Y_data.append(np.array([0, 1]))
del dataset_true
X_data = np.array(X_data)
Y_data = np.array(Y_data)
x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data)

# Инициализация векторизации текста
word2vector = tf.keras.layers.TextVectorization()
word2vector.adapt(X_data)

# Построение модели
model = tf.keras.Sequential([word2vector,
                             tf.keras.layers.Embedding(input_dim=len(word2vector.get_vocabulary()),
                                                       output_dim=2,
                                                       mask_zero=True),
                             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=2)),
                             tf.keras.layers.Dense(units=2, activation='softmax')])
model.summary()

# Компиляция и обучение модели
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x=x_train,
                    y=y_train,
                    validation_data=(x_test, y_test),
                    epochs=5)

# Показатели модели на тестовом наборе
loss, acc = model.evaluate(x=x_test,
                           y=y_test)
print('Test loss:', loss, ', test accuracy:', acc)

# Графики
def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)
plt.show()
