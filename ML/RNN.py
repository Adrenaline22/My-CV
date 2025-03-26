import keras
import pymysql.cursors
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
import warnings
import tensorflow as tf

from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTMe

warnings.filterwarnings('ignore')

connection1 = pymysql.connect(host='localhost',
                              user='root',
                              password='root',
                              charset='utf8mb4',
                              cursorclass=pymysql.cursors.DictCursor)
print('Подлючение к базе данных прошло успешно!')
with connection1.cursor() as cursor:
    sql1 = "use forecast1"
    cursor.execute(sql1)
    connection1.commit()
    cursor.execute("select * from complete_no_null_water5")
    lst = cursor.fetchall()
    df = pd.DataFrame(lst)
    df['predict 1'] = df['Уровень воды'].shift(periods=1, fill_value=0)
    df['predict 2'] = df['Уровень воды'].shift(periods=2, fill_value=0)
    df['predict 3'] = df['Уровень воды'].shift(periods=3, fill_value=0)
    df = df[['Код поста', 'Уровень воды', 'Температура воздуха', 'Атмосферное давление', "Толщина снежного покрова",
             'Скорость ветра', 'Количество осадков', 'predict 1', 'predict 2', 'predict 3']]
    df = df.fillna(0)
    connection1.close()
num = str(input("Введите код поста:"))
df = df[df['Код поста'] == num]

X = df[
    ['Температура воздуха', 'Скорость ветра', 'Количество осадков', 'Толщина снежного покрова', 'Атмосферное давление',
     'predict 1', 'predict 2', 'predict 3']].astype(float)
y = df['Уровень воды'].astype(float)


scaler = MinMaxScaler()
scaler.fit(X)
New_Data = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(New_Data, y, test_size=0.2)

input_dim = 4
output_dim = 1

model = keras.models.Sequential()
model.add(keras.Input(shape=(8,)))  # Входной слой с 4 признаками
model.add(layers.Dense(15, activation='relu', use_bias=True))  # Пример скрытого слоя
model.add(tf.keras.layers.Dense(1))


model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.997,
                                                                            ema_momentum=0.99,
                                                                            ema_overwrite_frequency=None,
                                                                            jit_compile=True,
                                                                            name="Adam", ), metrics=["accuracy"])

# #
# # keras.optimizers.Adam(lr=0.0001)
# tf.keras.optimizers.
batch_size = 10
epochs = 250

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.25, verbose=2)
#
# model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
predictions = model.predict(X_test)
for i in range(len(predictions)):
    print(f"Predicted: {predictions[i][0]}, Actual: {y_test.iloc[i]}")