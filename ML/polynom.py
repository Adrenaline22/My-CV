import pandas as pd
import pymysql.cursors
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

con = pymysql.connect(host='localhost',
                      user='root',
                      password='root',
                      charset='utf8mb4',
                      db='forecast1',
                      cursorclass=pymysql.cursors.DictCursor)

print('Подключение к базе данных прошло успешно!')

with con.cursor() as cursor:
    con.commit()
    cursor.execute("select * from complete_no_null_water5")
    lst = cursor.fetchall()
    df = pd.DataFrame(lst)
    df = df[['Код поста', 'Дата - время', 'Уровень воды', 'Температура воздуха', 'Атмосферное давление', 'Скорость ветра', 'Количество осадков']]
    df = df.fillna(0)
    post = str(input("Введите код поста: "))
    df = df[df['Код поста'] == post]
    con.close()

start_date = pd.to_datetime(str(input("Введите дату начала прогнозирования (ГГГГ-ММ-ДД): ")))
end_date = pd.to_datetime(str(input("Введите дату конца прогнозирования (ГГГГ-ММ-ДД): ")))

df['Дата - время'] = pd.to_datetime(df['Дата - время'])
df_filtered = df[(df['Код поста'] == post) & (df['Дата - время'] >= start_date) & (df['Дата - время'] <= end_date)]

if not df_filtered.empty:
    df['predict1'] = df.groupby('Код поста')['Уровень воды'].shift(1)
    df[['predict1']] = df.groupby('Код поста')[['predict1']].fillna(0)
    X = df[['Код поста', 'Температура воздуха', 'Скорость ветра', 'Количество осадков', 'Атмосферное давление', 'predict1']]
    y = df['Уровень воды']
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    check_train = model.score(X_train, y_train)
    check_test = model.score(X_test, y_test)
    print(f"MAE train: {mean_absolute_error(y_train, model.predict(X_train))}")
    print(f"MAE test: {mean_absolute_error(y_test, model.predict(X_test))}")

    # Прогнозы для дат между началом и концом прогнозирования
    date_range = pd.date_range(start=start_date, end=end_date)
    filtered_df = df[(df['Код поста'] == post) & (df['Дата - время'].isin(date_range))]
    X_forecast = filtered_df[['Код поста', 'Температура воздуха', 'Скорость ветра', 'Количество осадков', 'Атмосферное давление', 'predict1']]
    X_forecast_poly = poly.transform(X_forecast)
    y_forecast = model.predict(X_forecast_poly)
    forecast_df = pd.DataFrame({'Дата - время': filtered_df['Дата - время'], 'Прогнозируемые уровни воды': y_forecast})
    real_values = df_filtered[['Дата - время', 'Уровень воды']]
    merged_df = forecast_df.merge(real_values, on='Дата - время', how='left')
    print(merged_df)
else:
    print("Нет данных для выбранной комбинации кода поста и дат прогнозирования!")
