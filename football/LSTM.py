from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

action_size = 19

regression = Sequential()
regression.add(LSTM(units=100, activation="relu", return_sequences=True, input_shape = ((10, 48))))
regression.add(Dropout(0.2))

regression.add(LSTM(units=160, activation="relu", return_sequences=True))
regression.add(Dropout(0.3))

regression.add(LSTM(units=200, activation="relu", return_sequences=True))
regression.add(Dropout(0.4))

regression.add(LSTM(units=120, activation="relu"))
regression.add(Dropout(0.5))

regression.add(Dense(units = action_size))

regression.summary()