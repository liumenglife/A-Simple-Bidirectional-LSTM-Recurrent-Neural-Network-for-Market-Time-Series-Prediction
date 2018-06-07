'''
    File name: model.py
    Author: Giacomo Corrias
    Date created: 01/06/2018
    Python Version: 3.6
'''
import numpy as np
from keras import layers
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from matplotlib.pyplot import legend
from pandas import DataFrame
from pandas import concat
from pandas import read_csv

# Returns a positive value if the close curve grows between two successive timesteps else returns a negative value.
def compute_close_target(first_timestep, second_timestep):
    if (first_timestep - second_timestep >= 0):
        return 0  # negative value
    else:
        return 1  # positive value

# Return a DataFrame for a supervised learning problem. 
# The output DataFrame contains three columns: 
# Close values at timestep t-1
# Close values at timestep t 
# Results of computeCloseTarget() 
def timeseries_to_supervised(data, lag=1):  # lag: number of shift value
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)] # Shift Close values by one timestep
    columns.append(df) # Second column
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)  # fill NaN value
    df.columns = ['Close_prec', 'Close'] # Rename columns
    df['Target'] = df.apply(lambda row: compute_close_target(row['Close_prec'], row['Close']), axis=1) # Third colum
    return df

# Generator method for timestep forecasting
# The original source code is taken from "Deep Learning with Python", Francois Chollet
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=64, step=1):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batchSize >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batchSize, max_index))
            i += len(rows)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(int(rows[j] - lookback), int(rows[j]), step)
            samples[j] = data[indices]
            targets[j] = data[int(rows[j]) + delay][2] # 2 denotes the third column
        yield samples, targets

# Load dataset
dataframe = read_csv('data/DAX_full.csv', parse_dates=[0], header=0, index_col=0, squeeze=True, usecols=["Date", "Close"])
dataset = dataframe.values
dataset = dataset.astype('float32')

# Normalizing, rescale the data
mean = dataset[:].mean(axis=0)
dataset -= mean
std = dataset[:].std(axis=0)
dataset /= std

# Convert the data for a supervised timeseries forecasting approach
supervised = timeseries_to_supervised(dataset, 1)
supervised_values = supervised.values

# Take data for train, validation and test sets
train_rate = 30
validation_rate = 40
train_row = round(int(len(supervised_values) * (train_rate / 100)))
validation_row = round(train_row + int(len(supervised_values)) * (validation_rate / 100))
test_row = validation_row + 1

# Problem Statement
# Given data going as far back as LOOKBACK timesteps and sampled every STEPS timesteps,
# can you predict the target of close values in DELAY timesteps?
# 1 timestep every 5 minutes
lookback = 672 # 672 timesteps = 1 week
steps = 96 # 96 timesteps = 8h
delay = 96 # 96 timesteps = 8h
batchSize = 64 

# Creating the generators
train_gen = generator(supervised_values, lookback=lookback, delay=delay, min_index=0, max_index=train_row, shuffle=True, step=steps, batch_size=batchSize)
val_gen = generator(supervised_values, lookback=lookback, delay=delay, min_index=train_row+1, max_index=validation_row, step=steps, batch_size=batchSize)
test_gen = generator(supervised_values, lookback=lookback, delay=delay, min_index=test_row, max_index=None, step=steps, batch_size=batchSize)

# Exact steps for every gen
validation_steps = (validation_row - (train_row+1) - lookback) / batchSize
test_steps = (len(supervised_values) - (validation_row+1) - lookback)
train_steps = (len(supervised_values) / batchSize)

# Simple bidirectional LSTM model
model = Sequential()
model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), input_shape=(None, supervised_values.shape[-1])))
model.add(layers.Bidirectional(layers.LSTM(256, dropout=0.2, recurrent_dropout=0.2), input_shape=(None, supervised_values.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early Stopping
#es = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto')

# Take the history for plot
history = model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=3, validation_data=val_gen, use_multiprocessing=False, validation_steps=validation_steps, workers=4)

# save model to single file
model.save('lstm_model.h5')

# load model from single file
#model = load_model('lstm_model.h5')

# Evaluating the model accuracy 
scores = model.evaluate_generator(test_gen, steps=1)
print("Accuracy Train: %.2f%%" % (scores[0] * 100))
print("Accuracy Test: %.2f%%" % (scores[1] * 100))

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

# Summarize history for loss
loss = history.history['loss']
validation_loss = history.history['val_loss']
epoches = range(1, len(loss) + 1)
plt.figure()
plt.plot(epoches, loss, 'bo', label='Training loss')
plt.plot(epoches, validation_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
legend(['train', 'validation'], loc='upper left')
plt.show()
