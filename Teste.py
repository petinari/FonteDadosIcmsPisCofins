import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Input
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
from database import GetNcms
from imblearn.under_sampling import RandomUnderSampler

# obtem a lista de NCMs que estão no banco de dados
itensNcms = GetNcms()

# cria um dataframe com os ncms do banco de dados
dfNcms = pd.DataFrame(itensNcms)

# drop all rows with CSOSN = 500
# dfNcms = dfNcms[dfNcms['CSOSN'] != 500]

# remove a coluna _id do dataframe
dfNcms.drop(columns=['_id', 'Regime'], inplace=True)

# transforma todas as colunas em string
dfNcms = dfNcms.astype(str)

# separa os dados em X e y
X = dfNcms.iloc[:, :2]
y1 = dfNcms.iloc[:, 2]
y2 = dfNcms.iloc[:, 3]

# Fit and resample separately for y1, but keep indexes
rus1 = RandomUnderSampler(sampling_strategy='not majority')
X_res1, y1_res = rus1.fit_resample(X, y1)
indices = X_res1.index

# Use the indices from resampling y1 to also resample X and y2
X_res2 = X.loc[indices]
y2_res = y2.loc[indices]

# initializing label encoder
le = LabelEncoder()

# Looping for each column in dataframe
for col in X_res2.columns:
    X_res2[col] = le.fit_transform(X_res2[col])

# Create your model inputs
inputs = Input(shape=(X_res2.shape[1],))

dense1 = tf.keras.layers.Dense(units=128, activation='relu')(inputs)
dense2 = tf.keras.layers.Dense(units=256, activation='relu')(dense1)
dense3 = tf.keras.layers.Dense(units=256, activation='relu')(dense2)

output_layer_1 = tf.keras.layers.Dense(units=3, activation='sigmoid', name='CSOSN')(dense3)
output_layer_2 = tf.keras.layers.Dense(units=2, activation='sigmoid', name='CFOP')(dense3)

model = tf.keras.models.Model(inputs=inputs, outputs=[output_layer_1, output_layer_2])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

y1_res_encoded = LabelEncoder().fit_transform(y1_res)
y2_res_encoded = LabelEncoder().fit_transform(y2_res)

history = model.fit(X_res2, {'CSOSN': y1_res_encoded, 'CFOP': y2_res_encoded}, batch_size=32, epochs=100, validation_split=0.1)

plt.plot(history.history['CSOSN_accuracy'])
plt.plot(history.history['CFOP_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train A', 'Train B', 'Validation A', 'Validation B'], loc='upper left')
plt.show()

#... Continue with your model evaluation and saving here