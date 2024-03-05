import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Input
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
from database import GetNcms
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle

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

valores = ['500']


#
# df_duplicado_pj = dfNcms[dfNcms['PJ_PF'].isin(["pj"])]
#
# df_duplicado_pf = dfNcms[dfNcms['PJ_PF'].isin(["pf"])]

df_duplicado_CSOSN = dfNcms[dfNcms['CSOSN'].isin(valores)]

dfNcms = pd.concat([dfNcms, df_duplicado_CSOSN, df_duplicado_CSOSN,df_duplicado_CSOSN ])

#quadruplica os dados para balancear as classes



dfNcms = dfNcms.reset_index(drop=True)



# separa os dados em X e y
X = dfNcms.iloc[:, :1]

y = dfNcms.iloc[:, 1:]

# y = y.astype(int)
# cria um onehot encoder para as classes



X = X.values

y = y.values

#faz o shuffle dos dados x e y

X, y = shuffle(X, y, random_state=0)


encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

X = encoder.fit_transform(X)


#X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=453)

#ann = tf.keras.models.Sequential()

# Define the structure of the layers
inputs = Input(shape=(X.shape[1], ))  # The shape of your input data
dense1 = tf.keras.layers.Dense(units=32, activation='relu')(inputs)
dense2 = tf.keras.layers.Dense(units=64, activation='relu')(dense1)
dense3 = tf.keras.layers.Dense(units=64, activation='relu')(dense2)

# Now let's add two output layers for each of the class
class_A_layer = tf.keras.layers.Dense(units=3, activation='sigmoid', name='y_class_CSOSN')(dense3)
class_B_layer = tf.keras.layers.Dense(units=2, activation='sigmoid', name='y_class_CFOP')(dense3)


# Create the model
model = tf.keras.Model(inputs=inputs, outputs=[class_A_layer, class_B_layer])

# Compile the model with different loss functions for each output layer
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

y_class_CSOSN = y[:, 0]
y_class_CFOP = y[:, 1]

label_encoderCSOSN = LabelEncoder()
label_encoderCFOP = LabelEncoder()

y_class_CSOSN = label_encoderCSOSN.fit_transform(y_class_CSOSN)
y_class_CFOP = label_encoderCFOP.fit_transform(y_class_CFOP)

# Fit the model
history = model.fit(X, {'y_class_CSOSN': y_class_CSOSN, 'y_class_CFOP': y_class_CFOP},
                    batch_size=32, epochs=20, validation_split=0.1)


# treina o modelo de rede neural, separa 20% dos dados para validação e utiliza early stopping para evitar overfitting
# = tf.keras.optimizers.Adam()

# ajusta o learning rate do otimizador

#opt.learning_rate.assign(0.005)

#loss = tf.keras.losses.SparseCategoricalCrossentropy()
##ann.compile(optimizer=opt, loss=loss, metrics=['accuracy'], outputs=[class_A_layer, class_B_layer])

# early_stopping = tf.keras.callbacks.EarlyStopping(
#     patience=2,
#     min_delta=0.001,
#     restore_best_weights=True,
#     monitor='loss'
# )

#history = ann.fit(X, y, batch_size=64, epochs=5, callbacks=[early_stopping], validation_split=0.1)

# faz a predição dos dados de teste
# y_pred = ann.predict(X_test)


# transforma os dados de predição em classes
# y_pred_classes = np.argmax(y_pred, axis=-1)

# y_pred = label_encoder.inverse_transform(y_pred_classes)

# y_test = label_encoder.inverse_transform(y_test)

# imprime os dados de X_test com todas as colunas e y_test com a predição para comparação
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))


# print("score: ", accuracy_score(y_test, y_pred))


# imprime a matriz de confusão com e com a legenda
# print(confusion_matrix(y_test, y_pred))

print(encoder.get_feature_names_out())

# teste = ['25232100','500']

teste1 = ['25232100']

teste1 = np.array(pd.DataFrame(encoder.transform([teste1])))

res = model.predict(teste1)


print("CSOSN: ", label_encoderCSOSN.inverse_transform([np.argmax(res[0][0])]),
      "CFOP: ", label_encoderCFOP.inverse_transform([np.argmax(res[1][0])]))

print(history.history.keys())


plt.plot(history.history['y_class_CSOSN_accuracy'])
plt.plot(history.history['y_class_CFOP_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train A', 'Train B', 'Validation A', 'Validation B'], loc='upper left')
plt.show()


joblib.dump(encoder, 'encoder_ncm.joblib')

#salva o label encoder
joblib.dump(label_encoderCSOSN, 'label_encoderCSOSN.joblib')
joblib.dump(label_encoderCFOP, 'label_encoderCFOP.joblib')


model.save('modelo_ncm.keras')

# Obtém os nomes das classes do encoder
class_names = encoder.get_feature_names_out()

print("Nomes das classes:", class_names)
