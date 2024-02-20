import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from database import GetNcms
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit


#obtem a lista de NCMs que estão no banco de dados
itensNcms = GetNcms()

#cria um dataframe com os ncms do banco de dados
dfNcms = pd.DataFrame(itensNcms)

#drop all rows with CSOSN = 500
#dfNcms = dfNcms[dfNcms['CSOSN'] != 500]

#remove a coluna _id do dataframe
dfNcms.drop(columns=['_id', 'Regime'], inplace=True)

#transforma todas as colunas em string
dfNcms = dfNcms.astype(str)

#separa os dados em X e y
X = dfNcms.iloc[:,:-1]

y = dfNcms.iloc[:, -1]






#cria um onehot encoder para as classes

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

X = np.array(pd.DataFrame(encoder.fit_transform(X)))

#faz o encode de y

# Use LabelEncoder for the target variable instead of OneHotEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)



X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, shuffle=True, random_state=453)





ann = tf.keras.models.Sequential()

# Regularização
ann.add(tf.keras.layers.Dense(units=128, activation='relu'))

ann.add(tf.keras.layers.Dense(units=256, activation='relu' ))
ann.add(tf.keras.layers.Dense(units=256, activation='relu' ))

ann.add(tf.keras.layers.Dense(units=3, activation='softmax'))


#treina o modelo de rede neural, separa 20% dos dados para validação e utiliza early stopping para evitar overfitting
opt = tf.keras.optimizers.Adam()

#ajusta o learning rate do otimizador

opt.learning_rate.assign(0.005)


loss = tf.keras.losses.SparseCategoricalCrossentropy()
ann.compile(optimizer=opt,  loss=loss, metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=2,
    min_delta=0.001,
    restore_best_weights=True,
    monitor='loss'
)
# Do inverse transform on encoded targets to get the original labels




history = ann.fit(X_train, y_train, batch_size=64, epochs=100, callbacks=[early_stopping], validation_split=0.1)

#faz a predição dos dados de teste
y_pred = ann.predict(X_test)



#transforma os dados de predição em classes
y_pred_classes = np.argmax(y_pred, axis=-1)

y_pred = label_encoder.inverse_transform(y_pred_classes)

y_test = label_encoder.inverse_transform(y_test)

#imprime os dados de X_test com todas as colunas e y_test com a predição para comparação
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))


print("score: ", accuracy_score(y_test, y_pred))



#imprime a matriz de confusão com e com a legenda
print(confusion_matrix(y_test, y_pred))

print(encoder.get_feature_names_out())

#teste = ['25232100','1','500']

teste1 = ['28111200','1']

teste1 = np.array(pd.DataFrame(encoder.transform([teste1])))

res = ann.predict(teste1)

res_classes = np.argmax(res, axis=-1)
print(label_encoder.inverse_transform(res_classes))




#plota o gráfico de acurácia
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()




#salva o modelo treinado

ann.save('modelo_ncm.h5')

# Obtém os nomes das classes do encoder
class_names = encoder.get_feature_names_out()

print("Nomes das classes:", class_names)









