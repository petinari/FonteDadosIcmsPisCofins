import pandas as pd
from database import SalvaNCMs


dataframeNCM = pd.read_json('Tabela_NCM_Vigente_20240205.json')

# Flattening the json
jsonNcm = pd.json_normalize(dataframeNCM['Nomenclaturas'])

#remove as linhas que o codigo da ncm tem menos de 8 caracteres
jsonNcm = jsonNcm[jsonNcm['Codigo'].str.len() == 8]



#adiciona a coluna codigo a variavel CSOSN
jsonNcm['CSOSN'] = ""

jsonNcm['CFOP'] = ""

new_rows = []

for index, row in jsonNcm.iterrows():
    ncm = row['Codigo']
    new_rows.append({'ncm': ncm, 'CSOSN': row['CSOSN'], 'CFOP': row['CFOP']})




jsonNcm = pd.DataFrame(new_rows)

#adicina a coluna regime a variavel jsonNcm na posição 2
jsonNcm.insert(2, 'Regime', 'Simples Nacional')

jsonNcm['CSOSN'] = "101/102"

jsonNcm['CFOP'] = 5102


SalvaNCMs(jsonNcm.to_dict('records'))