import pandas as pd
from database import SalvaNCMs


dataframeNCM = pd.read_json('Tabela_NCM_Vigente_20240205.json')

# Flattening the json
jsonNcm = pd.json_normalize(dataframeNCM['Nomenclaturas'])

#remove as linhas que o codigo da ncm tem menos de 8 caracteres
jsonNcm = jsonNcm[jsonNcm['Codigo'].str.len() == 8]

#adiciona a coluna codigo a variavel PJ ou PF
jsonNcm['PJ_PF'] = ""

#adiciona a coluna codigo a variavel CSOSN
jsonNcm['CSOSN'] = ""

new_rows = []

for index, row in jsonNcm.iterrows():
    ncm = row['Codigo']
    new_rows.append({'ncm': ncm, 'PJ_PF': 0, 'CSOSN': row['CSOSN']})
    new_rows.append({'ncm': ncm, 'PJ_PF': 1, 'CSOSN': row['CSOSN']})

jsonNcm = pd.DataFrame(new_rows)

#adicina a coluna regime a variavel jsonNcm na posição 2
jsonNcm.insert(2, 'Regime', 'Simples Nacional')

#preenche com 101 o CSOSN quando o PJ_PF for 0
jsonNcm.loc[jsonNcm['PJ_PF'] == 0, 'CSOSN'] = 101

#preenche com 102 o CSOSN quando o PJ_PF for 1
jsonNcm.loc[jsonNcm['PJ_PF'] == 1, 'CSOSN'] = 102


SalvaNCMs(jsonNcm.to_dict('records'))