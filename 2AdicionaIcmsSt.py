from itertools import chain

from database import GetItensIcms, GetNcms, SalvaNCMs
import pandas as pd
from thefuzz import fuzz

#obtem a lista de NCMs que estão no banco de dados
itensNcms = GetNcms()

#obtem os itens do banco de dados que são referentes ao ICMS ST
itensNcmsST = GetItensIcms()

listNcmSt = []

for item in itensNcmsST:
    listNcmSt.append(item["ncm"])

#cria um dataframe com os itens do banco de dados que são referentes ao ICMS ST
dfSt = pd.DataFrame(listNcmSt, columns=['NCM'])

dfSt = dfSt.astype(str)

#agrupo os NCMs por quantidade de caracteres e conta a quantidade de NCMs que tem a mesma quantidade de caracteres
print(dfSt['NCM'].str.len().value_counts().sort_index(ascending=False))



#cria um dataframe com os ncms do banco de dados
dfNcms = pd.DataFrame(itensNcms)

dfNcms.drop(columns=['_id'], inplace=True)



NCM_LENGTHS = dfSt['NCM'].str.len().unique()

# def calculate_fuzz_ratios(subset, ncm_subset):
#     return subset['ncm'].apply(lambda x: max([fuzz.ratio(str(x[:4]), str(ncm)) for ncm in ncm_subset.values]))

def calculate_fuzz_ratios(subset, ncm_subset,lenght):
    return subset['ncm'].apply(lambda x: max([(str(x[:lenght]) == str(ncm)) * 100 for ncm in ncm_subset.values]))

def update_csosn(df, indices):
    df.loc[indices, 'CSOSN'] = 500
    df.loc[indices, 'CFOP'] = 5405


def AjustaNcmSt(dfNcms, dfSt):

    ncm_by_len = {length: dfSt[dfSt['NCM'].str.len() == length] for length in NCM_LENGTHS}


    #verifica se tem algum item com nome 9 no dict ncm_by_len
    # ncm_by_len = {key: value for key, value in ncm_by_len.items() if key <= 8}


    mask_by_len = {length: dfNcms['ncm'].str[:length].isin(ncm_set['NCM']) for length, ncm_set in ncm_by_len.items()}

    # ratios = calculate_fuzz_ratios(dfNcms[mask_by_len[8]], ncm_by_len[8]['NCM'])
    # print(ratios[dfNcms['ncm'] == '73151100'])

    print(dfNcms.loc[dfNcms['ncm'] == '73151100', 'ncm'].str[:4])

    for length, mask in mask_by_len.items():
        dfNcms_subset = dfNcms[mask]
        ratios = calculate_fuzz_ratios(dfNcms_subset, ncm_by_len[length]['NCM'], length)
        index_to_update = ratios[ratios == 100].index
        update_csosn(dfNcms, index_to_update)

    return dfNcms




dfNcms = AjustaNcmSt(dfNcms, dfSt)

SalvaNCMs(dfNcms.to_dict('records'))







