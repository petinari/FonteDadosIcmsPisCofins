from pymongo import MongoClient


# metodo que retorna uma conexao ao banco de dados
def __get_connection():
    return MongoClient('localhost', 27017)

def SalvaNCMs(itens_list):
    conn = __get_connection()
    db = conn['db']
    collection = db['NCMS']

    collection.delete_many({})

    collection.insert_many([item for item in itens_list])
    conn.close()

def GetItensIcms():
    with __get_connection() as conn:
        db = conn['db']
        collection = db['ICMS']
        itens = collection.find()
        # retorna os itens e não precisa se preocupar em fechar a conexão
        return [item for item in itens]

def GetNcms():
    with __get_connection() as conn:
        db = conn['db']
        collection = db['NCMS']
        itens = collection.find()
        # retorna os itens e não precisa se preocupar em fechar a conexão
        return [item for item in itens]