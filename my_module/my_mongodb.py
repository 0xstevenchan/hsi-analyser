import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from copy import deepcopy
from .my_script import *

class MongodbInit:
    def __init__(self, address:str, db_name:str):
        self.address = address
        self.db_name = db_name
        if 'mongodb+srv' in self.address:
            self.mongo_client = MongoClient(self.address, server_api=ServerApi('1'))
        else:
            if ':' in self.address:
                host_port = self.address.split(':')
            else:
                host_port = [self.address, '27017']
            self.mongo_client = MongoClient(host_port[0], int(host_port[1]))
        self.db = self.mongo_client[self.db_name]
    def list_database_names(self):
        return sorted(self.mongo_client.list_database_names())
    def list_collection_names(self):
        return sorted(self.db.list_collection_names())

class MongodbReader(MongodbInit):
    def __init__(self, address:str, db_name:str, collection_name:str):
        super().__init__(address, db_name)
        self.collection_name = collection_name
        self.collection = self.db[self.collection_name]
    def collection_to_list(self, query={}, projection={}):
        return list(self.collection.find(query, projection))
    def collection_to_dataframe(self, query={}, projection={}):
        i = self.collection_to_list(query, projection)
        if i is not None: 
            df = pd.DataFrame(i)
            if '_id' in df:
                return df.set_index('_id')
            elif 'index' in df:
                return df.set_index('index')
    def find_last(self):
        return self.collection.find_one(sort=[('_id', -1)])
    def find_last_id(self):
        i = self.find_last()
        if i is not None: return i.get('_id')

class MongodbReaders(MongodbInit):
    def __init__(self, address:str, db_name:str):
        super().__init__(address, db_name)
        self.readers = {i:MongodbReader(address, db_name, i) for i in self.db.list_collection_names()}
    def get_collection_names_dataframes(self):
        return {k:v.collection_to_dataframe() for k,v in self.readers.items()}
    def get_collection_names_last_ids(self):
        return {k:v.find_last_id() for k,v in self.readers.items()}

class MongodbWriter(MongodbInit):
    def __init__(self, address=str, db_name=str, collection_name=str, data=pd.DataFrame or pd.Series or list or dict):
        super().__init__(address, db_name)
        self.collection_name = collection_name
        self.collection = self.db[self.collection_name]
        self.data = data
    def data_to_list(self):
        i = deepcopy(self.data)
        if isinstance(i, pd.Series):
            i = [i.to_dict('records')]
        elif isinstance(i, pd.DataFrame):
            i.columns = columns_to_strings(i.columns)
            i['_id'] = i.index
            i = i.to_dict('records')
        else:
            i = [i]
        return i
    def update_data(self):
        datum = self.data_to_list()
        results = []
        for i in datum:
            if isinstance(i, dict):
                i = {k:v for k,v in i.items() if not (isinstance(v, float) and math.isnan(v))}
                if '_id' in i:
                    filter = {'_id': i.pop('_id')}
                    data = {'$set': i}
                    result = self.collection.update_one(filter, data, upsert=True)
                else:
                    result = self.collection.insert_one(i)
            elif isinstance(i, list):
                result = self.collection.insert_one(i)
            results.append(result.acknowledged)
        return results

class MongodbWriters(MongodbInit):
    def __init__(self, address='', db_name='', collection_names_datum={}):
        if isinstance(collection_names_datum, dict):
            self.__collection_names_datum__ = collection_names_datum
        else:
            self.__collection_names_datum__ = {}
        super().__init__(address, db_name)
    def update_all_datum(self):
        results = []
        for k,v in self.__collection_names_datum__.items():
            result = self.update_data(k, v)
            results.append(result)
        return results

class MongodbDeleter(MongodbInit):
    def __init__(self, address='', db_name='', collection_names=list, action=str):
        super().__init__(address, db_name)
        self.collection_names = collection_names if isinstance(collection_names, list) else [collection_names]
        self.action = action
    def list_deletes(self):
        match self.action:
            case 'delete':
                return self.collection_names
            case 'keep':
                return [i for i in self.list_collection_names() if i not in self.collection_names]
            case _:
                return self.list_collection_names()
    def delete_documents(self):
        for i in self.list_deletes():
            collection = self.db[i]
            collection.delete_many({})
    def drop_collections(self):
        for i in self.list_deletes():
            collection = self.db[i]
            collection.drop()

if __name__ == '__main__':

    pass