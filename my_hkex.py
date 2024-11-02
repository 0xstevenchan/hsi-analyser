import os
import pandas as pd
from copy import deepcopy
from datetime import datetime, timedelta
from my_module.my_io import DirPath, Downloader, ZipTool, CsvTool
from my_module.my_mongodb import MongodbWriter, MongodbReader
from my_module.my_script import *
from my_module.my_yfinance import YfWriter

class CbbcCsvPath(DirPath):
    def __init__(self):
        self.DELIMITER = '\t'
        self.TIMEDELTA_DAYS = 1
        self.csv_directory = os.path.join(os.getcwd(),'hkex','cbbc','csv')
        DirPath.__init__(self, self.csv_directory,'csv')
    def get_csv_file_name(self, year, month):
        return f"CBBC_{str(year)}_{str(month).zfill(2)}.csv"
    def get_zip_file_name(self, month):
        return f"CBBC{str(month).zfill(2)}.zip"
    def get_csv_file(self, year, month):
        i = self.get_csv_file_name(year, month)
        return os.path.join(self.csv_directory, i)
    def get_last_date(self):
        i = self.get_last_file()
        data = CsvTool(i, self.DELIMITER).filter_data()
        if data:
            return string_to_datetime(data[1][2])
    def get_timedelta(self, date):
        if date is not None:
            return datetime.now() - date
    def list_missing_years_months(self):
        last = self.get_last_date()
        delta = self.get_timedelta(last)
        if delta is not None:
            yms = []
            if delta >= timedelta(days=self.TIMEDELTA_DAYS):
                y = last.year
                m = last.month
                now = datetime.now()
                while y <= now.year and m <= now.month:
                    yms.append((y,m))
                    if m > 12:
                        m = 1
                        y += 1
                    else:
                        m += 1
            return yms
    def list_missing_csvs(self):
        yms = self.list_missing_years_months()
        if yms is None:
            return self.list_files()
        else:
            if yms:
                return [self.get_csv_file(*i) for i in yms]
            else:
                return []

class CbbcDownloader(CbbcCsvPath,Downloader):
    def __init__(self):
        super().__init__()
        self.BASE_URL = "https://www.hkex.com.hk/eng/cbbc/download"
        self.download_directory = os.path.join(os.getcwd(), 'hkex', 'cbbc', 'download')
        self.SLEEP_TIME = 1
        self.urls = self.list_missing_urls()
        Downloader.__init__(self, self.urls, self.download_directory, self.SLEEP_TIME)
    def list_missing_months(self):
        yms = self.list_missing_years_months()
        if yms:
            return set([i[1] for i in yms])
    def get_url(self, month):
        i = self.get_zip_file_name(month)
        return os.path.join(self.BASE_URL, i)
    def list_urls(self):
        return [self.get_url(i) for i in list(range(1, 13))]
    def list_missing_urls(self):
        return [self.get_url(i) for i in self.list_missing_months()]
    def extract(self, files):
        return ZipTool(files, self.csv_directory).extract()
    def rename(self, files):
        csvs = []
        for i in files:
            data = CsvTool(i, '\t').csv_to_list()
            date = data[1][2]
            ymd = str(date).split('-')
            name = self.get_csv_file_name(ymd[0], ymd[1])
            file = os.path.join(self.csv_directory, name)
            csvs.append(file)
            if os.path.exists(file):
                os.remove(file)
            os.rename(i, file)
        return csvs
    def download_extract_rename(self):
        i = self.download()
        j = self.extract(i)
        return self.rename(j)

class CbbcCsvReader(CbbcCsvPath):
    def __init__(self):
        super().__init__()
        self.csv_files = self.list_missing_csvs()
    def csv_to_dataframe(self):
        df = pd.concat([CsvTool(i, self.DELIMITER).csv_to_dataframe(1) for i in self.csv_files])
        if df is not None:
            df.columns = [str(i).lower().replace(' *', '').replace('**', '').replace('^', '').replace('.', '').replace(' ', '_').replace('/', '_') for i in df.columns] 
            for i in ['bull_bear', 'cbbc_type']:
                df[i] = df[i].str.rstrip().str.lower()
            for i in ['trade_date', 'listing_date', 'last_trading_date', 'maturity_date', 'delisting_date']:
                df[i] = pd.to_datetime(df[i].replace('-', np.nan))
            for i in [
                'average_price_per_cbbc_bought','average_price_per_cbbc_sold',
                'ent_ratio',
                'day_high','day_low',
                'strike_level','call_level',
                'closing_price',
                '%_of_issue_still_out_in_market','no_of_cbbc_still_out_in_market',
                'no_of_cbbc_bought', 'no_of_cbbc_sold', 
                'total_issue_size',
                'turnover','volume', 
                ]:
                df[i] = df[i].replace('N/A', np.nan).replace('-', np.nan).astype(float)
            return df

class CbbcDataFactory(CbbcCsvReader):
    def __init__(self):
        super().__init__()
        self.UNDERLYINGS = []
        self.INDEXES = ['bull_bear', 'underlying', 'mce', 'trade_date']
        self.COLUMNS = ['no_of_cbbc_sold','no_of_cbbc_bought','no_of_cbbc_still_out_in_market','turnover','volume']
        self.COLUMN_NAMES = ['num_sold', 'num_bought', 'in_market', 'turnover', 'volume']
        self.DROP_COLUMNS = ['strike_level', 'issuer', 'ent_ratio', '%_of_issue_still_out_in_market',
            'day_high','day_low','closing_price',
            'average_price_per_cbbc_bought','average_price_per_cbbc_sold', 
            'listing_date','maturity_date','delisting_date','last_trading_date',
            'cbbc_code','cbbc_name','cbbc_type','cbbc_category',
            'trading_currency','strike_call_currency']
        self.ADDRESS = 'mongodb+srv://sckyem:DirtBeCkqwdZutGZ@cluster0.xovoill.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
        self.df = None
        self.ohlcv = {}
    def set_dataframe(self):
        if self.df is None:
            self.df = self.csv_to_dataframe()
        return self.df
    def list_underlyings(self):
        self.set_dataframe()
        return self.df['underlying'].unique().tolist()
    def filter_underlyings(self):
        self.set_dataframe()
        df = deepcopy(self.df)
        if self.UNDERLYINGS:
            df = df[df['underlying'].isin(self.UNDERLYINGS)]
        return df
    def set_indexes(self):
        df = self.filter_underlyings()
        df = df.set_index(self.INDEXES)
        return df
    def set_columns(self, append_column=''):
        df = self.set_indexes()
        df['bought'] = df['no_of_cbbc_bought'] * df['average_price_per_cbbc_bought']
        df['sold'] = df['no_of_cbbc_sold'] * df['average_price_per_cbbc_sold']
        df['share'] = df['no_of_cbbc_still_out_in_market'] / df['ent_ratio']
        news = ['bought', 'sold', 'share']
        columns = news + self.COLUMNS
        if append_column:
            columns.append(append_column)
        df = df[columns]
        names = news + self.COLUMN_NAMES
        if append_column:
            names.append(append_column)
        df.columns = names
        return df
    def group_dataframe(self):
        df = self.set_columns()
        df = df.groupby(df.index).agg('sum')
        df.index = pd.MultiIndex.from_tuples(df.index)
        return df    
    def get_bull_bear_diff(self, dataframe=pd.DataFrame):
        dfs = []
        for i in dataframe:
            sub = dataframe[i].unstack(level=0)
            diff = sub['bull'] - sub['bear']
            diff.name = i
            dfs.append(diff)
        return pd.concat(dfs, axis=1)
    def unstack_multiindex(self, df=pd.DataFrame or pd.Series):
        if isinstance(df.index, pd.MultiIndex):
            levels = list(range(df.index.nlevels -1))
            df = df.unstack(levels)
            df = df.sort_index(axis=1)
        return df
    def make_hkex_cbbc(self):
        df = self.group_dataframe()
        df = self.get_bull_bear_diff(df)
        df = self.unstack_multiindex(df)
        df = df.swaplevel(0,1, axis=1)
        return df
    def make_hkex_cbbc_ratio(self):
        i = self.group_dataframe()
        df = pd.DataFrame()
        df['net_bought'] = i['bought'] - i['sold']
        df['net_num_bought'] = i['num_bought'] - i['num_sold']
        df['num_bought_to_bought'] = i['num_bought'] / i['bought']
        df['num_sold_to_sold'] = i['num_sold'] / i['sold']
        df['num_cbbc_to_share'] = i['in_market'] / i['share']
        df['volume_to_turnover'] = i['volume'] / i['turnover']
        df = self.get_bull_bear_diff(df)
        df = self.unstack_multiindex(df)
        df = df.swaplevel(0,1,axis=1)
        return df
    def get_close(self, underlying, date):
        if underlying not in self.ohlcv:
            yf_symbol = cbbc_underlying_to_yf_symbol(underlying)
            self.ohlcv[underlying] = MongodbReader(self.ADDRESS, 'yfinance', yf_symbol).collection_to_dataframe()
        close = self.ohlcv[underlying].loc[date, 'Close']
        return close
    def group_call_level(self):
        df = self.set_columns('call_level')
        df = df.xs('N', level=2, axis=0)
        df = df.groupby(df.index)
        dfs = []
        for index, i in df:
            underlying = index[1]
            date = index[2]
            close = self.get_close(underlying, date)
            i['call_level'] = (i['call_level'] / close).round(2)
            i = i.set_index('call_level', append=True)
            i = i.groupby(i.index).sum()
            dfs.append(i)
        df = pd.concat(dfs, axis=1).fillna(0).sort_index(ascending=False)
        df.index = pd.MultiIndex.from_tuples(df.index)
        return df

class CbbcMongodbWriter(CbbcDataFactory):
    def __init__(self):
        super().__init__()
        self.REGISTERED_COLLECTIONS = ['hkex_cbbc','hkex_cbbc_ratio']
        self.START = '2023-1-1'
        self.ADDRESS = 'mongodb+srv://sckyem:DirtBeCkqwdZutGZ@cluster0.xovoill.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
    def update_cbbc(self):
        for i in self.REGISTERED_COLLECTIONS:
            func_name = f"make_{i}"
            df = getattr(self, func_name)()
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.stack(level=0, future_stack=True).replace(0, np.nan).dropna(axis=0, how='all')
                dfs = df.groupby(level=1)
                for j,sub in dfs:
                    sub = sub.droplevel(1)
                    MongodbWriter(self.ADDRESS, i, j, sub).update_data()
                    print(f"Updating {i} {j}")
    def update_yf(self):
        i = [cbbc_underlying_to_yf_symbol(i) for i in self.list_underlyings()]
        YfWriter(self.ADDRESS, i, self.START).update()
    def update(self):
        missing = self.list_missing_csvs()
        if missing:
            CbbcDownloader().download_extract_rename()
            self.update_cbbc()
            self.update_yf()

if __name__ == '__main__':
    
    i = CbbcMongodbWriter().update()
    print(i)
    pass