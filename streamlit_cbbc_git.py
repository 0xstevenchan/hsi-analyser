import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from my_module.my_script import *
from my_module.my_mongodb import MongodbReaders, MongodbReader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datetime import time, timedelta

class StCbbc:
    def __init__(self):
        self.INDICATORS_NAMES = ['hkex_cbbc','hkex_cbbc_ratio']
        self.PAGE_TITLE = 'HSI Analyser'
        self.CONTENT_TITLE = 'HSI Analyser'
        self.CHART_TYPES = ['Line Chart', 'Table']
        self.N_COMPONENTS = 1
        self.mongodb_address = self.get_mongodb_address()
        st.session_state['dbs_underlyings'] = {}
    def get_mongodb_address(self):
        secrets = os.path.join(os.getcwd(), '.streamlit' , 'secrets.toml')
        if os.path.exists(secrets):
            return f'mongodb+srv://{st.secrets["user"]}:{st.secrets["pwd"]}@cluster0.xovoill.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
        else:
            return ''
    def select_item(self, key, options=list):
        if key in st.query_params:
            index = options.index(st.query_params[key])
        elif key in st.session_state:
            value = st.session_state[key]
            if value in options:
                index = options.index(value)
            else:
                index = 0
        else:
            if 'HSI' in options:
                index = options.index('HSI')
            else:
                index = 0
        option = st.sidebar.selectbox(f'Choose {str(key).capitalize()}', options, index, key=key)
        return option
    def select_indicator_name(self):
        return self.select_item('indicator', self.INDICATORS_NAMES)
    def select_underlying(self, db_name):
        if db_name not in st.session_state['dbs_underlyings']:
            underlyings = MongodbReaders(self.mongodb_address, db_name).list_collection_names()
            st.session_state['dbs_underlyings'].update({db_name: underlyings})
        else:
            underlyings = st.session_state['dbs_underlyings'][db_name]
        return self.select_item('underlying', underlyings)
    def get_collection(self, db_name, collection_name):
        if db_name not in st.session_state:
            st.session_state[db_name] = {}
        if collection_name in st.session_state[db_name]:
            collection = st.session_state[db_name][collection_name]
        else:
            collection = MongodbReader(self.mongodb_address, db_name, collection_name).collection_to_dataframe()
            if collection is not None:
                collection = collection.fillna(0)
            st.session_state[db_name][collection_name] = collection
        return collection
    def select_date(self, collection):
        date = st.sidebar.date_input('Choose Date', collection.index[-1], collection.index[252], collection.index[-1])
        date = datetime.combine(date, time.min)
        while date not in collection.index:
            date = date - timedelta(days=1)
        return date
    def select_ma(self, dataframe=pd.DataFrame or pd.Series):
        if 'ma' in st.query_params:
            value = int(st.query_params['ma'])
        else:
            value = 0
        ma = st.sidebar.number_input('Moving Average', min_value=0, value=value, step=5)
        st.query_params['ma'] = ma
        if ma:
            return dataframe.rolling(int(ma)).mean()
        else:
            return dataframe
    def get_pca(self, dataframe=pd.DataFrame):
        scaled_data = StandardScaler().fit_transform(dataframe)
        pca = PCA(n_components=self.N_COMPONENTS)
        pca_result = pca.fit_transform(scaled_data)
        pca_df = pd.DataFrame(data=pca_result, columns=[f'PCA{i}' for i in list(range(1, self.N_COMPONENTS+1))], index=dataframe.index)
        return pca_df
    def select_chart_type(self):
        return self.select_item('chart_type', self.CHART_TYPES)
    def show_chart(self, chart_type, collection):
        match chart_type:
            case 'Line Chart':
                for i in collection:
                    st.write(i)
                    st.line_chart(collection[i])
            case 'Table':
                st.write(collection)
    def run(self):
        st.set_page_config(self.PAGE_TITLE, layout="wide")
        st.markdown("""
            #GithubIcon {visibility: hidden;}
            """, unsafe_allow_html=True)
        st.title(self.CONTENT_TITLE)
        indicator_name = self.select_indicator_name()
        underlying = self.select_underlying(indicator_name)
        indicator = self.get_collection(indicator_name, underlying)
        indicator_ma = self.select_ma(indicator)
        chart_type = self.select_chart_type()
        self.show_chart(chart_type, indicator_ma)
class StOhlcv(StCbbc):
    def __init__(self):
        super().__init__()
        self.OHLCV_NAME = 'yfinance'
    def get_ohlcv(self, underlying):
        symbol = cbbc_underlying_to_yf_symbol(underlying)
        return self.get_collection(self.OHLCV_NAME, symbol)
    def show_candlestick(self, chart_type, ohlcv):
        if ohlcv is not None:
            match chart_type:
                case 'Line Chart':
                    fig = go.Figure(
                        go.Candlestick(
                            x=ohlcv.index,
                            open=ohlcv['Open'],
                            high=ohlcv['High'],
                            low=ohlcv['Low'],
                            close=ohlcv['Close'],
                            )
                        )
                    fig.update_layout(xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig)
                case 'Table':
                    st.dataframe(ohlcv, use_container_width=True)
    def run(self):
        st.set_page_config(self.PAGE_TITLE, layout="wide")
        st.title(self.CONTENT_TITLE)
        indicator_name = self.select_indicator_name()
        underlying = self.select_underlying(indicator_name)
        ohlcv = self.get_ohlcv(underlying)
        chart_type = self.select_chart_type()
        self.show_candlestick(chart_type, ohlcv)
class StScore(StOhlcv):
    def __init__(self):
        super().__init__()
        self.WINDOWS = list(range(1, 20, 1))
        self.sigma_multiplier = 0.5
    def get_ma(self, indicator, windows):
        df = deepcopy(indicator)
        dfs = []
        for i in df:
            for j in windows:
                if j > 1:
                    sub = df[i].rolling(j).mean()
                else:
                    sub = df[i]
                j = str(j).zfill(len(str(max(windows))))
                if isinstance(sub.name, tuple):
                    sub.name = (*i, j)
                else:
                    sub.name = f'{i},{j}'
                dfs.append(sub)
        df = pd.concat(dfs, axis=1).iloc[max(windows)-1:]
        return df
    def get_sigma(self, indicator, multiply=1):
        i = deepcopy(indicator)
        df = i / i.std()
        df = df * multiply
        return df
    def get_sign(self, indicator):
        i = deepcopy(indicator)
        df = np.sign(i)
        df = pd.DataFrame(df, columns=i.columns, index=i.index)
        return df
    def get_sigma_sign(self, sigma):
        i = deepcopy(sigma)
        df = np.where(i > 0, np.ceil(i), i)
        df = np.where(df < 0, np.floor(df), df)
        df = pd.DataFrame(df, columns=i.columns, index=i.index)
        return df
    def get_sign_of_last(self, sign):
        return sign.apply(lambda x: np.where(x == x.iloc[-1], 1, 0))
    def get_benchmark(self, ohlcv):
        return np.log(ohlcv['Close']).diff()
    def get_benchmark_open(self, ohlcv):
        return np.log(ohlcv['Open']).diff()
    def get_intraday_benchmark(self, ohlcv):
        i = deepcopy(ohlcv)
        i['up'] = np.log(i['High'] / i['Open'])
        i['down'] = np.log(i['Low'] / i['Open'])
        return i[['up', 'down']]
    def get_pnl_table(self, benchmark, sign, shift=2):
        df = pd.concat([benchmark, sign.shift(shift)], axis=1).dropna()
        dfs = []
        for i in sign:
            sub = df[i] * df[df.columns[0]]
            sub.name = i
            dfs.append(sub)
        return pd.concat(dfs, axis=1)
    def get_cpnl(self, pnl_table):
        return pnl_table.cumsum()
    def get_odd_ratio(self, pnl):
        p = (pnl[pnl > 0]).sum()
        n = (pnl[pnl < 0]).sum()
        return -p/n
    def get_trade(self, signs):
        trade = np.where((signs != 0) & (signs.shift() == 0), 1, 0)
        trade = pd.DataFrame(trade, columns=signs.columns, index=signs.index)
        return trade
    def get_score(self, pnl_table, trade, exposure):
        pnl = pnl_table.sum()
        num_trade = trade.sum()        
        df = pd.concat([pnl, num_trade, exposure], axis=1)
        df.columns = ['score', 'trade', 'exposure']
        df['adjust'] = np.where(df['score'] >= 0, df['trade'] * -0.01, df['trade'] * 0.01)
        df['adjust_score'] = df['score'] + df['adjust']
        df['adjust_score'] = np.where(df['score'] / df['adjust_score'] >= 0, df['adjust_score'], 0)
        df['adjust_score'] = df['adjust_score'] / df['exposure']
        df['action'] = np.where(pnl_table.iloc[-1] == 0, False, True)
        return df
    def get_result(self, adjust_score):
        df = deepcopy(adjust_score['adjust_pnl'])
        df.index = strings_to_columns(df.index)
        df = df.unstack(level=-1)
        df.columns = df.columns.astype(int)
        df = df.T.sort_index()
        df.columns = columns_to_strings(df.columns)
        return df
    def show_result(self, result, benchmark=None):
        chunk_size = 4
        chunk = [result.columns[i:i + chunk_size] for i in range(0, len(result.columns), chunk_size)]
        for i in chunk:
            sub = result[i]
            if benchmark is not None:
                sub['benchmark'] = benchmark.sum()
            st.line_chart(sub, use_container_width=True)
    def show_t2(self, date, ohlcv, benchmark):
        t2 = ohlcv.index.tolist().index(date) + 2            
        if t2 < len(ohlcv.index):
            st.metric(f'{var_to_ymd(ohlcv.index[t2])} Open', ohlcv['Open'].iloc[t2], f'{round(benchmark.iloc[t2]*100, 2)}%')
    def run(self):
        st.set_page_config(self.PAGE_TITLE, layout="wide")
        st.title(self.CONTENT_TITLE)
        indicator_name = self.select_indicator_name()
        underlying = self.select_underlying(indicator_name)
        ohlcv = self.get_ohlcv(underlying)
        if ohlcv is not None:
            benchmark = self.get_benchmark_open(ohlcv)
            indicator = self.get_collection(indicator_name, underlying)
            date = self.select_date(indicator)
            self.show_t2(date, ohlcv, benchmark)
            ohlcv = ohlcv.loc[:date]
            self.show_candlestick('Line Chart', ohlcv)
            indicator = indicator.loc[:date]
            ma = self.get_ma(indicator, self.WINDOWS)
            sigma = self.get_sigma(ma, self.sigma_multiplier)
            sign = self.get_sigma_sign(sigma)
            last = self.get_sign_of_last(sign)
            exposure = last.sum() / len(last.index)
            pnl_table = self.get_pnl_table(benchmark, last)

            if not pnl_table.empty:
                trade = self.get_trade(last)
                score = self.get_score(pnl_table, trade, exposure)
                threshold = abs(benchmark.sum())
                long = score.query(f'adjust_score > {threshold} and action == True')['adjust_score'].sort_values(ascending=False).reset_index(drop=True)
                short = score.query(f'adjust_score < {-threshold} and action == True')['adjust_score'].sort_values().reset_index(drop=True).abs()
                dont_long = score.query(f'adjust_score > {threshold} and action == False')['adjust_score'].sort_values(ascending=False).reset_index(drop=True)
                dont_short = score.query(f'adjust_score < {-threshold} and action == False')['adjust_score'].sort_values().reset_index(drop=True).abs()
                st.dataframe(sigma.tail(1))
                result = pd.concat([long, short, dont_long, dont_short], axis=1)
                result.columns = ['Long', 'Short', 'Dont Long', 'Dont Short']
                st.write('Larger value represents higher priority. Y axis for daily trading. X axis for monthly trading.')
                st.line_chart(result)


if __name__ == '__main__':
    
    i = StScore()
    j = i.run()
