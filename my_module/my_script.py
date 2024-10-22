import pandas as pd
from datetime import datetime
import urllib.parse
import numpy as np
import calendar
import multiprocessing
import re
import math
import random
from copy import deepcopy

def get_between(value, lower, upper):
    return max(lower, min(value, upper))

def print_result(*datas):
    for data in datas:
        print()
        print(data)
        if isinstance(data, pd.Series):
            data = data.to_frame()
        if isinstance(data, pd.DataFrame):
            df = data.describe().T
            if 'mean' in df.columns: 
                df['sharpe'] = df['mean'] / df['std']
            df['nan'] = data.isna().sum()
            df['0'] = (data.eq(0)).sum()
            df['1'] = (data.eq(1)).sum()
            df['-1'] = (data.eq(-1)).sum()
            df['inf'] = data.isin([np.inf]).sum()
            df['-inf'] = data.isin([-np.inf]).sum()
            if 'sharpe' in df.columns:
                df = df.sort_values('sharpe', ascending=False)
                print(df)
            else:
                print(df)

def is_datetime(index):
    return pd.api.types.is_datetime64_any_dtype(index)

def col_to_index(df, column):
    if isinstance(df, pd.DataFrame):
        if column in df.columns:        
            df.set_index(column, inplace=True, drop=True)
    return df

def col_to_datetimeindex(df, column, unit=None):
    df = col_to_index(df, column)
    df.index.name = '_id'
    if unit: df.index = pd.to_datetime(df.index, unit=unit)
    else: df.index = pd.to_datetime(df.index)
    return df

def datetimeindex_to_unixindex(data, column, unit=None): 
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        if unit=='s': m = 10**9
        if unit=='ms': m = 10**6
        if unit:
            data.index = data.index.astype('int64') / m
            data.index = data.index.astype('int64')
    return data

def get_current_unix(is_milliseconds=False):
    unix = int(datetime.datetime.now().timestamp())
    if is_milliseconds: 
        return unix * 1000
    else: 
        return unix

def get_current_datetime():
    return datetime.datetime.now()

def get_argument_names(func):
    import inspect
    return inspect.getfullargspec(func).args

def get_duration(interval, window):
    if 'd' in interval:
        duration = window * string_to_num(interval) * 24 * 60 * 60
    elif 'h' in interval:
        duration = window * string_to_num(interval) * 60 * 60
    elif 'm' in interval:
        duration = window * string_to_num(interval) * 60
    return int(duration)

def rename_cols(df, column_names):
    if isinstance(df, pd.Series) and isinstance(column_names, str):
        df.name = column_names
    elif isinstance(df, pd.Series) and isinstance(column_names, list):
        df.name = column_names[0]
    elif isinstance(df, pd.DataFrame) and isinstance(column_names, list):
        df.columns = column_names
    elif isinstance(df, pd.DataFrame) and (len(df.columns) == 1) and isinstance(column_names, str):
        df.columns = [column_names]
    return df

def string_to_camelcase(string):
    words = string.replace("_", " ").replace("-", " ").split()
    camel_case = words[0].lower() + ''.join(word.title() for word in words[1:])
    return camel_case

def string_to_capital(string):
    if isinstance(string, tuple) or isinstance(string, list):
        string = ' '.join([str(s) for s in string])
    words = str(string).replace('_', ' ').replace('-', ' ').split()
    capital = ' '.join(word.title() for word in words)
    return capital

def cols_to_lowercase(cols):
    if isinstance(cols, pd.MultiIndex):
        cols = cols.map(lambda x: (str(x[col]).lower().replace('-', '') for col in x))
    else:
        cols = cols.str.lower()
    return cols

def get_params_url(params):
    if 'args' in params: 
        params.update(params.pop('args'))
    params = {k:v for k, v in params.items() if v is not None}
    if params: 
        params_url = '?{}'.format(urllib.parse.urlencode(params))
    else: 
        params_url = ''
    return params_url

def get_missing_date(df, interval):
    interval = str(interval).replace('m', 'ME') if 'm' in interval else interval
    if df is not None and (  isinstance(df, pd.DataFrame) or isinstance(df, pd.Series)  ) and isinstance(df.index , pd.DatetimeIndex):
        return pd.date_range(start=df.index[0], end=df.index[-1], freq=interval).difference(df.index)

def get_the_first_missing_date(df, interval):
    dates = get_missing_date(df, interval)
    if dates is not None and not dates.empty: return dates[0]

def print_missing_date(df, interval):
    interval = str(interval).replace('m', 'ME') if 'm' in interval else interval
    if df is not None and (  isinstance(df, pd.DataFrame) or isinstance(df, pd.Series)  ) and isinstance(df.index , pd.DatetimeIndex):
        dates = pd.date_range(start=df.index[0], end=df.index[-1], freq=interval).difference(df.index)
        if dates.empty: print('No missing date')
        else:
            for i in dates: print(i, 'missing')
            return True

def ffill_for_missing_date(data, freq):    
    date_range = pd.date_range(start=data.index[0], end=data.index[-1], freq=str(freq))
    df = data.reindex(date_range).ffill()
    return df

def fill0_for_missing_date(data, freq):    
    freq = str(freq).replace('m', 'ME') if 'm' in freq else freq
    date_range = pd.date_range(start=data.index[0], end=data.index[-1], freq=freq)
    print(date_range)
    data = data.reindex(date_range).fillna(0)
    return data

def add_prefix(data, prefix):
    if isinstance(data, pd.Series):
        data.name = '{}{}'.format(str(prefix).lower(), str(data.name).lower().replace('-', '_'))
    elif isinstance(data, pd.DataFrame):
        data.columns = [col.lower().replace('-', '_') for col in data.columns]
        data = data.add_prefix(prefix)
    elif isinstance(data, list):
        data = [str(prefix).lower() + str(string) for string in data]
    return data

def add_suffix(data, suffix):
    if isinstance(data, pd.Series):
        data.name = f"{str(data.name).lower().replace('-', '_')}{str(suffix).lower()}"
    elif isinstance(data, pd.DataFrame):
        data.columns = [str(col).lower().replace('-', '_') for col in data.columns]
        data = data.add_suffix(suffix)
    elif isinstance(data, pd.MultiIndex):
        data.columns = [(str(col).lower().replace('-', '_') for col in multicol) for multicol in data.columns]
        data.columns = [str(multicol[-1]) + str(suffix).lower() for multicol in data.columns]
    return data

def replace_substring(df, old, new):
    if isinstance(df, pd.Series):
        df.name = str(df.name).lower().replace('-', '_').replace(old, new)
    elif isinstance(df, pd.DataFrame):
        df.columns = [col.lower().replace('-', '_').replace(old, new) for col in df.columns]
    return df

def add_0value_row(data):
    new_row = pd.DataFrame({col: [0] for col in data.columns}, index=[(data.index[0] - np.diff(data.index)[0])])
    df = pd.concat([new_row, data])
    return df

def string_to_num(string):
    nums = re.findall(r'[-+]?\d*\.\d+|\d+', string)
    if len(nums) > 0:
        num = float(nums[0])
        return num
    else:
        return string

def interval_to_second(interval, is_milliseconds=False):
    interval = str(interval).lower()
    num = string_to_num(interval)
    if 'm' in interval:
        second = num * 60
    elif 'h' in interval:
        second = num * 60 * 60
    elif 'd' in interval:
        second = num * 60 * 60 * 24
    if is_milliseconds: 
        return int(second) * 1000
    else: 
        return int(second)

def interval_to_timedelta(interval, multiply=1):
    interval = str(interval).lower()
    value = interval[:-1]
    unit = interval[-1]
    match unit:
        case 'y':
            days = float(value) * multiply *365
            return datetime.timedelta(days=days)   
        case 'm':
            days = float(value) * multiply *30
            return datetime.timedelta(days=days)   
        case 'w':
            days = float(value) * multiply *7
            return datetime.timedelta(days=days)   
        case 'd':
            days = float(value) * multiply
            return datetime.timedelta(days=days)    
        case 'd':
            hours = float(value) * multiply
            return datetime.timedelta(hours=hours)
        case 'd':
            minutes = float(value)* multiply
            return datetime.timedelta(minutes=minutes)

def count_interval(duration, interval):
    if isinstance(duration, int):
        return int(duration / interval_to_second(interval))
    elif isinstance(duration, str):
        return int(interval_to_second(duration) / interval_to_second(interval))

def annualized_factor(days, interval):
    interval = str(interval).lower()
    num = string_to_num(interval)
    if 'm' in interval:
        factor = (24 * 60) / num
    elif 'h' in interval:
        factor = 24 / num
    elif 'd' in interval:
        factor = 1 / num
    return math.sqrt(days * factor)

def year_month_day_to_unix(year, month, day, hour=0, minute=0, second=0):
    string = '{}-{}-{} {}:{}:{}'.format(year, month, day, hour, minute, second)
    date_time = datetime.datetime.strptime(string, '%Y-%m-%d %H:%M:%S')
    return int(date_time.timestamp())

def var_to_datetime(var, format='%Y-%m-%d', unit='s'):
    if isinstance(var, str):
        return string_to_datetime(var, format)
    elif isinstance(var, int):
        return unix_to_datetime(var, unit)
    else:
        return var

def var_to_ymd(var=int or datetime or str, unit='s'):
    if isinstance(var, str):
        return var
    elif isinstance(var, int):
        var = unix_to_datetime(var, unit)
    return var.strftime('%Y-%m-%d')

def string_to_datetime(string, format='%Y-%m-%d'):
    if isinstance(string, str):
        return datetime.datetime.strptime(string, format)

def unix_to_datetime(unix, unit='s'):
    if unit == 'ms': unix /= 10**3
    return datetime.datetime.fromtimestamp(unix)

def string_to_unix(year_month_day, hour_minute_second=''):
    if hour_minute_second:
        date_time = datetime.datetime.strptime(f"{year_month_day} {hour_minute_second}", '%Y-%m-%d %H:%M:%S')
    else:        
        date_time = datetime.datetime.strptime(year_month_day, '%Y-%m-%d')
    return int(date_time.timestamp())

def round_up_datetime_index(df, freq):
    df.index = df.index.to_series().apply(lambda x: x.ceil(freq).date())
    return df

def get_last_day(year, month):
    last_day = calendar.monthrange(year, month)[1]
    return last_day

def format_number(num):
    if isinstance(num, float):
        if (num == -np.inf) or (num == np.inf):
            return 0
        elif (num <= -10):
            return int(num)        
        elif (num >= 10):
            return int(num)
        elif (-10 < num < 10) and (round(num, 2) != 0):
            return round(num, 2) 
        else:
            return '{:.1e}'.format(num)
    else: 
        return num

def drop_rows(df, window):
    if isinstance(window, int):
        return df.iloc[window:]
    elif isinstance(window, str):
        return df.loc[df.index[0] + pd.Timedelta(window):]

def drop_rows_with_inf_and_0(data):
    df = data.copy()
    df = df.drop(df[df.eq(0).any(axis=1)].index)
    df = df.drop(df[df.eq(np.inf).any(axis=1)].index)
    df = df.drop(df[df.eq(-np.inf).any(axis=1)].index)
    return df

def check_rolling_data_length(rolling_data, window):   
    if isinstance(window, str):
        if len(rolling_data) > 1:
            timedelta = rolling_data.index[-1] - rolling_data.index[0] + rolling_data.index[1] - rolling_data.index[0]
            if timedelta == pd.to_timedelta(window):
                return True
    elif isinstance(window, int):
        if len(rolling_data) == window:
            return True
    else: return False

def check_grouped_data_length(grouped_data, freq):
    if len(grouped_data) > 1:
        timedelta = grouped_data.index[-1] - grouped_data.index[0] + grouped_data.index[1] - grouped_data.index[0]
        if timedelta == pd.to_timedelta(freq):
            return True
    else: return False
        
def interval_to_resample_freq(interval):
    if 'm' in interval:
        freq = str(interval).replace('m', 'min')
    elif 'h' in interval:
        freq = str(interval).replace('h', 'H')
    elif 'd' in interval:
        freq = str(interval).replace('d', 'D')
    return freq

def apply_rolling_function(data, window, func, suffix='', print_result=False):
    df = data.dropna().rolling(window).apply(func).dropna()
    df = drop_rows(df, window)
    if suffix:
        df = add_suffix(df, '_{}{}'.format(suffix, window))
    if print_result:
        print_result(df)
    return df

def multi_process(func, args):
    results = []
    with multiprocessing.Pool(4) as pool:
        for i in args:
            if isinstance(i, tuple) or isinstance(i, list):            
                r = pool.apply_async(func,*i)
            else:
                r = pool.apply_async(func,i)
            results.append(r.get())
        pool.close()
        pool.join()
    return results

def cbbc_underlying_to_yf_symbol(underlying=str):
    match underlying:
        case 'HSTEC':
            return 'HSTECH.HK'
        case 'HSCEI':
            return '^HSCE'
    if str(underlying).isalpha():
        return f'^{underlying}'
    elif str(underlying).isnumeric() and len(str(underlying)) > 4:
        return f'{underlying[-4:]}.HK'
    else:
        return underlying
    
# def multi_process(func, tasks):
#     cpu_count = multiprocessing.cpu_count() // 2
#     if len(tasks) == 1:
#         return [func(tasks[0])]
#     elif len(tasks) > 1:
#         print(f"Multi processing {len(tasks)} tasks for {func.__name__}() with {cpu_count} processors.")
#         with multiprocessing.Pool(cpu_count) as pool:
#             if isinstance(tasks[0], list) or isinstance(tasks[0], tuple):
#                 results = pool.starmap(func, tasks)
#             else:
#                 results = pool.map(func, tasks)
#         pool.close()
#         pool.join()
#         return [i for i in results if i is not None]
#     else:
#         return []

# def multi_process(func, args):
#     def wrap(queue, func, arg):
#         if isinstance(arg, tuple) or isinstance(arg, list):            
#             result = func(*arg)
#         else:
#             result = func(arg)
#         queue.put(result)

#     # Create a multiprocessing Queue
#     queue = multiprocessing.Queue()
#     # Create a list to hold the process references
#     processes = []

#     # Create and start a process for each number from 1 to 10
#     for i in args:
#         process = multiprocessing.Process(target=wrap, args=(queue,func,i))
#         processes.append(process)
#         process.start()

#     # Wait for all processes to finish
#     for process in processes:
#         process.join()

#     # Retrieve results from the queue
#     r = []
#     while not queue.empty():
#         r.append(queue.get())
#     print(r)

def replace_inf_with_previous_non_inf(df):
    mask = ~np.isinf(df)
    df = df.where(mask).ffill()
    return df

def list_funcs_names_start_with_get(object):
    attributes = dir(object)
    return [attr for attr in attributes if attr.startswith('get') and callable(getattr(dir(object), attr))]

def get_url(base_url, slug='', params={}):
    if params and isinstance(params, dict):
        for k, v in params.items():
            if isinstance(v, dict):
                params.update(params.pop(k))
        params = {k:v for k, v in params.items() if v}

    if params and slug:
        return f"{base_url}/{slug}?{urllib.parse.urlencode(params)}"
    elif params and not slug:
        return f"{base_url}?{urllib.parse.urlencode(params)}"
    elif not params and slug:
        return f"{base_url}/{slug}"
    else:
        return base_url

def sort_dataframe_by_list(df, column_name, list):
    # Assign the desired sort order to the column
    df[column_name] = pd.Categorical(df[column_name], categories=list, ordered=True)

    # Sort the DataFrame based on the sorted column
    df = df.sort_values(column_name)

    # Reset the index of the sorted DataFrame if desired
    return df.reset_index(drop=True)

def df_to_list(df):
    if isinstance(df, pd.DataFrame):
        return {'columns':df.columns.tolist(), 'values':df.values.tolist()}

def format_time_info(from_ndays_before=0, to_ndays_before=0, from_datetime=None, to_datetime=None, df=None, interval='', unit=1):
    current_datetime = get_current_datetime()
    current_unix = current_datetime.timestamp() * unit

    from_unix = None
    to_unix = None
    if from_ndays_before:
        from_datetime = current_datetime - datetime.timedelta(  days=from_ndays_before  ) 
    if to_ndays_before:
        to_datetime = current_datetime - datetime.timedelta(  days=to_ndays_before  )
    if from_datetime:        
        from_unix = int(from_datetime.timestamp()) * unit
    if to_datetime:        
        to_unix = int(to_datetime.timestamp()) * unit
    if interval:
        delta_datetime = interval_to_timedelta(interval)
        delta = interval_to_second(interval) * unit
    else: delta = None

    if (  isinstance(df, pd.DataFrame) or isinstance(df, pd.Series)  ) and not df.empty:        
        first_datetime = df.first_valid_index()
        first_missing_date = get_the_first_missing_date(df, interval)
        last_datetime = df.last_valid_index()
        first_unix = int(first_datetime.timestamp()) * unit
        last_unix = int(last_datetime.timestamp()) * unit
    else: 
        first_datetime, last_datetime, first_unix, last_unix = None, None, None, None
    return {
        'current_datetime': current_datetime,
        'current_unix': current_unix,
        'delta_datetime': delta_datetime, 
        'delta': delta, 
        'from_datetime': from_datetime, 
        'to_datetime': to_datetime, 
        'from_unix': from_unix, 
        'to_unix': to_unix, 
        'first_datetime': first_datetime, 
        'last_datetime': last_datetime,
        'first_unix': first_unix, 
        'last_unix': last_unix, 
        }

def time_to_unix(time, is_milliseconds=False):
    if isinstance(time, str):
        time = datetime.datetime.strptime(time, '%Y-%m-%d').timestamp()
    elif isinstance(time, datetime.datetime):
        time = time.timestamp()
    if is_milliseconds: 
        if len(str(time)) <= 10:
            time *= 1000
    return int(time)

def time_to_datetime(time):
    if isinstance(time, str):
        return datetime.datetime.strptime(time, '%Y-%m-%d')
    elif isinstance(time, int):
        digi = len(str(time))
        if digi > 10:
            time = time / 10**(digi - 10)
        return datetime.datetime.fromtimestamp(time)
    else: return time

def time_to_str(time, is_YMD=True):
    if is_YMD:
        format = '%Y-%m-%d'
    else:
        format = '%Y-%m-%d %H:%M:%S'
    if isinstance(time, int) or isinstance(time, float):
        if len(str(int(time))) > 10:
            time = time / 10**(len(str(time)) - 10)
        return datetime.datetime.fromtimestamp(time).strftime(format)
    elif isinstance(time, datetime.datetime):
        return time.strftime(format)
    else: 
        return time

def format_str(string):
    return string.replace('-', '_').replace(' ', '_').replace('.', '_')

def format_interval(interval):
    if interval == 'daily':
        return '1d'

def get_arguments(func):
    import inspect
    return inspect.getfullargspec(func).args

def get_lags_forwards(y, x, start=-1, end=1, step=1):
    if isinstance(x, pd.Series): x = x.to_frame()        
    lags = [y]
    for i in range(start, end+1, step):
        lag = x.shift(i)
        lag.columns = [f'{c}_{i}' for c in lag.columns]
        lags.append(lag)
    return pd.concat(lags, axis=1)

def sort_column_order_by_the_last_row_values(df, is_ascending=False):
    last_row = df.iloc[-1]
    sorted_columns = last_row.sort_values(ascending=is_ascending).index
    return df[sorted_columns]

def check_list_of_tuples(lst):
    for item in lst:
        if not isinstance(item, tuple):
            return False                     
    return True

def first_invalid_index(df):
    result = df.index[df.isnull().any(axis=1)].tolist()
    if result:
        return result[0]

def count_nan_values_in_each_column(df):
    # Count the number of NaN values in each column
    return df.isnull().sum()

def get_int_time():
    current = datetime.datetime.now()
    return current.hour *100 + current.minute

def get_rolling_correlation(df, window=250):
    y = df.columns[0]
    xs = df.columns[1:]
    dfs = []
    for x in xs:
        corr = df[y].rolling(window).corr(df[x])
        dfs.append(  corr  )
    return pd.concat(dfs, axis=1)

def get_the_first_position(number):
    if number == 0:
        return 0    
    number = abs(number)
    exponent = math.floor(math.log10(number))
    position = math.pow(10, exponent)
    return position

def get_the_last_position(number):
    if number == 0:
        return 0    
    if isinstance(number, float):
        number -= float(str(number)[:-1])
    elif isinstance(number, int):
        number -= int(str(number)[:-1]) *10
    return get_the_first_position(number)

def get_scaled_df(df, min=0, max=0):
    if isinstance(df, pd.DataFrame) and len(df.columns) == 1:
        df = df.squeeze()
    if isinstance(df, pd.Series):
        return (df - df.min()) / (df.max() - df.min()) * (max - min) + min
    elif isinstance(df, pd.DataFrame):
        columns = df.columns
        first_df = df[columns[0]]
        first_min = first_df.min()
        first_max = first_df.max()
        dfs = [first_df]
        for i in columns[1:]:
            this_df = df[i]
            this_min = this_df.min()
            this_max = this_df.max()
            this_df = (this_df - this_min) / (this_max - this_min) * (first_max - first_min) + first_min
            dfs.append(this_df)
        return pd.concat(dfs, axis=1)

def get_round_param(number, num_of_digits=1):
    if number == 0:
        return 0
    return num_of_digits - str(float(number)).index('.')

def get_random_color():
    return "rgb" + str(tuple(random.sample(range(0, 256), 3)))

def columns_to_strings(columns):
    if isinstance(columns, pd.MultiIndex) or all([  isinstance(c, tuple) for c in columns   ]):
        return [','.join(str(i) for i in c) for c in columns]
    else:
        return [str(c) for c in columns]    

def strings_to_columns(columns):
    columns = [  int(c) if c.isdigit() else c for c in columns  ]
    columns = [  eval(c) if c.startswith('(') and c.endswith(')') else c for c in columns  ]
    columns = [  float(c) if '.' in c else c for c in columns  ]
    columns = [  tuple(c.split(',')) if isinstance(c, str) and ',' in c else c for c in columns  ]
    if all(isinstance(c, tuple) for c in columns):
        columns = pd.MultiIndex.from_tuples(columns)
    return columns

def logarithm(dataframe, start_time_delta=-1, duration=1):
    df = dataframe.copy()
    start = -start_time_delta
    end = start - duration
    df = np.log(df.shift(end)) - np.log(df.shift(start))
    return df.replace([np.inf, -np.inf], np.nan).dropna()

def get_log_returns(ohlc, start_time_delta=-1, duration=1):
    p = ohlc.copy()
    start = -start_time_delta
    end = start - duration
    if isinstance(p, pd.Series):
        return np.log(p.shift(end) / p.shift(start))
    elif isinstance(p, pd.DataFrame):
        opens = ['Open', 'open', 'o']
        closes = ['Close', 'close', 'c']
        returns = []
        if isinstance(p.columns, pd.MultiIndex):
            df = p[[c for c in p.columns for x in opens + closes if x in c]]
            return pd.DataFrame(np.log(df.shift(end) / df.shift(start)))
        else:
            for i in closes + opens:
                if i in p.columns:
                    p[f'{i}_{i}'] = np.log(p[i].shift(end) / p[i].shift(start))
                    returns.append(p[f'{i}_{i}'])
            for o in opens:
                for c in closes:
                    if o in p.columns and c in p.columns:
                        p[f'{c}_{o}'] = np.log(p[o].shift(end) / p[c].shift(start))
                        p['Intraday'] = np.log(p[c].shift(end) / p[o].shift(end))
                        returns.append(p[[f'{c}_{o}', 'Intraday']])
        return pd.concat(returns, axis=1).dropna()

def search_columns(dataframe, *searches, is_index=False):    
    if isinstance(dataframe, pd.Series):
        return dataframe
    df = dataframe.copy()
    if searches:
        for search in searches:
            if not isinstance(search, list) and not isinstance(search, tuple):
                search = [search]

            if is_index:
                search = [  c for c in search if c < len(df.columns)  ]
                if search:
                    return df.iloc[:,search]
            else:            
                search = [c for c in df.columns for s in search if s in c]            
                df = df[search]
    return df

def combination(elements, no_of_elements_per_combination, is_lists=True):
    import itertools
    if no_of_elements_per_combination > 1:
        if len(elements) >= no_of_elements_per_combination:
            combination = list(itertools.combinations(elements, no_of_elements_per_combination))            
            return [list(i) for i in combination] if is_lists else combination
    return [[i] for i in elements]

def as_df(series):    
    df = series.copy()
    return df.to_frame() if isinstance(series, pd.Series) else df
    
def as_list(var):    
    if not isinstance(var, list):
        return [var]
    
def filter_out_None(array):
    if isinstance(array, list):
        return [  i for i in array if i is not None  ]

def is_structure_same(old, new):
    if (set(old.columns) == set(new.columns)):
        if (type(old.index) == type(new.index)) or (isinstance(old.index, pd.Index) and isinstance(new.index, pd.RangeIndex)) or (isinstance(old.index, pd.RangeIndex) and isinstance(new.index, pd.Index)):
            return True
        else:
            print("The index types are different")
            print(type(old.index))
            print(type(new.index))
            return False
    else:
        print("The column names are different")
        print(list(old.columns))
        print(list(new.columns))
        return False

def between_time(dataframe, start, end, freq):
    return dataframe.loc[dataframe.index[dataframe.index.isin(pd.date_range(start, end, freq=freq))]]

class DFTool:
    def __init__(self, DataFrame):
        self.IS_DF = isinstance(DataFrame, pd.DataFrame)
        self.IS_SERIES = isinstance(DataFrame, pd.Series)            
        self.DF = deepcopy(DataFrame)
    def get_indexes(self):
        if self.IS_DF or self.IS_SERIES:
            return self.DF.index
    def get_index_dtype(self):
        i = self.get_indexes()
        return i.dtype
    def get_columns(self):
        if self.IS_DF:
            return self.DF.columns
    def get_nth_row(self, n=-1):
        if self.IS_DF:
            return self.DF.iloc[n]
    def get_nth_column(self, n=1):
        if self.IS_DF:
            return self.DF.iloc[:, n]
    def get_column(self, column_name):
        if self.IS_DF and column_name in self.DF.columns:
            return self.DF[column_name]
    def list_column_values(self, column_name):
        i = self.get_column(column_name)
        return i.tolist()
    # def get_stats(self):
    #     df = self.DF.describe().T
    #     df['nan'] = df.isna().sum()
    #     df['0'] = (df.eq(0)).sum()
    #     df['1'] = (df.eq(1)).sum()
    #     df['-1'] = (df.eq(-1)).sum()
    #     df['inf'] = df.isin([np.inf]).sum()
    #     df['-inf'] = df.isin([-np.inf]).sum()
    #     return df

def is_picklable(obj):
    import pickle
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PicklingError, TypeError):
        return False