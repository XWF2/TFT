#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中国房地产需求预测 - 特征工程（改进版）
China Real Estate Demand Prediction - Feature Engineering (Enhanced)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import warnings
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self, data_path='.'):
        self.data_path = data_path
        self.data = {}
        self.features = {}
        self.target = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        
        # 加载所有数据
        self.data['city_indexes'] = pd.read_csv(f'{self.data_path}/train/city_indexes.csv')
        self.data['city_search_index'] = pd.read_csv(f'{self.data_path}/train/city_search_index.csv')
        self.data['sector_POI'] = pd.read_csv(f'{self.data_path}/train/sector_POI.csv')
        self.data['land_transactions'] = pd.read_csv(f'{self.data_path}/train/land_transactions.csv')
        self.data['new_house_transactions'] = pd.read_csv(f'{self.data_path}/train/new_house_transactions.csv')
        self.data['pre_owned_house_transactions'] = pd.read_csv(f'{self.data_path}/train/pre_owned_house_transactions.csv')
        self.data['land_transactions_nearby'] = pd.read_csv(f'{self.data_path}/train/land_transactions_nearby_sectors.csv')
        self.data['new_house_transactions_nearby'] = pd.read_csv(f'{self.data_path}/train/new_house_transactions_nearby_sectors.csv')
        self.data['pre_owned_house_transactions_nearby'] = pd.read_csv(f'{self.data_path}/train/pre_owned_house_transactions_nearby_sectors.csv')
        
        # 统一时间格式
        for key in ['land_transactions', 'pre_owned_house_transactions', 
                    'land_transactions_nearby', 'new_house_transactions_nearby', 
                    'pre_owned_house_transactions_nearby']:
            if 'month' in self.data[key].columns:
                self.data[key]['month'] = pd.to_datetime(self.data[key]['month'], format='%Y-%b', errors='coerce')
        
        print("数据加载完成！")
    
    def create_temporal_features(self):
        """创建时间特征（改进版）"""
        print("\n创建时间特征...")
        
        # 处理新房交易数据
        df = self.data['new_house_transactions'].copy()
        df['month'] = pd.to_datetime(df['month'])
        
        # 基础时间特征
        df['year'] = df['month'].dt.year
        df['month_num'] = df['month'].dt.month
        df['quarter'] = df['month'].dt.quarter
        df['day_of_year'] = df['month'].dt.dayofyear
        df['week_of_year'] = df['month'].dt.isocalendar().week
        df['is_month_start'] = (df['month'].dt.day == 1).astype(int)
        df['is_month_end'] = df['month'].dt.is_month_end.astype(int)
        
        # 季节性特征
        df['is_spring'] = (df['month_num'].isin([3, 4, 5])).astype(int)
        df['is_summer'] = (df['month_num'].isin([6, 7, 8])).astype(int)
        df['is_autumn'] = (df['month_num'].isin([9, 10, 11])).astype(int)
        df['is_winter'] = (df['month_num'].isin([12, 1, 2])).astype(int)
        
        # 周期性特征（改进）
        df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        df['year_sin'] = np.sin(2 * np.pi * (df['year'] - 2019) / 6)  # 6年周期
        df['year_cos'] = np.cos(2 * np.pi * (df['year'] - 2019) / 6)
        
        # 时间趋势特征
        df['time_index'] = (df['year'] - 2019) * 12 + df['month_num'] - 1
        df['time_index_squared'] = df['time_index'] ** 2
        
        # 是否为节假日月份（中国春节通常在1-2月）
        df['is_holiday_month'] = df['month_num'].isin([1, 2, 10]).astype(int)
        
        self.data['new_house_transactions'] = df
        
        print(f"时间特征创建完成，新增特征数: {len([col for col in df.columns if col not in ['month', 'sector', 'amount_new_house_transactions']])}")
    
    def create_lag_features(self):
        """创建滞后特征（改进版）"""
        print("\n创建滞后特征...")
        
        df = self.data['new_house_transactions'].copy()
        
        # 按区域排序
        df = df.sort_values(['sector', 'month'])
        
        # 创建滞后特征（扩展滞后期）
        lag_periods = [1, 2, 3, 6, 12, 24]
        lag_features = ['amount_new_house_transactions', 'price_new_house_transactions', 
                       'area_new_house_transactions', 'num_new_house_transactions']
        
        for feature in lag_features:
            for lag in lag_periods:
                df[f'{feature}_lag_{lag}'] = df.groupby('sector')[feature].shift(lag)
        
        # 创建移动平均特征（扩展窗口）
        ma_periods = [3, 6, 12, 24]
        for feature in lag_features:
            for period in ma_periods:
                rolling = df.groupby('sector')[feature].rolling(window=period, min_periods=1)
                df[f'{feature}_ma_{period}'] = rolling.mean().reset_index(0, drop=True)
                df[f'{feature}_std_{period}'] = rolling.std().reset_index(0, drop=True)
                df[f'{feature}_min_{period}'] = rolling.min().reset_index(0, drop=True)
                df[f'{feature}_max_{period}'] = rolling.max().reset_index(0, drop=True)
                df[f'{feature}_median_{period}'] = rolling.median().reset_index(0, drop=True)
        
        # 创建增长率特征
        for feature in lag_features:
            df[f'{feature}_growth_rate'] = df.groupby('sector')[feature].pct_change()
            df[f'{feature}_yoy_growth'] = df.groupby('sector')[feature].pct_change(periods=12)
            df[f'{feature}_mom_growth'] = df.groupby('sector')[feature].pct_change(periods=1)
            df[f'{feature}_qoq_growth'] = df.groupby('sector')[feature].pct_change(periods=3)
        
        # 创建趋势特征（加速度）
        for feature in lag_features[:3]:  # 只对主要特征计算
            df[f'{feature}_acceleration'] = df.groupby('sector')[f'{feature}_growth_rate'].diff()
        
        # 创建相对位置特征（当前值在历史窗口中的分位数）
        for feature in lag_features[:3]:
            for period in [6, 12]:
                def calc_percentile(series):
                    if len(series) <= 1:
                        return 50.0
                    try:
                        current = series.iloc[-1]
                        return stats.percentileofscore(series.values, current)
                    except:
                        return 50.0
                
                rolling = df.groupby('sector')[feature].rolling(window=period, min_periods=1)
                df[f'{feature}_percentile_{period}'] = rolling.apply(calc_percentile).reset_index(0, drop=True)
        
        self.data['new_house_transactions'] = df
        
        print("滞后特征创建完成")
    
    def create_ratio_features(self):
        """创建比率特征"""
        print("\n创建比率特征...")
        
        df = self.data['new_house_transactions'].copy()
        
        # 价格相关比率
        df['price_per_area'] = df['price_new_house_transactions'] / (df['area_new_house_transactions'] + 1e-8)
        df['amount_per_transaction'] = df['amount_new_house_transactions'] / (df['num_new_house_transactions'] + 1e-8)
        df['area_per_transaction'] = df['area_new_house_transactions'] / (df['num_new_house_transactions'] + 1e-8)
        
        # 供需比率
        df['supply_demand_ratio'] = df['area_new_house_available_for_sale'] / (df['area_new_house_transactions'] + 1e-8)
        df['sell_through_rate'] = 1 / (df['period_new_house_sell_through'] + 1e-8)
        
        # 价格波动率
        df['price_volatility'] = df.groupby('sector')['price_new_house_transactions'].rolling(window=6, min_periods=1).std().reset_index(0, drop=True)
        
        self.data['new_house_transactions'] = df
        
        print("比率特征创建完成")
    
    def create_sector_features(self):
        """创建区域特征"""
        print("\n创建区域特征...")
        
        # 合并POI数据
        df = self.data['new_house_transactions'].copy()
        poi_df = self.data['sector_POI'].copy()
        
        # 重命名sector列以匹配
        poi_df['sector'] = poi_df['sector'].str.replace('sector ', 'sector ')
        
        # 合并数据
        df = df.merge(poi_df, on='sector', how='left')
        
        # 创建区域特征
        # 人口密度相关
        df['population_density'] = df['population_scale'] / (df['sector_coverage'] + 1e-8)
        df['resident_density'] = df['resident_population'] / (df['sector_coverage'] + 1e-8)
        
        # 商业密度
        df['commercial_density'] = df['commercial_area'] / (df['sector_coverage'] + 1e-8)
        df['shop_density'] = df['number_of_shops'] / (df['sector_coverage'] + 1e-8)
        
        # 交通便利性
        df['transportation_score'] = df['bus_station_cnt'] + df['subway_station_cnt'] * 2
        df['transportation_density'] = df['transportation_score'] / (df['sector_coverage'] + 1e-8)
        
        # 教育配套
        df['education_score'] = df['education'] * 3 + df['education_training_school_education_middle_school'] * 2 + df['education_training_school_education_primary_school']
        df['education_density'] = df['education_score'] / (df['sector_coverage'] + 1e-8)
        
        # 医疗配套
        df['medical_score'] = df['medical_health_general_hospital'] * 3 + df['medical_health_clinic'] * 2 + df['medical_health_specialty_hospital']
        df['medical_density'] = df['medical_score'] / (df['sector_coverage'] + 1e-8)
        
        # 周边房价影响
        df['price_attractiveness'] = df['surrounding_housing_average_price'] / (df['price_new_house_transactions'] + 1e-8)
        
        self.data['new_house_transactions'] = df
        
        print("区域特征创建完成")
    
    def create_market_features(self):
        """创建市场特征"""
        print("\n创建市场特征...")
        
        # 合并搜索指数数据
        df = self.data['new_house_transactions'].copy()
        search_df = self.data['city_search_index'].copy()
        
        # 处理搜索数据
        search_df['month'] = pd.to_datetime(search_df['month'])
        search_pivot = search_df.pivot_table(
            index='month', 
            columns=['keyword', 'source'], 
            values='search_volume', 
            fill_value=0
        )
        
        # 计算搜索指数特征
        search_features = {}
        for keyword in search_df['keyword'].unique():
            for source in search_df['source'].unique():
                col_name = f'search_{keyword}_{source}'
                if (keyword, source) in search_pivot.columns:
                    search_features[col_name] = search_pivot[(keyword, source)]
        
        # 合并搜索特征
        search_df_features = pd.DataFrame(search_features)
        search_df_features = search_df_features.reset_index()
        
        df['month'] = pd.to_datetime(df['month'])
        df = df.merge(search_df_features, on='month', how='left')
        
        # 创建搜索指数聚合特征
        search_cols = [col for col in df.columns if col.startswith('search_')]
        if search_cols:
            df['total_search_volume'] = df[search_cols].sum(axis=1)
            df['avg_search_volume'] = df[search_cols].mean(axis=1)
            df['search_volatility'] = df[search_cols].std(axis=1)
            
            # 买房相关搜索
            buy_house_cols = [col for col in search_cols if '买房' in col]
            if buy_house_cols:
                df['buy_house_search'] = df[buy_house_cols].sum(axis=1)
            
            # 房价相关搜索
            price_cols = [col for col in search_cols if '房价' in col]
            if price_cols:
                df['price_search'] = df[price_cols].sum(axis=1)
        
        self.data['new_house_transactions'] = df
        
        print("市场特征创建完成")
    
    def create_economic_features(self):
        """创建经济特征"""
        print("\n创建经济特征...")
        
        # 合并城市指标数据
        df = self.data['new_house_transactions'].copy()
        city_df = self.data['city_indexes'].copy()
        
        # 处理城市数据
        city_df['city_indicator_data_year'] = pd.to_datetime(city_df['city_indicator_data_year'], format='%Y')
        city_df['year'] = city_df['city_indicator_data_year'].dt.year
        
        # 为每个月份匹配对应的年份数据
        df['year'] = df['month'].dt.year
        df = df.merge(city_df, on='year', how='left')
        
        # 创建经济特征
        # GDP相关
        df['gdp_per_capita_ratio'] = df['gdp_per_capita_yuan'] / (df['per_capita_disposable_income_absolute_yuan'] + 1e-8)
        df['gdp_growth_rate'] = df.groupby('sector')['gdp_100m'].pct_change()
        
        # 人口相关
        df['population_growth_rate'] = df.groupby('sector')['year_end_registered_population_10k'].pct_change()
        df['urbanization_rate'] = df['year_end_urban_non_private_employees_10k'] / (df['year_end_total_employed_population_10k'] + 1e-8)
        
        # 收入相关
        df['income_growth_rate'] = df.groupby('sector')['per_capita_disposable_income_absolute_yuan'].pct_change()
        df['wage_income_ratio'] = df['annual_average_wage_urban_non_private_employees_yuan'] / (df['per_capita_disposable_income_absolute_yuan'] + 1e-8)
        
        # 财政相关
        df['fiscal_balance'] = df['general_public_budget_revenue_100m'] - df['general_public_budget_expenditure_100m']
        df['fiscal_balance_ratio'] = df['fiscal_balance'] / (df['general_public_budget_revenue_100m'] + 1e-8)
        
        self.data['new_house_transactions'] = df
        
        print("经济特征创建完成")
    
    def create_land_transaction_features(self):
        """创建土地交易特征"""
        print("\n创建土地交易特征...")
        
        df = self.data['new_house_transactions'].copy()
        land_df = self.data['land_transactions'].copy()
        
        # 处理土地交易数据
        land_df = land_df.sort_values(['sector', 'month'])
        
        # 创建土地交易的滞后特征（土地交易对未来新房供应有预测作用）
        land_features = ['transaction_amount', 'construction_area', 'planned_building_area', 'num_land_transactions']
        lag_periods = [3, 6, 12, 18, 24]  # 土地交易通常需要较长时间才能转化为新房供应
        
        for feature in land_features:
            for lag in lag_periods:
                land_df[f'{feature}_lag_{lag}'] = land_df.groupby('sector')[feature].shift(lag)
        
        # 创建土地交易的移动平均
        for feature in land_features:
            for period in [6, 12, 24]:
                rolling = land_df.groupby('sector')[feature].rolling(window=period, min_periods=1)
                land_df[f'{feature}_ma_{period}'] = rolling.mean().reset_index(0, drop=True)
                land_df[f'{feature}_sum_{period}'] = rolling.sum().reset_index(0, drop=True)
        
        # 合并土地交易特征
        land_cols = [col for col in land_df.columns if col not in ['month', 'sector']]
        df = df.merge(land_df[['month', 'sector'] + land_cols], on=['month', 'sector'], how='left')
        
        # 创建土地与新房的比率特征
        if 'transaction_amount' in df.columns:
            df['land_new_house_ratio'] = df['transaction_amount'] / (df['amount_new_house_transactions'] + 1e-8)
            df['land_price_per_area'] = df['transaction_amount'] / (df['construction_area'] + 1e-8)
        
        self.data['new_house_transactions'] = df
        
        print("土地交易特征创建完成")
    
    def create_pre_owned_house_features(self):
        """创建二手房交易特征"""
        print("\n创建二手房交易特征...")
        
        df = self.data['new_house_transactions'].copy()
        pre_owned_df = self.data['pre_owned_house_transactions'].copy()
        
        # 处理二手房数据
        pre_owned_df = pre_owned_df.sort_values(['sector', 'month'])
        
        # 创建二手房的滞后特征
        pre_owned_features = ['amount_pre_owned_house_transactions', 'price_pre_owned_house_transactions',
                             'area_pre_owned_house_transactions', 'num_pre_owned_house_transactions']
        lag_periods = [1, 2, 3, 6, 12]
        
        for feature in pre_owned_features:
            for lag in lag_periods:
                pre_owned_df[f'{feature}_lag_{lag}'] = pre_owned_df.groupby('sector')[feature].shift(lag)
        
        # 创建移动平均
        for feature in pre_owned_features:
            for period in [3, 6, 12]:
                rolling = pre_owned_df.groupby('sector')[feature].rolling(window=period, min_periods=1)
                pre_owned_df[f'{feature}_ma_{period}'] = rolling.mean().reset_index(0, drop=True)
        
        # 合并二手房特征
        pre_owned_cols = [col for col in pre_owned_df.columns if col not in ['month', 'sector']]
        df = df.merge(pre_owned_df[['month', 'sector'] + pre_owned_cols], on=['month', 'sector'], how='left')
        
        # 创建新房与二手房的比率特征
        if 'price_pre_owned_house_transactions' in df.columns:
            df['new_pre_owned_price_ratio'] = df['price_new_house_transactions'] / (df['price_pre_owned_house_transactions'] + 1e-8)
            df['new_pre_owned_amount_ratio'] = df['amount_new_house_transactions'] / (df['amount_pre_owned_house_transactions'] + 1e-8)
            df['new_pre_owned_area_ratio'] = df['area_new_house_transactions'] / (df['area_pre_owned_house_transactions'] + 1e-8)
        
        self.data['new_house_transactions'] = df
        
        print("二手房交易特征创建完成")
    
    def create_nearby_sector_features(self):
        """创建邻近区域特征"""
        print("\n创建邻近区域特征...")
        
        df = self.data['new_house_transactions'].copy()
        
        # 处理邻近区域新房交易数据
        nearby_new_df = self.data['new_house_transactions_nearby'].copy()
        nearby_new_df = nearby_new_df.sort_values(['sector', 'month'])
        
        # 创建邻近区域的滞后特征
        nearby_features = ['amount_new_house_transactions_nearby_sectors',
                          'price_new_house_transactions_nearby_sectors',
                          'area_new_house_transactions_nearby_sectors']
        lag_periods = [1, 2, 3, 6]
        
        for feature in nearby_features:
            for lag in lag_periods:
                nearby_new_df[f'{feature}_lag_{lag}'] = nearby_new_df.groupby('sector')[feature].shift(lag)
        
        # 创建移动平均
        for feature in nearby_features:
            for period in [3, 6, 12]:
                rolling = nearby_new_df.groupby('sector')[feature].rolling(window=period, min_periods=1)
                nearby_new_df[f'{feature}_ma_{period}'] = rolling.mean().reset_index(0, drop=True)
        
        # 合并邻近区域新房特征
        nearby_cols = [col for col in nearby_new_df.columns if col not in ['month', 'sector']]
        df = df.merge(nearby_new_df[['month', 'sector'] + nearby_cols], on=['month', 'sector'], how='left')
        
        # 创建本区域与邻近区域的比率特征
        if 'amount_new_house_transactions_nearby_sectors' in df.columns:
            df['local_nearby_amount_ratio'] = df['amount_new_house_transactions'] / (df['amount_new_house_transactions_nearby_sectors'] + 1e-8)
            df['local_nearby_price_ratio'] = df['price_new_house_transactions'] / (df['price_new_house_transactions_nearby_sectors'] + 1e-8)
            df['local_nearby_area_ratio'] = df['area_new_house_transactions'] / (df['area_new_house_transactions_nearby_sectors'] + 1e-8)
        
        # 处理邻近区域二手房数据
        nearby_pre_owned_df = self.data['pre_owned_house_transactions_nearby'].copy()
        nearby_pre_owned_df = nearby_pre_owned_df.sort_values(['sector', 'month'])
        
        # 创建邻近区域二手房的移动平均
        nearby_pre_owned_features = ['amount_pre_owned_house_transactions_nearby_sectors',
                                    'price_pre_owned_house_transactions_nearby_sectors']
        for feature in nearby_pre_owned_features:
            for period in [3, 6]:
                rolling = nearby_pre_owned_df.groupby('sector')[feature].rolling(window=period, min_periods=1)
                nearby_pre_owned_df[f'{feature}_ma_{period}'] = rolling.mean().reset_index(0, drop=True)
        
        # 合并邻近区域二手房特征
        nearby_pre_owned_cols = [col for col in nearby_pre_owned_df.columns if col not in ['month', 'sector']]
        df = df.merge(nearby_pre_owned_df[['month', 'sector'] + nearby_pre_owned_cols], 
                     on=['month', 'sector'], how='left')
        
        self.data['new_house_transactions'] = df
        
        print("邻近区域特征创建完成")
    
    def create_interaction_features(self):
        """创建交互特征（改进版）"""
        print("\n创建交互特征...")
        
        df = self.data['new_house_transactions'].copy()
        
        # 价格与面积的交互
        df['price_area_interaction'] = df['price_new_house_transactions'] * df['area_new_house_transactions']
        df['price_area_ratio_squared'] = (df['price_new_house_transactions'] / (df['area_new_house_transactions'] + 1e-8)) ** 2
        
        # 人口与价格的交互
        if 'population_scale' in df.columns:
            df['population_price_interaction'] = df['population_scale'] * df['price_new_house_transactions']
            df['population_area_interaction'] = df['population_scale'] * df['area_new_house_transactions']
        
        # 搜索量与价格的交互
        if 'total_search_volume' in df.columns:
            df['search_price_interaction'] = df['total_search_volume'] * df['price_new_house_transactions']
            df['search_amount_interaction'] = df['total_search_volume'] * df['amount_new_house_transactions']
        
        # 供需与价格的交互
        if 'supply_demand_ratio' in df.columns:
            df['supply_price_interaction'] = df['supply_demand_ratio'] * df['price_new_house_transactions']
            df['supply_amount_interaction'] = df['supply_demand_ratio'] * df['amount_new_house_transactions']
        
        # 新房与二手房的交互
        if 'price_pre_owned_house_transactions' in df.columns:
            df['new_pre_owned_price_diff'] = df['price_new_house_transactions'] - df['price_pre_owned_house_transactions']
            df['new_pre_owned_price_product'] = df['price_new_house_transactions'] * df['price_pre_owned_house_transactions']
        
        # 邻近区域交互
        if 'amount_new_house_transactions_nearby_sectors' in df.columns:
            df['local_nearby_amount_diff'] = df['amount_new_house_transactions'] - df['amount_new_house_transactions_nearby_sectors']
            df['local_nearby_price_diff'] = df['price_new_house_transactions'] - df['price_new_house_transactions_nearby_sectors']
        
        self.data['new_house_transactions'] = df
        
        print("交互特征创建完成")
    
    def handle_missing_values(self):
        """处理缺失值（改进版）"""
        print("\n处理缺失值...")
        
        df = self.data['new_house_transactions'].copy()
        
        # 按区域和时间排序
        df = df.sort_values(['sector', 'month'])
        
        # 数值型特征处理
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                # 对于时间序列特征，优先使用前向填充
                if any(keyword in col for keyword in ['lag', 'ma', 'growth', 'percentile']):
                    # 先按区域前向填充
                    df[col] = df.groupby('sector')[col].ffill()
                    # 再按区域后向填充
                    df[col] = df.groupby('sector')[col].bfill()
                    # 最后用中位数填充
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    # 其他特征先用组内中位数填充
                    df[col] = df.groupby('sector')[col].transform(lambda x: x.fillna(x.median()))
                    # 如果还有缺失，用全局中位数填充
                    df[col].fillna(df[col].median(), inplace=True)
        
        # 分类特征用众数填充
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else '', inplace=True)
        
        self.data['new_house_transactions'] = df
        
        missing_count = df.isnull().sum().sum()
        print(f"缺失值处理完成，剩余缺失值: {missing_count}")
    
    def feature_selection(self):
        """特征选择（改进版）"""
        print("\n进行特征选择...")
        
        df = self.data['new_house_transactions'].copy()
        
        # 准备特征和目标变量
        target_col = 'amount_new_house_transactions'
        exclude_cols = ['month', 'sector', target_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # 只保留数值型特征
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        # 移除包含无穷大或异常值的特征
        valid_features = []
        for feat in numeric_features:
            if df[feat].notna().sum() > 0:
                if not np.isinf(df[feat]).any():
                    valid_features.append(feat)
        
        X = df[valid_features].fillna(0)
        y = df[target_col]
        
        # 方法1: 随机森林特征重要性
        print("计算随机森林特征重要性...")
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_importance = pd.DataFrame({
            'feature': valid_features,
            'rf_importance': rf.feature_importances_
        })
        
        # 方法2: F统计量
        print("计算F统计量...")
        try:
            f_selector = SelectKBest(f_regression, k=min(200, len(valid_features)))
            f_selector.fit(X, y)
            f_importance = pd.DataFrame({
                'feature': valid_features,
                'f_score': f_selector.scores_
            })
        except:
            f_importance = pd.DataFrame({
                'feature': valid_features,
                'f_score': np.zeros(len(valid_features))
            })
        
        # 方法3: 互信息
        print("计算互信息...")
        try:
            mi_selector = SelectKBest(mutual_info_regression, k=min(200, len(valid_features)))
            mi_selector.fit(X, y)
            mi_importance = pd.DataFrame({
                'feature': valid_features,
                'mi_score': mi_selector.scores_
            })
        except:
            mi_importance = pd.DataFrame({
                'feature': valid_features,
                'mi_score': np.zeros(len(valid_features))
            })
        
        # 合并三种方法的结果
        feature_importance = rf_importance.merge(f_importance, on='feature').merge(mi_importance, on='feature')
        
        # 归一化各分数
        for col in ['rf_importance', 'f_score', 'mi_score']:
            if feature_importance[col].max() > 0:
                feature_importance[f'{col}_norm'] = (feature_importance[col] - feature_importance[col].min()) / \
                                                   (feature_importance[col].max() - feature_importance[col].min() + 1e-8)
            else:
                feature_importance[f'{col}_norm'] = 0
        
        # 计算综合得分（加权平均）
        feature_importance['combined_score'] = (
            feature_importance['rf_importance_norm'] * 0.5 +
            feature_importance['f_score_norm'] * 0.3 +
            feature_importance['mi_score_norm'] * 0.2
        )
        
        feature_importance = feature_importance.sort_values('combined_score', ascending=False)
        
        print("\n前30个最重要的特征:")
        print(feature_importance[['feature', 'rf_importance', 'f_score', 'mi_score', 'combined_score']].head(30))
        
        # 选择综合得分大于0.01的特征，或选择前100个特征
        threshold = max(0.01, feature_importance['combined_score'].quantile(0.3))
        selected_features = feature_importance[
            feature_importance['combined_score'] > threshold
        ]['feature'].tolist()
        
        # 如果选择的特征太少，至少选择前50个
        if len(selected_features) < 50:
            selected_features = feature_importance.head(50)['feature'].tolist()
        
        print(f"\n选择了 {len(selected_features)} 个特征")
        
        return selected_features, feature_importance
    
    def create_final_dataset(self, selected_features):
        """创建最终数据集"""
        print("\n创建最终数据集...")
        
        df = self.data['new_house_transactions'].copy()
        
        # 选择特征
        feature_cols = ['month', 'sector'] + selected_features + ['amount_new_house_transactions']
        final_df = df[feature_cols].copy()
        
        # 处理时间列
        final_df['month'] = pd.to_datetime(final_df['month'])
        
        # 处理区域列
        final_df['sector_num'] = final_df['sector'].str.extract('(\d+)').astype(int)
        
        # 保存最终数据集
        final_df.to_csv('final_dataset.csv', index=False)
        
        print(f"最终数据集已保存，形状: {final_df.shape}")
        
        return final_df
    
    def run_feature_engineering(self):
        """运行完整的特征工程流程（改进版）"""
        print("=" * 60)
        print("开始特征工程（改进版）...")
        print("=" * 60)
        
        # 加载数据
        self.load_data()
        
        # 创建各种特征
        self.create_temporal_features()
        self.create_lag_features()
        self.create_ratio_features()
        self.create_sector_features()
        self.create_market_features()
        self.create_economic_features()
        
        # 新增特征
        self.create_land_transaction_features()
        self.create_pre_owned_house_features()
        self.create_nearby_sector_features()
        self.create_interaction_features()
        
        # 处理缺失值
        self.handle_missing_values()
        
        # 特征选择
        selected_features, feature_importance = self.feature_selection()
        
        # 创建最终数据集
        final_df = self.create_final_dataset(selected_features)
        
        print("\n" + "=" * 60)
        print("特征工程完成！")
        print(f"最终特征数量: {len(selected_features)}")
        print(f"数据集形状: {final_df.shape}")
        print("=" * 60)
        
        return final_df, selected_features, feature_importance

if __name__ == "__main__":
    # 创建特征工程器实例
    engineer = FeatureEngineer()
    
    # 运行特征工程
    final_df, selected_features, feature_importance = engineer.run_feature_engineering()
