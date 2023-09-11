# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import copy
from WindPy import w
from datetime import *
import datetime as dt
import pandas.tseries as pt
import time 


w.start()


class TrackingError:

    def __init__(self, names_sec, name_index, sec_com, type_data):
        
        """
        :param names_sec: a list of strs which are the sec codes of interest
        :param name_index: this is a string which is the index in interest
        :param sec_com: df that explains what firm runs
        """
        # before any comparison we specify the index and securities we choose from
        self.td = type_data
        self.secs = names_sec # the secs we need to compare 
        self.ind = name_index # the index of interest
        self.sec_com = sec_com # the company and security function
        self.etf_evaluation() # the function runs auto if put in init
        

    def trading_days(self):
        
        "this function returns all the trading days in most recent one year"
        today = dt.date.today()
        startd = w.tdaysoffset(-252, dt.date.today()).Data[0]
        wl = pd.DataFrame(w.tdays(startd[0], today).Data[0])
        wkk = pd.Series(wl.iloc[:, 0]).to_list()
        return wkk
        
    def datetime_to_str(dt_obj):
        
        format = '%b %d %Y %I:%M%p'
        datetime_str = dt.datetime.strptime(dt_obj, format)
        return datetime_str
    
    def com_dict(self, name_of_sec):
        # this is how we construct the 'sec-code-to-chinese-name' dictionary
        list1 = self.sec_com['security'].to_list()
        list2 = self.sec_com['company'].to_list()
        dk = dict(zip(list1, list2))
        # we use the dictionary to construct the fund company name list in chinese
        return dk[name_of_sec]
        
    def get_data(self):
        
        wll = self.trading_days()
        type_data = self.td
        wl_type = ['m', '10m']
        # the three types are 
        # minute data from the latest 1 month 
        # 10m data from the latest 3 months
        # daily data from the latest 5y
        company_list = []
        for sec in self.secs:
            company_list.append(self.com_dict(sec))
        
        if type_data not in set(wl_type):
            print('invalid type')
            return 0
        
        if type_data == 'm':
            tdl = self.trading_days()
            msting = dt.datetime.strptime('1500', '%H%M').time()
            
            daylist = wll[-22]
            starting = dt.datetime.combine(daylist, 
                                                 msting)
            dfn = pd.DataFrame(w.wsi(self.ind, 
                                      "close", 
                                      starting,
                                      dt.datetime.now(),    
                                      usedf=True)[1])

        # then the ETFs
            for i in self.secs:
                dfn = pd.concat([dfn, 
                                 pd.DataFrame(w.wsi(i, "close",
                                                    starting,
                                                    dt.datetime.now(), 
                                                    usedf=True)[1])], 
                                              axis=1)
        if type_data == '10m':
            
            msting = dt.datetime.strptime('1500', '%H%M').time()
            daylist = wll[-64]
            starting = dt.datetime.combine(daylist, 
                                                 msting)
            
            dfn = pd.DataFrame(w.wsi(self.ind, 
                                      "close", 
                                      starting,
                                      dt.datetime.now(), 
                                      usedf=True)[1])
            # then the ETFs
            for i in self.secs:
                dfn = pd.concat([dfn, 
                                 pd.DataFrame(w.wsi(i, 
                                                    "close",
                                                    starting,
                                                    dt.datetime.now(), 
                                                    usedf=True)[1])], 
                                                    axis=1)
        
            dfn = dfn.iloc[np.arange(0, len(dfn.index), 10), :]
            
        dfn.columns = [self.com_dict(self.ind)] + company_list
        dfn = dfn.ffill(axis=0)
        dfn = dfn.dropna()
        return dfn
            
    def etf_evaluation(self): 
    
        print('a new run of the evaluation')
        wll = self.trading_days()
        company_list = []
        for sec in self.secs:
            company_list.append(self.com_dict(sec))
            
        dfn = self.get_data()
        dfn.to_csv('this is dfn.csv')
        dfd = copy.deepcopy(dfn)
        
        daylist = wll[-120]
        starting = daylist.date()
        dfbs = pd.DataFrame(w.wsd(self.ind, 
                                  "close", 
                                  starting,
                                  dt.date.today(), 
                                  usedf=True)[1])
        for i in self.secs:
            dfbs = pd.concat([dfbs, 
                             pd.DataFrame(w.wsd(i, 
                                                "close",
                                                starting,
                                                dt.date.today(), 
                                                usedf=True)[1])], 
                                          axis=1)

        dfbs.columns = [self.com_dict(self.ind)] + company_list
        dfbs = dfbs.ffill(axis=0)
        dfbs = dfbs.dropna()
        cal_ind = 0
    
        for tspot in dfd.index:
            
            if tspot.date() > dfd.index[0].date() + timedelta(days=1):
                # mytime = dt.datetime.strptime('1500', '%H%M').time()
                day_before = wll[wll.index(tspot.date()) -1]
                d1 = day_before.day
                m1 = day_before.month
                y1 = day_before.year
                ndb = dt.datetime(y1, m1, d1)
                print('the previous business day', ndb)
                print('the time now', tspot)
                dfd.loc[tspot, :] = np.log(dfn.loc[tspot, :] / dfbs.loc[ndb.date(), :])
                
        dfd = dfd.loc[dfd.index.date > dfd.index[0].date() + timedelta(days=1), :]
        # now the dataframe df_error is going to report the values of the error
        colname = ['基金公司', '测度一', '测度二', '测度三', '测度四', '测度五']
        
        # 测度五 = largest difference in the return difference 
        
        df_error = pd.DataFrame(columns=colname)
        df_error['基金公司'] = pd.Series(company_list)
        df_error.set_index('基金公司')
        for i in np.arange(1, (len(company_list) + 1), 1):
            df_error.loc[i - 1, '测度一'] = np.mean(abs(dfd.iloc[:, i] - dfd.iloc[:, 0]))
            df_error.loc[i - 1, '测度二'] = np.sqrt(np.mean(np.square(dfd.iloc[:, i] - dfd.iloc[:, 0])))
            df_error.loc[i - 1, '测度三'] = abs(
                1 - np.cov(dfd.iloc[:, i], dfd.iloc[:, 0])[0][1] / np.var(dfd.iloc[:, 0]))
            df_error.loc[i - 1, '测度四'] = np.std(dfd.iloc[:, i] - dfd.iloc[:, 0])
            df_error.loc[i - 1, '测度五'] = np.max(abs(dfd.iloc[:, i] - dfd.iloc[:, 0]))

        dfk = copy.deepcopy(df_error)
        dfk = dfk.iloc[:, :-2]
        dfk.columns = ['基金公司', '条件概率一', '条件概率二', '综合概率']

        for i in np.arange(1, (len(company_list) + 1), 1):

            # a1 = higher when positive return
            # b1 = lower when negative return
            # a2 = positive index return
            # b2 = negative index return

            a1 = sum(((dfd.iloc[:, i] > dfd.iloc[:, 0]) & (dfd.iloc[:, 0] > 0)).to_list())
            b1 = sum(((dfd.iloc[:, i] < dfd.iloc[:, 0]) & (dfd.iloc[:, 0] < 0)).to_list())
            a2 = sum((dfd.iloc[:, 0] > 0).to_list())
            b2 = sum((dfd.iloc[:, 0] < 0).to_list())

            print('a1 =', a1)
            print('b1 =', b1)
            print('a2 =', a2)
            print('b2 =', b2)

            dfk.loc[i - 1, '条件概率一'] = a1 / a2  # 条件概率一: 指数上涨时ETF上涨且幅度更大的条件概率
            dfk.loc[i - 1, '条件概率二'] = b1 / b2  # 条件概率二: 指数下降时ETF下降且幅度更大的条件概率
            dfk.loc[i - 1, '综合概率'] = (a1 + b1) / (a2 + b2)  # 综合概率: 指数上涨或者下降时ETF同向波动的概率
        
        msting = dt.datetime.strptime('1500', '%H%M').time()
        daylist = wll[-22]
        starting = dt.datetime.combine(daylist, 
                                                 msting)
            
        dfv = pd.DataFrame(w.wsi(self.ind,
                                 "amt", 
                                 starting,
                                 dt.datetime.now(), 
                                 usedf=True)[1])
        for i in self.secs:
            dfv = pd.concat([dfv, pd.DataFrame(w.wsi(i, 
                                                     "amt", 
                                                     starting,
                                                     dt.datetime.now(),
                                                     usedf=True)[1])], axis=1)
        
        
        
        if self.td == '10m':
            
            msting = dt.datetime.strptime('1500', '%H%M').time()
            daylist = wll[-64]
            starting = dt.datetime.combine(daylist, 
                                                 msting)
            
            dfv = pd.DataFrame(w.wsi(self.ind, 
                                      "amt", 
                                      starting,
                                      dt.datetime.now(), 
                                      usedf=True)[1])
            # then the ETFs
            for i in self.secs:
                dfv = pd.concat([dfv, 
                                 pd.DataFrame(w.wsi(i, 
                                                    "amt",
                                                    starting,
                                                    dt.datetime.now(), 
                                                    usedf=True)[1])], 
                                                    axis=1)
        
            dfv = dfv.iloc[np.arange(0, len(dfv.index), 10), :]
            
        dfv = dfv.fillna(0)
        dfv = dfv.loc[dfv.index.date > dfd.index[0].date() + timedelta(days=1), :]
        
        spread_series = [self.new_spread(target_volume=300000, code_sec=sename) 
                        for sename in self.secs]
        
        vol_series = dfv.mean().to_list()
        df_combine = copy.deepcopy(df_error)
        df_combine['成交量'] = vol_series[1:] # the first is the one with index 
        df_combine['价差'] = spread_series
        
        print(df_combine.head())
        k1 = pd.concat([df_combine.iloc[:, 1:6].rank(), df_combine['成交量'].rank(ascending=False)], axis=1)
        k1 = pd.concat([k1, df_combine['价差'].rank()], axis=1)
        k1['IEI'] = pd.Series([0] * len(k1.index))
        
        iei_weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.25, 0.25]
        for i in range(len(k1.index)):
            k1.iloc[i, -1] = sum([k1.iloc[i, j] * iei_weights[j] for j in range(7)])
        k1.insert(loc=0, column='基金公司', value=pd.Series(company_list))

        dfk.to_csv('results/ prob_ind={}  freq={}.csv'.format(self.com_dict(self.ind), self.td))
        p1, p2, p3 = dfk['条件概率一'].mean(), dfk['条件概率二'].mean(), dfk['综合概率'].mean()
        df_combine.to_csv('results/ value_ind={} freq={}.csv'.format(self.com_dict(self.ind), self.td))
        k1.to_csv('results/ rank_ind={} freq={}.csv'.format(self.com_dict(self.ind), self.td))
        file1 = open("info_doc.txt", "a")
        L = ["the lastest run at{}\n".format(dt.datetime.now), 
             "the index is{}\n".format(self.ind), 
             'the data frequency = '.format(self.td),
             "which starts from{}".format(dfd.index[0]),
             "and it ends at{}\n".format(dfd.index[-1]), 
             "average up prob = {}\n".format(p1), 
             "average down prob = {}\n".format(p2), 
             "overall prob = {}\n".format(p3)]
        file1.writelines(L)
        file1.close()
        
    def new_spread(self, target_volume=300000, code_sec='510300.SH'):
        # the gauge for bid ask spread per a new way of calculation 
        # proposed by Lyxor Asset Management
        # given the target volume and the most recent 7 day tick data
        """
        target_volume: target volume
        code_sec: the sec of interest 
        """
        fields = 'bid1, bid2, bid3, bid4, bid5, \
                        ask1, ask2, ask3, ask4, ask5, \
                            bsize1, bsize2, bsize3, bsize4, bsize5, \
                                asize1, asize2, asize3, asize4, asize5'
                                
        wll = self.trading_days()
        msting = dt.datetime.strptime('0930', '%H%M').time()
        sday = wll[-7]
        starting = dt.datetime.combine(sday, msting)
            
        mc_tsdf = pd.DataFrame(w.wst(code_sec, 
                                      fields, 
                                      starting,
                                      dt.datetime.now(), 
                                      usedf=True)[1])
        
        whl = []
        ka = 0
        print('start of a new calculation of new_spread')
        for indx in range(len(mc_tsdf.index)):
            
            ka += 1
            print(ka)
            
            wl_bid = (mc_tsdf.iloc[indx, 10:15]).to_list()
    
            new_l_bid = [np.max([indx, np.min([wl_bid[i], 
                                        target_volume - np.sum(wl_bid[:i])])]) 
                         for i in np.arange(5)]
        
            wl_ask = (mc_tsdf.iloc[indx, 15:20]).to_list()

            new_l_ask = [np.max([0, np.min([wl_ask[i], 
                                        target_volume - np.sum(wl_ask[:i])])]) 
                         for i in np.arange(5)]
        
            avg_bid = np.sum(mc_tsdf.iloc[indx, 0:5] * new_l_bid) / np.sum(new_l_bid)
            avg_ask = np.sum(mc_tsdf.iloc[indx, 5:10] * new_l_ask) / np.sum(new_l_ask)
            coef_c = np.max([1, target_volume / np.min([np.sum(wl_ask), np.sum(wl_bid)])])
            new_spread = (avg_ask - avg_bid) / ((avg_ask + avg_bid) * 0.5) * coef_c
        
            whl.append(new_spread * 10000)
            
        
        wh = pd.Series(whl)
        wh = wh.dropna()
        print('the mean is {} bps'.format(wh.mean()))
        return wh.mean()
    


if __name__ == '__main__':
    sc1 = pd.read_excel('sec_com.xlsx')
    a1 = '159845.SZ'
    b1 = '512100.SH'
    c1 = '159629.SZ'
    d1 = '159633.SZ'
    e1 = '516300.SH'
    
    a2 = '510500.SH'
    b2 = '159922.SZ'
    c2 = '512500.SH'
    d2 = '510510.SH'
    e2 = '510580.SH'
    f2 = '512510.SH'
    
    a3 = '510300.SH'
    b3 = '510330.SH'
    c3 = '510310.SH'
    d3 = '515330.SH'
    e3 = '159925.SZ'
    
    a4 = '588000.SH'
    b4 = '588080.SH'
    c4 = '588050.SH'
    d4 = '588090.SH'
    e4 = '588150.SH'
    
    names_sec1000 = [a1, b1, c1, d1, e1]
    names_sec500 = [a2, b2, c2, d2, e2, f2]
    names_sec300 = [a3, b3, c3, d3, e3]
    names_sec50 = [a4, b4, c4, d4, e4]
    
    name_index1000 = '000852.SH'
    name_index500 = '000905.SH'
    name_index300 = '000300.SH'
    name_index50 =  '000688.SH'
    
    sta = time.time()
    TrackingError(names_sec1000, name_index1000, sc1, 'm')
    TrackingError(names_sec500, name_index500, sc1, 'm')
    TrackingError(names_sec300, name_index300, sc1, 'm')
    TrackingError(names_sec50, name_index50, sc1, 'm')
    
    TrackingError(names_sec1000, name_index1000, sc1, '10m')
    TrackingError(names_sec500, name_index500, sc1, '10m')
    TrackingError(names_sec300, name_index300, sc1, '10m')
    TrackingError(names_sec50, name_index50, sc1, '10m')
    
    eed = time.time()
    print(' the length of running time is {} seconds'.format(eed-sta))
    
    
    











