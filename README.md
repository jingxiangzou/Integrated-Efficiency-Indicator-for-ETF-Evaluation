# Integrated-Efficiency-Indicator-for-ETF-Evaluation

This is a research project I conducted which aims to find the optimal ETF w.r.t. a equity market index taking into consideration liquidity and tracking errors 

# Contents

本文件夹包含项目的全部文件：

1. README
2. choosing the optimal ETF.py
3. results文件夹

results文件夹中有24个.xslx文件：
它们是文件2的运行结果
两个频率（分钟频&近一个月 及 十分钟频&近三个月）， 各对应12个文件
每个频率有4个指数，每个指数对应三个文件，分别计算
1. 指标的值
2. 指标的排名和IEI得分
3. 三个概率

文件2实现了基于wind的自动计算，每次运行均使用最新的数据。运行文件2需要安装wind并修复接口。
