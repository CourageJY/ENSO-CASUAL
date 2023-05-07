import pandas as pd
# 定义一组数据
df = pd.DataFrame({'num_legs': [2, 4, 8, 0],
                   'num_wings': [2, 0, 0, 0],
                   'num_specimen_seen': [10, 2, 1, 8]},
                  index=['falcon', 'dog', 'spider', 'fish'])
print(df)

#random_state（类似于random库中随机种子的作用）确保示例的可复现性
extract = df['num_legs'].sample(n=3, random_state=1)
print(extract)

#replace=True时表示有放回抽样
extract2 = df.sample(frac=0.5, replace=True, random_state=1)
print(extract2)

#frac=0.5表示随机抽取50%的数据
extract2 = df.sample(frac=0.5, replace=True, random_state=1)
print(extract2)

#当frac>1时必须设置replace=True
extract3 = df.sample(frac=2, replace=True, random_state=1)
print(extract3)

#权重抽样
extract4 = df.sample(n=2, weights='num_specimen_seen', random_state=1)
print(extract4)