import pandas as pd
import datetime

current_date = datetime.date.today( )
formatted_date = current_date.strftime("%m_%d_")
# 读取csv文件
df = pd.read_csv(f"metricss/{formatted_date}metrics.csv")

# 计算某一列的平均值
import pdb
pdb.set_trace()
mean_value = df['SSIM'][-27:].astype(float).mean( )

print('平均值为：', mean_value)
