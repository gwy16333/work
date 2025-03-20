import inspect
from statsmodels.tsa.arima.model import ARIMA

# 打印 ARIMA 类的 fit 方法的参数签名
print(inspect.signature(ARIMA.fit))
