# vegetation_recovery/logistic_growth.py

import numpy as np
from scipy.optimize import curve_fit

class RecoveryModeler:
    def __init__(self, pre_ndvi, target_ratio):
        self.pre_ndvi = pre_ndvi
        self.target_ratio = target_ratio
        
    def _logistic_function(self, t, K, r, t0, C):
        """逻辑斯蒂生长方程"""
        return (K / (1 + np.exp(-r * (t - t0)))) + C

    def fit_and_calculate_time(self, time_array, ndvi_array):
        """
        对 NDVI 时间序列进行非线性拟合，计算达标恢复时长
        """
        print("[RECOVERY] 正在拟合植被逻辑斯蒂恢复轨迹...")
        
        # 初始化参数：[最大恢复容量, 恢复速率, 拐点时间, 本底常数]
        initial_guess = [0.8, 0.5, time_array[0] + 2, 0.1]
        
        try:
            popt, _ = curve_fit(
                self._logistic_function, 
                time_array, 
                ndvi_array, 
                p0=initial_guess, 
                maxfev=10000
            )
            
            K, r, t0, C = popt
            target_ndvi = self.target_ratio * self.pre_ndvi
            
            # 依据方程求逆解算达标年份
            target_t = t0 - (1 / r) * np.log((K / (target_ndvi - C)) - 1)
            recovery_years = target_t - time_array[0]
            
            print(f"[RECOVERY] 拟合成功。预估恢复时间: {recovery_years:.2f} 年。")
            return popt, recovery_years
            
        except RuntimeError:
            print("[RECOVERY] 序列波动过大，模型拟合未收敛。")
            return None, None