# topography/roughness_calc.py

import numpy as np
from scipy import ndimage

class TopographyAnalyzer:
    def __init__(self, window_size=3):
        self.window_size = window_size

    def calculate_roughness(self, pre_dem_array, post_dem_array):
        """
        提取高程变化并计算表面特征粗糙度
        """
        print("[TOPOGRAPHY] 开始计算 DEM 差值与表面特征粗糙度...")
        
        # 1. 深度积分的物质位移 (高程变化量)
        elevation_change = post_dem_array - pre_dem_array
        
        # 2. 表面粗糙度提取 (利用邻域移动窗口的标准差表征)
        def local_std(window):
            return np.nanstd(window)
            
        roughness = ndimage.generic_filter(
            elevation_change, 
            local_std, 
            size=self.window_size
        )
        
        print("[TOPOGRAPHY] 特征粗糙度空间分布计算完毕。")
        return elevation_change, roughness