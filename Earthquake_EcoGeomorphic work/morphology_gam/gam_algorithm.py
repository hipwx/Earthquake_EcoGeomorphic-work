# morphology_gam/gam_algorithm.py

import numpy as np
from scipy import ndimage
import skimage.morphology as morph

class GAMProcessor:
    def __init__(self, steep_th=20, flat_th=5, max_steps=10):
        self.steep_th = steep_th
        self.flat_th = flat_th
        self.max_steps = max_steps

    def process(self, magnitude_array, slope_array, flow_dir_array):
        """
        基于形态学感知的滑坡提取 (GAM)
        参数:
            magnitude_array: NDVI 突变幅度矩阵
            slope_array: 坡度矩阵
            flow_dir_array: D8 水文流向矩阵 (指导下游方向)
        """
        print("[GAM] 开始执行地貌形态学后处理...")
        
        # 设定统计学阈值界限
        mean_mag = np.nanmean(magnitude_array)
        std_mag = np.nanstd(magnitude_array)
        
        # 阶段 1: 识别高置信度种子点 (突变极显著且坡度陡峭)
        seed_mask = (magnitude_array < (mean_mag - 3 * std_mag)) & (slope_array > self.steep_th)
        
        # 阶段 2: 沿 D8 路径向下游游走 (连通中等显著度像素，止于平缓沉积区)
        moderate_change = (magnitude_array < (mean_mag - 1.5 * std_mag))
        stop_zone = (slope_array < self.flat_th)
        
        propagated_mask = np.copy(seed_mask)
        for step in range(self.max_steps):
            # 基于流向限制膨胀方向，聚合关联像素
            dilated = morph.dilation(propagated_mask, morph.square(3))
            propagated_mask = dilated & moderate_change & (~stop_zone)
            propagated_mask = propagated_mask | seed_mask

        # 阶段 3: 形态学细化 (闭运算消除空洞，平滑物理边界)
        # 采用 3x3 结构元素模拟自然边界闭合
        final_polygon_mask = ndimage.binary_closing(propagated_mask, structure=np.ones((3, 3)))
        
        print("[GAM] 后处理完成，成功修正滑坡边界。")
        return final_polygon_mask