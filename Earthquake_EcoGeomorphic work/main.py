# main.py

import numpy as np
import config
from gee_ccdc.ccdc_processor import CCDCProcessor
from morphology_gam.gam_algorithm import GAMProcessor
from topography.roughness_calc import TopographyAnalyzer
from vegetation_recovery.logistic_growth import RecoveryModeler

def main():
    print("=== 震后生态地貌恢复动态分析系统启动 ===")

    # ---------------------------------------------------------
    # 步骤 1: 运行长时序 CCDC 突变检测
    # ---------------------------------------------------------
    ccdc_engine = CCDCProcessor(
        bbox=config.ROI_BBOX,
        start_date=config.START_YEAR,
        end_date=config.END_YEAR
    )
    # ccdc_results = ccdc_engine.run_ccdc()
    # ccdc_engine.export_break_results(ccdc_results, 'CCDC_Earthquake_Outputs')

    # ---------------------------------------------------------
    # 步骤 2: 读取本地栅格数据，执行 GAM 形态学修正
    # ---------------------------------------------------------
    gam_engine = GAMProcessor(
        steep_th=config.GAM_STEEP_SLOPE_THRESHOLD, 
        flat_th=config.GAM_FLAT_SLOPE_THRESHOLD, 
        max_steps=config.GAM_MAX_PROPAGATION_STEPS
    )
    
    # 从本地实际导出的 GeoTIFF 文件中读取矩阵数组 (此处以 numpy array 作为输入载体)
    # 实际应用中，通常使用 rasterio.open('filepath.tif').read(1) 获取
    magnitude_data = np.load('data_array/magnitude.npy') 
    slope_data = np.load('data_array/slope.npy')
    flow_dir_data = np.load('data_array/flow_direction.npy')
    
    landslide_mask = gam_engine.process(magnitude_data, slope_data, flow_dir_data)

    # ---------------------------------------------------------
    # 步骤 3: 震前后高精度 DEM 差分及粗糙度提取
    # ---------------------------------------------------------
    topo_engine = TopographyAnalyzer(window_size=3)
    
    # 读取震前、震后的 DEM 高程矩阵
    pre_dem_data = np.load('data_array/pre_earthquake_dem.npy')
    post_dem_data = np.load('data_array/post_earthquake_dem.npy')
    
    elev_change, roughness = topo_engine.calculate_roughness(pre_dem_data, post_dem_data)

    # ---------------------------------------------------------
    # 步骤 4: 提取滑坡面时序序列，计算植被恢复时间
    # ---------------------------------------------------------
    recovery_engine = RecoveryModeler(
        pre_ndvi=config.PRE_EARTHQUAKE_NDVI, 
        target_ratio=config.RECOVERY_TARGET_RATIO
    )
    
    # 提取提取单个或平均滑坡体的时间轴与 NDVI 值
    time_series = np.load('data_array/temporal_years.npy')
    ndvi_series = np.load('data_array/temporal_ndvi_series.npy')
    
    params, time_to_recover = recovery_engine.fit_and_calculate_time(time_series, ndvi_series)

    print("=== 全流程分析执行完毕 ===")

if __name__ == "__main__":
    main()