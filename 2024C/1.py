import pandas as pd
import numpy as np

print("开始执行农作物种植策略优化模型 - 第一题解决方案...")

# ====================1.数据加载与预处理======================
def load_data():
    """加载所有必要的数据"""
    try:
        # 读取地块信息
        land_info = pd.read_csv("001.csv", encoding='utf-8')
        crop_info = pd.read_csv("002.csv", encoding='utf-8')
        planting_2023 = pd.read_csv("003.csv", encoding='utf-8')
        economic_data = pd.read_csv("004.csv", encoding='utf-8')
        expected_sales = pd.read_excel("ExpectedSales.xlsx")
    except UnicodeDecodeError: 
        land_info = pd.read_csv("001.csv", encoding='gbk')
        crop_info = pd.read_csv("002.csv", encoding='gbk')
        planting_2023 = pd.read_csv("003.csv", encoding='gbk')
        economic_data = pd.read_csv("004.csv", encoding='gbk')
        expected_sales = pd.read_excel("ExpectedSales.xlsx")
    
    # 清理列名
    land_info.columns = ['地块名称', '地块类型', '地块面积/亩', '说明']
    crop_info.columns = ['编号', '作物名称', '作物类型', '种植地块', '说明']
    planting_2023.columns = ['种植地块', '编号', '作物名称', '作物类型', '种植面积/亩', '种植季次']
    economic_data.columns = ['编号', '作物编号', '作物名称', '地块类型', '种植季次', '亩产量/斤', '种植成本/(元/亩)', '销售单价/(元/斤)']
    
    # 处理经济数据
    price_split = economic_data['销售单价/(元/斤)'].str.split('-', expand=True)
    price_split[0] = pd.to_numeric(price_split[0], errors='coerce')
    price_split[1] = pd.to_numeric(price_split[1], errors='coerce').fillna(price_split[0])
    economic_data['平均单价'] = (price_split[0] + price_split[1]) / 2
    economic_data['每亩净收益'] = economic_data['亩产量/斤'] * economic_data['平均单价'] - economic_data['种植成本/(元/亩)']
    
    # 处理预期销售量数据，构建CROP_DEMANDS字典
    global CROP_DEMANDS
    CROP_DEMANDS = {}
    try:
        if '作物名称' in expected_sales.columns:
            if '预期销售量/斤' in expected_sales.columns:
                col = '预期销售量/斤'
            elif '预期销售量' in expected_sales.columns:
                col = '预期销售量'
            elif '销售量' in expected_sales.columns:
                col = '销售量'
            else:
                raise ValueError(f"ExpectedSales.xlsx的列名不符合预期: {expected_sales.columns.tolist()}")
            
            for _, row in expected_sales.iterrows():
                crop_name = str(row['作物名称']).strip()
                demand = pd.to_numeric(row[col], errors='coerce')
                if pd.notna(demand) and crop_name != 'nan':
                    CROP_DEMANDS[crop_name] = demand
        else:
            raise ValueError(f"ExpectedSales.xlsx缺少必要的列: {expected_sales.columns.tolist()}")
    except Exception as e:
        raise RuntimeError(f"读取预期销售量数据失败，请检查文件格式！错误信息: {e}")
    
    print(f"成功加载了 {len(CROP_DEMANDS)} 种作物的需求量数据")
    
    return land_info, crop_info, planting_2023, economic_data

# ====================2.基础约束和规则定义======================
# 定义特殊作物组
BEAN_CROPS = ['黄豆', '黑豆', '红豆', '绿豆', '爬豆', '豇豇豆', '刀豆', '芸豆']
SEASON2_VEG = ['大白菜', '白萝卜', '红萝卜']
FUNGI_CROPS = ['榆黄菇', '香菇', '白灵菇', '羊肚菌']

# 作物需求量将在load_data函数中从ExpectedSales.xlsx文件读取
CROP_DEMANDS = {}

# 定义地块类型与作物匹配规则
PLANT_RULES = {
    '平旱地': {'季节': ['单季'], '类型': ['粮食']},
    '梯田': {'季节': ['单季'], '类型': ['粮食']},
    '山坡地': {'季节': ['单季'], '类型': ['粮食']},
    '水浇地': {'季节': ['单季', '第一季', '第二季'], '类型': ['粮食', '蔬菜']},
    '普通大棚': {'季节': ['第一季', '第二季'], '类型': ['蔬菜', '食用菌']},
    '智慧大棚': {'季节': ['第一季', '第二季'], '类型': ['蔬菜']}
}

# ====================3.核心决策函数======================
def find_best_planting(plot_name, year, history, economic_data, land_info, scenario='waste'):
    """为单个地块做出种植决策"""
    # 获取地块信息
    plot_info = land_info[land_info['地块名称'] == plot_name]
    if plot_info.empty:
        return {'plan': [], 'profit': 0}
    
    plot_type = plot_info.iloc[0]['地块类型']
    area = plot_info.iloc[0]['地块面积/亩']

    # 获取历史种植记录 - 修改为按季次存储
    last_year_season1 = history.get(year - 1, {}).get(plot_name, {}).get('第一季', [])
    last_year_season2 = history.get(year - 1, {}).get(plot_name, {}).get('第二季', [])
    current_year_season1 = history.get(year, {}).get(plot_name, {}).get('第一季', [])
    
    # 检查豆类约束（三年内至少种植一次豆类）
    # 获取前两年的所有季次种植记录
    year_before_season1 = history.get(year - 2, {}).get(plot_name, {}).get('第一季', [])
    year_before_season2 = history.get(year - 2, {}).get(plot_name, {}).get('第二季', [])
    need_bean = not (any(c in BEAN_CROPS for c in last_year_season1 + last_year_season2 + year_before_season1 + year_before_season2))

    # 分地块类型处理
    if plot_type in ['平旱地', '梯田', '山坡地']:
        return _handle_single_season(plot_name, plot_type, area, last_year_season2, need_bean, economic_data, scenario)
    elif plot_type == '水浇地':
        return _handle_water_land(plot_name, area, last_year_season2, current_year_season1, need_bean, economic_data, scenario)
    elif plot_type == '普通大棚 ':  # 注意：实际数据中有空格
        return _handle_greenhouse(plot_name, area, last_year_season2, current_year_season1, need_bean, economic_data, is_smart=False, scenario=scenario)
    elif plot_type == '智慧大棚':
        return _handle_greenhouse(plot_name, area, last_year_season2, current_year_season1, need_bean, economic_data, is_smart=True, scenario=scenario)
    return {'plan': [], 'profit': 0}

def calculate_profit(crop_row, area, scenario, demand):
    """根据情景计算某作物在某地块的净收益"""
    base_yield = crop_row['亩产量/斤']
    base_cost = crop_row['种植成本/(元/亩)']
    base_price = crop_row['平均单价']

    if base_yield <= demand:
        profit = base_yield * base_price - base_cost
    else:
        if scenario == 'discount':
            profit = demand * base_price + (base_yield - demand) * base_price * 0.5 - base_cost
        else:  # waste 情况
            profit = demand * base_price - base_cost  # 超过部分浪费，不计收益
    
    return profit * area

def _handle_single_season(plot_name, plot_type, area, last_year_season2, need_bean, economic_data, scenario):
    candidates = economic_data[
        (economic_data['地块类型'] == plot_type) &
        (economic_data['种植季次'] == '单季')
    ].copy()

    # 检查重茬种植：不能与上一季次（去年第二季）种植相同作物
    candidates = candidates[~candidates['作物名称'].isin(last_year_season2)]
    if need_bean:
        candidates = candidates[candidates['作物名称'].isin(BEAN_CROPS)]

    if not candidates.empty:
        best_crop = candidates.nlargest(1, '每亩净收益').iloc[0]
        demand = CROP_DEMANDS.get(best_crop['作物名称'], 1000)  # 默认需求量1000
        profit = calculate_profit(best_crop, area, scenario, demand)
        return {
            'plan': [{'地块名称': plot_name, '作物名称': best_crop['作物名称'], '面积': area, '季次': '单季'}],
            'profit': profit
        }
    return {'plan': [], 'profit': 0}

def _handle_water_land(plot_name, area, last_year_season2, current_year_season1, need_bean, economic_data, scenario):
    # 选项1：单季水稻
    rice = economic_data[
        (economic_data['作物名称'] == '水稻') &
        (economic_data['种植季次'] == '单季')
    ]
    # 检查重茬种植：不能与上一季次（去年第二季）种植相同作物
    if not rice.empty and '水稻' not in last_year_season2:
        demand = CROP_DEMANDS.get('水稻', 1000)
        rice_profit = calculate_profit(rice.iloc[0], area, scenario, demand)
    else:
        rice_profit = -np.inf

    # 选项2：两季蔬菜
    # 第一季：不能与上一季次（去年第二季）种植相同作物
    veg1 = economic_data[
        (economic_data['种植季次'] == '第一季') &
        (~economic_data['作物名称'].isin(SEASON2_VEG)) &
        (~economic_data['作物名称'].isin(last_year_season2))
    ]
    if need_bean:
        veg1 = veg1[veg1['作物名称'].isin(BEAN_CROPS)]

    # 第二季：不能与上一季次（今年第一季）种植相同作物
    veg2 = economic_data[
        (economic_data['种植季次'] == '第二季') &
        (economic_data['作物名称'].isin(SEASON2_VEG)) &
        (~economic_data['作物名称'].isin(current_year_season1))
    ]

    if not veg1.empty and not veg2.empty:
        best_veg1 = veg1.nlargest(1, '每亩净收益').iloc[0]
        best_veg2 = veg2.nlargest(1, '每亩净收益').iloc[0]
        demand1 = CROP_DEMANDS.get(best_veg1['作物名称'], 1000)
        demand2 = CROP_DEMANDS.get(best_veg2['作物名称'], 1000)
        veg_profit = calculate_profit(best_veg1, area, scenario, demand1) + calculate_profit(best_veg2, area, scenario, demand2)
    else:
        veg_profit = -np.inf

    if rice_profit >= veg_profit:
        return {
            'plan': [{'地块名称': plot_name, '作物名称': '水稻', '面积': area, '季次': '单季'}],
            'profit': rice_profit
        }
    else:
        return {
            'plan': [
                {'地块名称': plot_name, '作物名称': best_veg1['作物名称'], '面积': area, '季次': '第一季'},
                {'地块名称': plot_name, '作物名称': best_veg2['作物名称'], '面积': area, '季次': '第二季'}
            ],
            'profit': veg_profit
        }

def _handle_greenhouse(plot_name, area, last_year_season2, current_year_season1, need_bean, economic_data, is_smart, scenario):
    # 第一季：不能与上一季次（去年第二季）种植相同作物
    season1_candidates = economic_data[
        (economic_data['种植季次'] == '第一季') &
        (~economic_data['作物名称'].isin(last_year_season2))
    ]
    if not is_smart:
        season1_candidates = season1_candidates[~season1_candidates['作物名称'].isin(SEASON2_VEG)]
    if need_bean:
        season1_candidates = season1_candidates[season1_candidates['作物名称'].isin(BEAN_CROPS)]

    if is_smart:
        # 第二季：不能与上一季次（今年第一季）种植相同作物
        season2_candidates = economic_data[
            (economic_data['种植季次'] == '第二季') &
            (~economic_data['作物名称'].isin(current_year_season1))
        ]
    else:
        season2_candidates = economic_data[
            (economic_data['作物名称'].isin(FUNGI_CROPS)) &
            (economic_data['种植季次'] == '第二季')
        ]
    
    # 添加调试信息
    if plot_name.startswith('E'):  # 普通大棚地块以E开头
        print(f"调试 - {plot_name}:")
        print(f"  第一季候选作物数量: {len(season1_candidates)}")
        print(f"  第二季候选作物数量: {len(season2_candidates)}")
        print(f"  去年第二季种植: {last_year_season2}")
        print(f"  今年第一季种植: {current_year_season1}")
        if not season1_candidates.empty:
            print(f"  第一季候选作物: {season1_candidates['作物名称'].tolist()}")
        if not season2_candidates.empty:
            print(f"  第二季候选作物: {season2_candidates['作物名称'].tolist()}")
        else:
            print(f"  第二季候选作物为空，检查菌类作物:")
            fungi_data = economic_data[economic_data['作物名称'].isin(FUNGI_CROPS)]
            print(f"    菌类作物数据: {fungi_data[['作物名称', '种植季次']].to_dict('records')}")
            # 检查所有第二季作物
            all_season2 = economic_data[economic_data['种植季次'] == '第二季']
            print(f"    所有第二季作物: {all_season2['作物名称'].tolist()}")

    if not season1_candidates.empty and not season2_candidates.empty:
        best_s1 = season1_candidates.nlargest(1, '每亩净收益').iloc[0]
        best_s2 = season2_candidates.nlargest(1, '每亩净收益').iloc[0]
        demand1 = CROP_DEMANDS.get(best_s1['作物名称'], 1000)
        demand2 = CROP_DEMANDS.get(best_s2['作物名称'], 1000)
        profit = calculate_profit(best_s1, area, scenario, demand1) + calculate_profit(best_s2, area, scenario, demand2)
        return {
            'plan': [
                {'地块名称': plot_name, '作物名称': best_s1['作物名称'], '面积': area, '季次': '第一季'},
                {'地块名称': plot_name, '作物名称': best_s2['作物名称'], '面积': area, '季次': '第二季'}
            ],
            'profit': profit
        }
    return {'plan': [], 'profit': 0}

# ====================4.辅助函数======================
def load_2023_data(planting_2023):
    """加载2023年的种植数据"""
    history_2023 = {}
    for _, row in planting_2023.iterrows():
        plot_name = row['种植地块']
        crop_name = row['作物名称']
        season = row['种植季次']
        
        # 将单季作物标记为第一季
        if season == '单季':
            season = '第一季'
            
        if plot_name not in history_2023:
            history_2023[plot_name] = {}
        if season not in history_2023[plot_name]:
            history_2023[plot_name][season] = []
        history_2023[plot_name][season].append(crop_name)
    return history_2023

def create_result_template(year_plans, year, scenario, land_info, crop_info):
    """创建结果模板 - 区分季次（单季算作第一季）"""
    # 获取所有地块
    all_plots = land_info['地块名称'].tolist()
    # 所有作物 + 季次
    all_crops = []
    for crop in crop_info['作物名称'].unique().tolist():
        all_crops.append(f"{crop}-第一季")
        all_crops.append(f"{crop}-第二季")
    
    # 创建结果矩阵
    result_matrix = pd.DataFrame(0.0, index=all_plots, columns=all_crops)
    result_matrix.index.name = '地块名称'
    
    # 填充数据
    for plan in year_plans:
        plot_name = plan['地块名称']
        crop_name = plan['作物名称']
        season = plan['季次']
        area = plan['面积']
        
        if season == '单季':
            season = '第一季'
        col_name = f"{crop_name}-{season}"
        
        if plot_name in all_plots and col_name in result_matrix.columns:
            result_matrix.at[plot_name, col_name] += area
    
    return result_matrix


def save_results(year_plans, year, scenario, land_info, crop_info):
    """保存结果 - 合并两个季次"""
    result_matrix = create_result_template(year_plans, year, scenario, land_info, crop_info)
    csv_filename = f'result1_{scenario}_{year}.csv'
    result_matrix.to_csv(csv_filename, encoding='utf-8-sig', float_format='%.2f')
    print(f"已保存{year}年{scenario}情况结果到 {csv_filename}")
    return result_matrix

# ====================5.主程序======================
def solve_problem_1_scenario(economic_data, planting_2023, land_info, crop_info, scenario):
    """解决问题1的特定情况"""
    print(f"\n=== 解决问题1：{scenario}情况 ===")
    
    # 初始化历史记录
    history = {2023: load_2023_data(planting_2023)}
    
    # 存储结果
    all_plans = []
    total_profit = 0
    
    for year in range(2024, 2031):
        print(f"\n开始规划{year}年种植方案...")
        year_profit = 0
        year_plans = []
        
        for plot_name in land_info['地块名称']:
            result = find_best_planting(plot_name, year, history, economic_data, land_info, scenario)
            year_plans.extend(result['plan'])
            year_profit += result['profit']
        
        # 更新历史记录
        history[year] = {}
        for p in year_plans:
            plot_name = p['地块名称']
            crop_name = p['作物名称']
            season = p['季次']
            
            # 将单季作物标记为第一季
            if season == '单季':
                season = '第一季'
                
            if plot_name not in history[year]:
                history[year][plot_name] = {}
            if season not in history[year][plot_name]:
                history[year][plot_name][season] = []
            history[year][plot_name][season].append(crop_name)
        
        all_plans.extend(year_plans)
        total_profit += year_profit
        
        print(f"{year}年预期净收益: {year_profit:,.2f} 元")
        
        # 保存结果
        save_results(year_plans, year, scenario, land_info, crop_info)
    
    # 计算并保存年均收益
    avg_profit = total_profit / 7
    print(f"\n{scenario}情况完成！2024-2030年总的年均净收益为 {avg_profit:,.2f} 元")
    
    return all_plans, avg_profit

def main():
    """主函数"""
    print("开始执行农作物种植策略优化模型...")
    
    land_info, crop_info, planting_2023, economic_data = load_data()
    print("数据加载完成")
    
    print("\n" + "="*60)
    print("情况1：超过部分滞销，造成浪费")
    print("="*60)
    waste_plans, waste_avg_profit = solve_problem_1_scenario(economic_data, planting_2023, land_info, crop_info, 'waste')
    
    print("\n" + "="*60)
    print("情况2：超过部分按2023年销售价格的50%降价出售")
    print("="*60)
    discount_plans, discount_avg_profit = solve_problem_1_scenario(economic_data, planting_2023, land_info, crop_info, 'discount')
    
    # ⚠️ 这里修改：不再保存到CSV，只输出
    print("\n" + "="*60)
    print("最终结果：")
    print("="*60)
    print(f"情况1（滞销）：年均净收益 {waste_avg_profit:,.2f} 元")
    print(f"情况2（降价）：年均净收益 {discount_avg_profit:,.2f} 元")
    print("="*60)

if __name__ == "__main__":
    main()