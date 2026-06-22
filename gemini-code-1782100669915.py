import numpy as np
from scipy.fft import fft
from scipy.stats import chisquare

def happy8_7D_CRA_algorithm(historical_data):
    # ================= 阶段一：信号提取 =================
    # 1. 离散傅里叶变换 (DFT)
    # 将历史数据按列(1-80)展开，计算频域信号，获取每个号码的振幅/周期权重
    freq_weights = np.abs(fft(historical_data, axis=0))[-1] 
    
    # 2. 卡方检验 (Chi-Square Test)
    # 检验各号码出现的实际频次与期望频次(25%)的差异，生成偏态系数
    expected_freq = len(historical_data) * 0.25
    observed_freq = np.sum(historical_data, axis=0) # 简化的频次统计
    chi_stat, p_values = chisquare(observed_freq, f_exp=expected_freq)
    bias_coefficients = 1 - p_values # p值越小，偏态牵引力越强
    
    # ================= 阶段二：混沌基底 =================
    # 3. 曼德尔公式 (Mandelbrot) & 4. 本福德定律 (Benford's Law)
    def generate_natural_chaos_seed(c_param):
        z = 0
        for _ in range(500): # 迭代
            z = z**2 + c_param
            # 简化版：提取数值并校验首位数字分布是否贴近本福德定律 (log10(1+1/d))
            # (此处省略本福德分布的严格拟合代码，假设返回一个符合法则的种子)
        return abs(z.real) * 1000000 
    
    # 将DFT最高频成分作为复数c输入
    c_complex = complex(np.max(freq_weights), np.mean(bias_coefficients))
    chaos_seed = generate_natural_chaos_seed(c_complex)
    
    # ================= 阶段三：双核伪随机 =================
    # 5. 平方取中法 (Middle-Square) -> 获取初始X0
    squared_str = str(int(chaos_seed**2)).zfill(12)
    x0 = int(squared_str[3:9]) # 取中间6位
    
    # 6. 线性同余法 (LCG)
    # 使用 DFT 权重动态生成参数 a 和 c
    a = int(np.mean(freq_weights) * 100) | 1 # 保证奇数
    c = int(np.max(bias_coefficients) * 100) | 1
    m = 80 # 对应 1-80 号码
    
    simulated_balls = []
    current_x = x0
    for _ in range(100000): # 扔下十万个模拟球
        current_x = (a * current_x + c) % m
        simulated_balls.append(current_x + 1) # +1 变 1-80
        
    # ================= 阶段四：物理坍缩 =================
    # 7. 高尔顿板 (Galton Board 带有动态权重偏移)
    final_slots = {i: 0 for i in range(1, 81)}
    
    for ball in simulated_balls:
        position = ball
        # 模拟高尔顿板下落的微扰，受卡方检验的 bias_coefficients 影响
        # 如果某号码近期偏冷/偏热达到临界点，小球有概率向该方向“滑落”
        perturbation = np.random.choice(
            [-1, 0, 1], 
            p=[0.1, 0.8, 0.1] # 初始概率
        )
        # 加入卡方偏态偏移修正
        if bias_coefficients[position-1] > 0.95: 
            position = position # 强偏态，如同磁铁吸附，不偏转
        else:
            position = (position + perturbation) % 80
            if position == 0: position = 80
            
        final_slots[position] += 1
        
    # 排序并选出落入小球最多的前20个槽位（号码）
    top_20_numbers = sorted(final_slots, key=final_slots.get, reverse=True)[:20]
    
    return sorted(top_20_numbers)

# 最终输出的选号结果即为这 20 个号码