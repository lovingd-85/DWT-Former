# DWT-Former
![Fram](https://github.com/user-attachments/assets/220678f7-298f-436e-bcc9-aaf30efb83f0)
这是论文 《DWT-Former: Fusing Wavelet-Based Multi-Scale Features and Transformer-Based Temporal Representations for Photovoltaic Power Forecasting》 的官方 PyTorch 实现。

光伏（PV）功率预测面临着捕捉长程时间依赖关系以及融合气象和历史功率等多源异构信息的挑战 。精确建模光伏功率从季节性到高频变化的多尺度动态是实现准确预测的关键 。本项目提出的 DWT-Former 是一种新颖的混合深度学习框架，旨在解决这些挑战 。其独特的双分支架构结合了 Transformer 模型在处理时间依赖性方面的优势和多尺度小波分析在捕捉频率特性方面的能力 。通过在多个真实光伏数据集上的出色表现，DWT-Former 证明了其在捕捉复杂光伏发电动态方面的强大能力，为优化电网管理和促进太阳能整合提供了有价值的工具 。
DWT-Former 模型的核心是一个双分支协作预测框架，旨在解决光伏功率预测中固有的非平稳性、多尺度耦合特性和多源数据融合的挑战。

---------------------------------------
## 开始使用
1. 环境要求
  实验的软件环境基于 Python 3.11、PyTorch 2.5.1 和 CUDA 12.4 。
  * Python 3.11+
  * PyTorch 2.5.1
  * NumPy
  * Pandas

2. 安装
  克隆本仓库：
  ```bash
  # 安装依赖
  pip install -r requirements.txt  # 这行不会被高亮
  python setup.py install          # 这行也不会
  ```

  ``` base
  git clone https://github.com/your-username/DWT-Former.git
  cd DWT-Former
  ```

  (推荐) 创建一个虚拟环境：
  ``` bash
  python -m venv venv
  source venv/bin/activate  # on Windows use `venv\Scripts\activate`
  ```
 3. 安装依赖包：
  ``` bash
  pip install -r requirements.txt
  ```
## 模型架构
DWT-Former 模型的核心是一个双分支协作预测框架，旨在解决光伏功率预测中固有的非平稳性、多尺度耦合特性和多源数据融合的挑战 
模型的整体架构如上图所示 ，主要包括：
小波多尺度分支 (Wavelet Multi-scale Branch)：该分支利用离散小波变换（DWT）将原始功率数据分解为趋势（Trend）和季节性（Season）分量 。随后，一个精细的多尺度处理模块会分析这些分量，构建一个包含时间、频率和尺度的丰富三维特征空间 。
Transformer 分支 (Transformer Branch)：该分支采用一种改进的 Transformer 架构（灵感来源于 iTransformer），负责高效地从融合了气象数据和历史功率数据的输入中捕捉长期依赖关系和变量间的相互作用 。
特征融合模块 (Feature Fusion Module)：最后，一个融合模块将来自上述两个分支的特征进行整合，从而实现高精度的功率预测 。
![fram](https://github.com/user-attachments/assets/f4b29a00-c1bf-4171-8902-603e6e0f283c)


## 结果
DWT-Former 在所有四个数据集上的表现优于或显著优于当前主流的时间序列预测基准模型（如 LSTM, GRU, Informer, TimesNet 等）
