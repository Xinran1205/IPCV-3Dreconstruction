# 三维球体重建项目

## 项目概述
本项目使用对极几何和三角测量等技术，利用双摄像头视角的2D图像进行3D球体重建，并有效验证了重建结果。核心代码全部手写，未使用任何外部库，采用独创的优化方法提高了模型的精确度和实用性。
## 详细步骤

### 圆形检测
- 使用霍夫圆变换在两个视角（view0 和 view1）的图像中检测圆形。
- 调整如最小距离（minDist）、Canny算子阈值（param1）和累加器阈值（param2）等参数，以优化圆的检测精度。

### 对极几何应用
- 计算本质矩阵（Essential Matrix）和基础矩阵（Fundamental Matrix），后者结合了摄像机的内参，用于确定两个视图中点对应的极线位置。

### 圆心匹配
- 使用基础矩阵为view0中的每个圆心在view1中找到对应的极线，并匹配最接近的圆心对。

### 三维重建
- 利用匹配的圆心对进行三角测量，重建它们在三维空间中的坐标。

### 半径的重建
- 选择每个圆的边缘点，找到对应的匹配点，再通过三角测量确定球体的半径。

### 验证和可视化
- 在重建的球体和真实球体（ground truth）上进行点采样，并将这些点重新投影到两个摄像机的视角，对比重建结果与实际数据。

## 优化步骤

### 圆心匹配的优化（双向匹配）
- 单向匹配可能导致误匹配，因为其他圆心到极线的距离可能比真正的圆心到极线的距离更短。为解决这一问题，我采用了双向匹配法。这种方法通过从两个视图进行互相匹配和验证，显著减少了误匹配的可能性。即便如此，由于霍夫圆检测本身存在的误差，仍有可能在反向匹配中找到错误的点，但双向匹配显著降低了这种风险。

### 半径重建的策略改进
- 在重建球体半径时，我最初尝试在一个圆上随机选择点进行匹配，但是当极线与圆有两个交点时，无法确定正确匹配点。为解决这个问题，我观察并利用了两个摄像头的外部参数，改变了选取圆上点的策略：固定选择圆心右侧与x轴平行的点，并在有两个交点的情况下总选择x坐标最大的交点。这种方法有效地解决了交点匹配的问题，提高了半径计算的准确性。

### 结果案例
<img src="/pic/1.png" alt="view0">
<img src="/pic/2.png" alt="view1">