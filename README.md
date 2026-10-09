# Freqtune_Pipeline

基于准静态 S 参数目标，对粗粒度 PEEC 电路的电感矩阵 `L` 和电势系数矩阵 `P` 做频域校准。项目提供 Windows x64 可执行程序、一个双端口样例，以及 `Coarse_Freqtune.exe` 的 C++ 源码。

**目标曲线由外部独立生成。** 频域调参程序只读取 `Coarse/Target_map.txt`，不会启动准静态模型提取器或求解器。准静态程序保留在项目中，方便单独生成目标和核对结果。

## 目录与程序

| 路径 | 用途 |
| --- | --- |
| `Data/model.msh`、`Data/set.txt` | 样例网格和准静态求解配置 |
| `Coarse/Topology/` | 粗拓扑的节点、支路、端口及 PEEC 参数配置 |
| `Coarse/Model/` | 样例粗模型矩阵和关联文件 |
| `Coarse/Target_map.txt` | 样例准静态 S 参数目标 |
| `Coarse_Topology_Extract.exe` | 网格到可编辑粗拓扑 |
| `Coarse_Model_Extract.exe` | 粗拓扑到 `L/P/A_E/A_ET/B2N/Port`，并生成模型报告 |
| `Quasi_Static_Model_Extract.exe`、`Quasi_Static_Solver.exe` | 独立生成准静态响应 |
| `Coarse_Freqtune.exe` | 用目标 S 参数训练粗模型 `L/P` |
| `S_Parameter_Plotter.exe` | 绘制准静态求解器输出的 S 参数 |
| `src/Coarse_Freqtune/` | 调参程序源码、CMake 文件和 Eigen 3.4.0 |

仓库不包含本地备份、训练检查点、诊断实验、模型提取运行报告、求解器运行输出、编译 manifest 或约 96 MB 的可选 GNUplot 副本。这些文件由 `.gitignore` 排除；运行时会在本机重新生成。

## 用现有样例直接训练

在项目根目录打开 PowerShell：

```powershell
.\Coarse_Freqtune.exe --stop-mae-db 0.5
```

默认读取：

- 模型：`Coarse/Model/`
- 目标：`Coarse/Target_map.txt`
- 输出：`Coarse/Freqtune_output/`

`--stop-mae-db` 是必填项，单位为 dB。程序使用目标文件中的**全部频点**，初始模型和每轮更新后都检查阈值；达标即停止。当前版本只支持双端口。端口连接、顺序和参考阻抗均取自 `Coarse/Model/Port.txt`，没有阻抗覆盖接口。

训练结果见 `Coarse/Freqtune_output/summary.txt`；最佳模型在 `Coarse/Freqtune_output/best/`，其中包含 `L.txt`、`P.txt`、`C.txt`、原样复制的端口和拓扑文件、`predicted_map.txt` 及 `metrics.txt`。逐轮记录写入 `history.csv`，`state.bin` 用于续训。**原始 `Coarse/Model/` 不会被覆盖。**

训练前请确认目标文件与粗模型描述的是相同物理端口，端口顺序、极性和参考阻抗一致。`map.txt` 本身不保存参考阻抗，程序无法仅凭曲线判断是否匹配。当前样例的两个粗模型端口均为 50 Ω。

## 换用自己的模型和目标

1. **准备网格。** 将 Gmsh 2.2/4.1 ASCII 或 binary 网格放在 `Data/model.msh`，按需要修改 `Data/set.txt` 的 `FS`、`FE`、`N_FP`、`DIM` 等参数。拓扑提取器可读取网格 Physical Group，并按指定二维导体组提取；其他求解模块的网格格式要求仍以各自说明为准。
2. **准备粗拓扑。** 运行 `.\Coarse_Topology_Extract.exe`，在图形窗口中检查节点、支路和端口，然后保存到 `Coarse/Topology/`。可点击“导入结果”读取外部 Node/Branch 继续编辑，当前 `Data/model.msh` 不变；“查看网格组”只读显示 Physical Group，默认使用全部表面与线元，无需输入组 ID。绿色方框的 `Port0`/`Port1` 与 A→B 箭头只标示 msh 原始端口线元顺序，实际正负节点在保存弹窗中由用户填写。`Node.txt` 每行为 `x y z width`；`Branch.txt` 和拓扑 `Port.txt` 使用从 **1** 开始的节点编号。端口节点 `0` 目前不能交给下一步的模型提取器。
3. **提取粗模型。** 检查 `Coarse/Topology/PEEC_Config.txt` 中的 `E0`、`U0`、`UNIT_SCALE`，然后运行 `.\Coarse_Model_Extract.exe`。它在 `Coarse/Model/` 写出 `L.txt`、`P.txt`、`A_E.txt`、`A_ET.txt`、`B2N.txt`、`Port.txt` 和 `Model_Report.txt`。模型文件中的节点编号从 **0** 开始。双击 exe 运行结束后会显示结果窗口；命令行调用会自动退出，可用 `--show-summary` 或 `--no-summary` 控制。最近一次成功或失败的报告保存在 `Coarse/Model_Extract_Last_Run.txt`。新模型先写入临时目录并校验，失败时保留已有的 `Coarse/Model/`。
4. **独立生成目标。** 用适合自己模型的外部流程生成双端口 S 参数，保存为 `Coarse/Target_map.txt`。本项目的准静态程序可单独执行：

   ```powershell
   .\Quasi_Static_Model_Extract.exe
   .\Quasi_Static_Solver.exe
   Copy-Item .\Data\COMMON_DATA\map.txt .\Coarse\Target_map.txt
   ```

   复制前应核对 `Data/COMMON_DATA/PORT.txt` 与 `Coarse/Model/Port.txt` 的物理端口对应、顺序和参考阻抗。两套模型的节点编号不同，不能通过数字相等判断是否为同一端口。
5. **运行调参。** 指定所需的联合误差阈值，例如 `.\Coarse_Freqtune.exe --stop-mae-db 0.5`。

目标文件必须是无表头、逗号分隔的九列数字，频率单位 Hz，按频率严格递增：

```text
frequency_hz,S11_dB,S11_deg,S12_dB,S12_deg,S21_dB,S21_deg,S22_dB,S22_deg
```

`P.txt`、`L.txt`、`B2N.txt` 首行是矩阵行数和列数；`A_E.txt` 与 `A_ET.txt` 无尺寸表头。训练程序会检查矩阵尺寸、`A_ET=-A_E^T`、支路连接和端口连接；缺少 `Target_map.txt` 会立即报错。

## 损失、参数与停止条件

程序参照 `c29_structured_frequency_train.py` 的频域方程：先由 `P` 得到 `C=P^{-1}`，训练时保持 `C` 对称、非对角元非正且行和为正，并通过 Cholesky 因子使 `L` 保持正定。导出时再由训练后的 `C` 计算 `P=C^{-1}`。

默认优化目标是 S11 与 S12 的等权 dB Huber 损失，`beta=0.02`；优化器为 Adam。**停止和选择最佳模型**使用联合平均绝对 dB 误差：

```text
combined_mae_db =
  sum_over_target_frequencies(|S11_pred_dB - S11_target_dB|
                            + |S12_pred_dB - S12_target_dB|) / (2 * frequency_count)
```

当前样例测试后选定的默认值是 `C` 学习率 `1e-2`、`L` 学习率 `1e-3`、最大 120 轮、梯度范数裁剪 1。可用 `--c-lr`、`--l-lr`、`--max-epochs`、`--loss-kind` 等参数调整；完整选项运行 `.\Coarse_Freqtune.exe --help`。

若要在输入文件不变的前提下继续训练，并设置更严格的阈值：

```powershell
.\Coarse_Freqtune.exe --stop-mae-db 0.2 --max-epochs 300 --resume
```

新实验可使用 `--output-dir` 指向另一个目录。已有 `state.bin` 时，不加 `--resume` 不会覆盖之前的运行。返回代码 `0` 表示达到阈值，`2` 表示轮数用尽但已保存最佳结果，`1` 表示输入或计算出错。

## 可选绘图与源码构建

`S_Parameter_Plotter.exe` 读取 `Data/COMMON_DATA/S_parameter.csv`，在 `Data/Port_Responce/` 生成绘图数据和脚本。要生成 PNG，可安装 GNUplot 并加入系统 `PATH`，或运行 `.\S_Parameter_Plotter.exe --gnuplot <gnuplot.exe路径>`。没有 GNUplot 时仍可生成数据和脚本。

若需重编译频域调参程序，在 Windows 上安装 Visual Studio 2022 的 C++ 工具及 CMake，然后从项目根目录执行：

```powershell
cmake -S .\src\Coarse_Freqtune -B .\build\freqtune -G "Visual Studio 17 2022" -A x64
cmake --build .\build\freqtune --config Release
```

Release 版本会输出到项目根目录。Eigen 3.4.0 已随源码放在 `src/Coarse_Freqtune/third_party/`，其许可证见 `COPYING.MPL2`。仓库中的其余 exe 是预编译程序；其源码不在本仓库中。
