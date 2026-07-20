# LIFT2 HDF5 到 LingBot-VA / 4DWAM 的数据准备

本文档定义本地 Robbyant LingBot-VA + 4DWAM 代码库当前的 LIFT2 数据转换边界。
文档特意区分两种媒体布局，因为恢复后的上游 Step 3 脚本与标准 LeRobot
视频存储采用了不同的原始相机数据表示。

转换器路径如下：

```text
lingbot-va/preprocess/convert_hdf5_to_lingbot_lerobot.py
```

转换器只生成原始 LeRobot 阶段的数据。LingBot-VA / 4DWAM 训练还需要
latent `.pth` 文件；启用 trace alignment 作为 action representation 时，还需要
trace `.pth` 文件。

## 1. 已确定的设计与不可破坏的契约

### 1.1 保留原始 60 Hz LIFT2 时间线

原始 LIFT2 录制数据按 60 Hz 处理。转换器要求：

```text
source_fps == fps == 60
```

转换器明确拒绝执行以下操作：

- 从 60 Hz 下采样到 30 Hz；
- 使用混合 RGB 帧或线性插值的欧拉角进行上采样；
- 删除 episode 开头的静止片段；
- 修改 episode 边界。

这些操作会改变 `action_config`、latent `frame_ids`、TraceAnything 提取和
action supervision 所使用的帧索引空间。如果未来的数据清洗阶段需要删除帧，
必须同时输出从原始帧到转换后帧的显式映射，并重新生成所有下游产物。

### 1.2 保留原始 action 契约

转换器从 HDF5 顶层的 `action_eef` 写入以下 14 维绝对 action：

```text
[
  left_x, left_y, left_z,
  left_roll, left_pitch, left_yaw,
  left_gripper,
  right_x, right_y, right_z,
  right_roll, right_pitch, right_yaw,
  right_gripper,
]
```

转换器将 `observations/qpos` 原样写入 14 维 `observation.state` feature。

不要在转换器中预先计算 delta action、相对位姿、四元数或补齐后的 30 维
action。当前 `LatentLeRobotDataset` 会执行以下变换：

```text
14D 欧拉角 EEF action
  -> 16D 四元数 EEF action
  -> 在每个 action_config segment 内计算相对位姿
  -> 对齐到 30D multi-embodiment action 通道
  -> 归一化
```

LIFT2 原始夹爪指令通常位于 `[0, 5]`。转换器默认只将 action 的第 6 和第 13
通道归一化到 `[0, 1]`，不会修改 qpos state。这只是原始数据约定；完成 loader
中的全部 action 变换后，仍然必须重新计算 LIFT2 的归一化统计量。

### 1.3 相机顺序具有明确语义

三个必需的相机 feature key 由 `wan_va/configs/va_robotwin_cfg.py` 固定：

```text
observation.images.cam_high
observation.images.cam_left_wrist
observation.images.cam_right_wrist
```

loader 会先横向拼接左右手腕视角，再把 high 视角拼接到下方。除非同时修改
loader 和 `obs_cam_keys`，否则不要重命名、重新排序、删除或替换这些 key。

### 1.4 `action_config` 元数据与 fallback 行为

每个转换后的 episode 都包含一个覆盖完整 episode 的显式 segment：

```text
{
  "episode_index": 0,
  "tasks": ["Open the toolbox and find the wrench."],
  "length": T,
  "action_config": [
    {
      "start_frame": 0,
      "end_frame": T,
      "action_text": "Open the toolbox and find the wrench.",
      "skill": ""
    }
  ]
}
```

`start_frame` 和 `end_frame` 使用未修改的 60 Hz 转换后 episode 帧索引空间。
它们不是原始 HDF5 文件名、墙上时钟时间戳或下采样后的索引。

转换器会显式写入该元数据。对于原本没有 `action_config` 的数据集，Step 3
提取器也允许按默认行为生成相同的整段 episode fallback。该 fallback 是预期
设计：没有更细粒度 skill 标注的数据集仍可继续执行 latent 提取，同时保留完整
episode 边界。如果 episode 中存在必须分别保留的子任务边界，则应提供显式的
多个 segment。

## 2. HDF5 输入要求

每个源目录包含以下形式的文件：

```text
episode_0.hdf5
episode_1.hdf5
...
```

其中 `T` 表示当前 episode 的总时间步数，`N` 表示单帧图片编码缓冲区的固定
容量。

### 2.1 根属性和完整层级

LIFT2 HDF5 文件的根属性为：

```text
compress = true
sim = false
```

- `sim=false` 表示 episode 来自真实机器人数据，而不是仿真数据；
- `compress=true` 表示图片以编码后的字节流形式保存，不表示 HDF5 dataset
  本身启用了 gzip、lzf 等压缩 filter；
- 各 dataset 不依赖额外属性表达字段语义。

完整 HDF5 层级如下：

```text
/
├── action                                  float32 [T, 14]
├── action_base                             float32 [T, 6]
├── action_eef                              float32 [T, 14]
└── observations/
    ├── eef                                 float32 [T, 14]
    ├── effort                              float32 [T, 14]
    ├── qpos                                float32 [T, 14]
    ├── qvel                                float32 [T, 14]
    ├── robot_base                          float32 [T, 6]
    └── images/
        ├── head                            uint8 [T, N]
        ├── left_wrist                      uint8 [T, N]
        └── right_wrist                     uint8 [T, N]
```

数值 dataset 采用连续存储，即 `chunks=None`。三路图片 dataset 按单帧进行
chunk，形式为 `chunks=(1, N)`。

### 2.2 转换器必需字段和输出映射

每个 episode 至少必须提供以下五个同步 dataset：

```text
action_eef                               [T, 14] float32
observations/qpos                        [T, 14] float32
observations/images/head                 [T, N]  编码后的 JPEG 或 PNG 字节
observations/images/left_wrist           [T, N]  编码后的 JPEG 或 PNG 字节
observations/images/right_wrist          [T, N]  编码后的 JPEG 或 PNG 字节
```

它们与 LeRobot 输出的映射关系如下：

| HDF5 字段 | Shape | LeRobot 输出 | 处理方式 |
| --- | --- | --- | --- |
| `action_eef` | `[T, 14]` | `action` | 转为 `float32`；默认只归一化第 6、13 通道的夹爪值 |
| `observations/qpos` | `[T, 14]` | `observation.state` | 转为 `float32` 并保持原值 |
| `observations/images/head` | `[T, N]` | `observation.images.cam_high` | 解码 JPEG/PNG，再写为 RGB 图片或视频帧 |
| `observations/images/left_wrist` | `[T, N]` | `observation.images.cam_left_wrist` | 解码 JPEG/PNG，再写为 RGB 图片或视频帧 |
| `observations/images/right_wrist` | `[T, N]` | `observation.images.cam_right_wrist` | 解码 JPEG/PNG，再写为 RGB 图片或视频帧 |

以下字段允许存在，但当前转换器不会读取或写入：

```text
action
action_base
observations/eef
observations/effort
observations/qvel
observations/robot_base
```

尤其需要注意：LeRobot 输出中的 `action` 来源是 HDF5 的 `action_eef`，不是
HDF5 顶层同名的 `action`。两个 dataset 的数值语义和范围不同，不应混用。

### 2.3 14 维字段约定

`action_eef` 的 14 个通道按以下顺序解释：

```text
0  left_x             7  right_x
1  left_y             8  right_y
2  left_z             9  right_z
3  left_roll         10  right_roll
4  left_pitch        11  right_pitch
5  left_yaw          12  right_yaw
6  left_gripper      13  right_gripper
```

`observations/qpos` 同样是 14 维双臂状态，但转换器只把它作为 opaque 的
`observation.state` 保存，不在转换阶段重新解释、重排或归一化各通道。

LIFT2 原始夹爪值应位于 `[0, 5]`。使用默认 `gripper_mode=normalize` 时，
转换器会将 `action_eef` 的第 6、13 通道截断到 `[0, 5]`，再除以 5 映射到
`[0, 1]`；使用 `gripper_mode=raw` 时则保持原值。

### 2.4 图片编码格式

三路相机的每个时间步不是未压缩的 `[H, W, C]` 像素数组，而是一个长度固定
为 `N` 的 `uint8` 行。每行开头保存实际 JPEG/PNG 字节，末尾可以使用零填充
到固定长度：

```text
[ JPEG/PNG 有效字节 | 0x00 padding ... ]
```

不同帧的有效编码长度可以不同，因此不能把 `N` 当作真实编码长度，也不能把
`[T, N]` 当作图片分辨率。转换器将整行交给 OpenCV 解码；decoder 会在编码流
结束标记处停止，末尾的零 padding 不影响解码。随后，转换器将 OpenCV 的 BGR
输出转换为 RGB。

三路相机必须能解码为 `[H, W, 3]`。同一相机在整个 episode 内不得改变
分辨率；不同相机可以具有各自固定的分辨率。

### 2.5 同步、长度和有效性校验

转换器以 `action_eef.shape[0]` 作为 episode 长度 `T`，并要求：

```text
observations/qpos.shape[0] == T
observations/images/head.shape[0] == T
observations/images/left_wrist.shape[0] == T
observations/images/right_wrist.shape[0] == T
```

转换器还会检查：

- `action_eef` 和 `observations/qpos` 的 shape 均为 `[T, 14]`；
- `T > 0`；
- 所有 action 和 qpos 数值均为有限值，不包含 NaN 或 Inf；
- 每个相机帧都能成功解码；
- 同一相机的所有帧分辨率保持一致。

转换过程逐帧按相同索引读取 action、qpos 和三路图片，不执行时间戳对齐、插值、
跳帧或重采样。因此，同一个 `frame_index` 必须在源 HDF5 中已经表示同一采集
时刻的数据。遇到缺失字段、长度不一致、维度错误、非有限数值、图片损坏或
分辨率变化时，转换会直接失败，不会静默丢弃数据。

## 3. 两种有意保留的媒体布局

两种布局都会提供可供 `preprocess/extract_latents_from_pixels_adaptv1.py`
读取的 MP4 文件。应根据是否还需要在 parquet 中嵌入图片字节来选择布局。

### 3.1 `step3-compatible` 布局（默认）

当下游除了 Step 3 所用的 MP4 文件外，还需要 parquet 内嵌图片字节时，使用
此布局。

```text
dataset_root/
├── meta/
│   ├── info.json                    # 相机 feature 的 dtype=image
│   ├── episodes.jsonl               # 包含 action_config
│   └── lingbot_conversion.json
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet   # action、state 和图片字节
├── images/
│   └── observation.images.cam_high/
│       └── episode_000000/
│           └── frame_*.png
└── videos/
    └── chunk-000/
        └── observation.images.cam_high/
            └── episode_000000.mp4
```

这是一个显式的兼容层：

- Step 3 从 `videos/` 读取延后生成的 MP4 sidecar；
- Step 4 从 `videos/` 读取同一组 MP4 sidecar；
- `meta/info.json` 如实保持 `dtype=image`，不会修改 `total_videos` 来伪装
  相机 feature 是原生 video。

所有以图片为后端的 episode 保存完成后，转换器才会编码 MP4 sidecar，从而
避免在写入过程中违反 LeRobot writer 的断言。

当下游仍然需要 parquet 中的图片字节时使用此模式。它不是长期推荐的
LeRobot 存储形式。

### 3.2 `standard-video` 布局

使用此布局生成规范的 LeRobot v2.1 数据集：

```text
dataset_root/
├── meta/
│   ├── info.json                    # 相机 feature 的 dtype=video
│   ├── episodes.jsonl
│   └── lingbot_conversion.json
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet   # action/state/索引信息
└── videos/
    └── chunk-000/
        └── observation.images.cam_high/
            └── episode_000000.mp4
```

转换器使用 LeRobot 原生的 `dtype=video` feature writer。metadata 采用 CHW
视频 shape 语义（`[3, H, W]`），视频 metadata、MP4 创建和临时帧清理由
LeRobot 负责。

可直接在该布局上运行 `preprocess/extract_latents_from_pixels_adaptv1.py`。
脚本会读取 `videos/` 下的 MP4 文件并执行确定性的按步长采样。

## 4. 提取 Step 3 latent

使用 `preprocess/extract_latents_from_pixels_adaptv1.py`。提取器会：

- 从 `videos/` 读取相机 MP4 文件；
- 在 `latents/` 下按相机、按 segment 写入 `.pth` 文件；
- 根据 `ori_fps` 和目标 `fps` 采样帧；
- 将 latent 和文本 embedding tensor 保存到 CPU，避免 `.pth` 文件保留 CUDA
  设备信息。

转换器和提取器目前默认每个 chunk 包含 500 个 episode。对于该转换器生成的
数据集，使用默认值即可，也可以显式传入：

```bash
--episodes-per-chunk 500
```

### Step 3 默认时序契约

本地 RoboTwin / 4DWAM 配置为：

```text
action_per_frame = 16
Wan VAE temporal compression = 4
```

loader 按以下方式计算每个 VAE latent frame 对应的 action 数量：

```text
每个 VAE latent frame 的 action 数量 = source_frame_stride * 4
```

提取器通过整数除法计算源帧步长：

```text
source_frame_stride = ori_fps // target_fps
默认 source_frame_stride = 50 // 12 = 4
4 * 4 = 每个 VAE latent frame 对应 16 个 action
```

在当前默认配置（`ori_fps=50`、`target_fps=12`）下，计算出的 stride 始终为
4，与本地 4DWAM 配置一致。提取器会写入如下确定性 frame ID：

```text
0, 4, 8, 12, ...
```

同时，提取器会选择符合 Wan 时序约定的输入帧数，通常为 `4k + 1`。当前
Step 3 流程依赖以下行为：

1. 从 `meta/info.json` 读取数据集 fps 和 chunk size；
2. 使用已有的 `action_config` segment；缺失时生成预期的整段 episode fallback；
3. 从 MP4 或显式兼容输入中读取帧；
4. 通过 `ori_fps // target_fps` 计算帧步长，默认 `50 // 12` 时结果为 4；
5. 在默认配置下保持 `frame_stride * 4 == action_per_frame`；
6. 写入准确的 `frame_ids`、源 fps、采样 fps 和 latent 维度。

如果覆盖了 `ori_fps` 或 `target_fps`，应在开始训练前确认整数除法得到的 stride
仍与 `action_per_frame` 兼容。

## 5. 转换器命令行

在 LingBot-VA 仓库中运行：

```bash
cd /soft/wangxi/4DWAM/lingbot-va
uv run python preprocess/convert_hdf5_to_lingbot_lerobot.py --help
```

### 直接运行：兼容恢复后的 Step 3

```bash
uv run python preprocess/convert_hdf5_to_lingbot_lerobot.py \
  --data-dir /soft/wangxi/4DWAM/datasets/dataset_0402_wrench \
  --repo-id lift2_wrench_step3_compatible_60hz \
  --output-root /soft/wangxi/4DWAM/datasets_converted \
  --task-description "Open the toolbox, inspect its contents, and find the wrench." \
  --source-fps 60 \
  --fps 60 \
  --media-layout step3-compatible \
  --gripper-mode normalize \
  --overwrite
```

### 直接运行：原生 LeRobot 视频存储

```bash
uv run python preprocess/convert_hdf5_to_lingbot_lerobot.py \
  --data-dir /soft/wangxi/4DWAM/datasets/dataset_0402_wrench \
  --repo-id lift2_wrench_standard_video_60hz \
  --output-root /soft/wangxi/4DWAM/datasets_converted \
  --task-description "Open the toolbox, inspect its contents, and find the wrench." \
  --source-fps 60 \
  --fps 60 \
  --media-layout standard-video \
  --gripper-mode normalize \
  --overwrite
```

删除已有输出目录前必须显式提供 `--overwrite`。第一次转换时建议使用
`--max-episodes 2` 进行 smoke test。

## 6. YAML 配置

转换器支持仓库中已有的 `preprocess/*conversion*.yaml` 文件。安全的 LIFT2
配置示例如下：

```yaml
defaults:
  output_root: /soft/wangxi/4DWAM/datasets_converted
  source_fps: 60
  fps: 60
  media_layout: step3-compatible
  gripper_mode: normalize
  overwrite: false
  max_episodes: null

repo_id: lift2_wrench_step3_compatible_60hz

datasets:
  - data_dir: /soft/wangxi/4DWAM/datasets/dataset_0402_wrench
    task_description: Open the toolbox, inspect its contents, and find the wrench.
```

启用以下旧配置 key 时，转换器会主动拒绝执行：

```yaml
skip_static_start: true
push_to_hub: true
```

请删除这些 key，或将其设置为 `false`。转换器不会向 Hub 发布数据；只有完成
结构、时序和训练验证后，才能执行发布操作。

使用 YAML 配置运行转换：

```bash
uv run python preprocess/convert_hdf5_to_lingbot_lerobot.py \
  --config-path preprocess/my_lift2_conversion_single.yaml
```

## 7. Step 4 trace 要求

当前 `extract_trace_from_ta.py` 按照 latent 目录树生成 trace：

```text
latents/chunk-000/<camera>/episode_000000_0_<T>.pth
trace/chunk-000/<camera>/episode_000000_0_<T>.pth
```

脚本从每个 latent `.pth` 中读取 `frame_ids`，再从对应 MP4 中读取相同索引的
图片。因此，只有 Step 3 的 frame ID 可信时，trace 才可信。

接受 trace 输出前，应检查以下条件：

```text
trace 文件名 == latent 文件名
trace 使用与 latent 完全相同的 frame_ids
trace 时间维度 == latent_num_frames
trace token 数量 == latent_height * latent_width
```

当前 trace 脚本会将其中一种时间维度不匹配报告为 warning。这是当前预期行为；
调用方可以在验证阶段检查该 warning，但预处理脚本不会因此直接失败。

## 8. 验证门槛

### 8.1 原始转换验证

对于每个转换后的 episode，检查：

```text
action shape 为 [T, 14]
observation.state shape 为 [T, 14]
三个相机都存在且各有 T 帧
所有 action_config segment 满足 0 <= start < end <= T
meta/info.json 中的 fps 为 60
```

对于 `step3-compatible` 模式，还应检查三个 sidecar MP4 文件是否存在。
对于 `standard-video` 模式，应检查三个相机 metadata feature 的 dtype 是否均为
`video`。

### 8.2 Latent 验证

对于每个 `action_config` segment 和相机，都必须存在一个包含以下字段的
`.pth` 文件：

```text
episode_index
start_frame
end_frame
frame_ids
latent
latent_num_frames
latent_height
latent_width
latent_channels
text_emb
text_emb_n
text_emb_d
```

还必须满足：

```text
latent.shape[0] == latent_num_frames * latent_height * latent_width
相邻 frame_ids 使用预期的计算步长（默认 ori_fps=50、target_fps=12 时为 4）
len(frame_ids) 符合所选 Wan 时序约定
```

### 8.3 Loader 和训练验证

在执行完整预处理或 FSDP 训练前，先加载两个 episode 并调用：

```python
sample = dataset[0]
```

验证：

```text
能够发现三个相机对应的 latent 路径
能够构造 T 形 latent mosaic
action 包含 30 个通道，每个 latent frame 对应 16 个 action slot
enable_trace=True 时能够重排并拼接 trace
不会出现归一化除零或 shape assertion 错误
```

随后运行一次单 GPU、十个 step 的训练 smoke test。只有该测试成功后，才应
转换完整 LIFT2 数据集、执行 latent 和 trace 预处理，并启动 FSDP 训练。

## 9. LIFT2 归一化

不要直接复用 RobotWin 的 `q01` 和 `q99`。必须在执行本地 loader 的完整变换后
计算 LIFT2 统计量：

```text
绝对 14D LIFT2 EEF action
  -> 四元数 16D action
  -> 按 action_config 计算相对位姿
  -> 映射到 30D action 通道
  -> 计算 q01 / q99
```

直接使用 HDF5 `action_eef` 计算 quantile，得到的是另一种 action 表示的统计量，
不适用于当前 4DWAM 训练流程。

## 10. 推荐的后续实施顺序

1. 使用 `step3-compatible` 模式转换两个 episode，并验证 Step 3 和 Step 4 的
   文件读写。
2. 验证默认 Step 3 配置确实计算出 `50 // 12 = 4`，写入确定性的
   `frame_ids`，并保持每个 latent frame 对应 16 个 action 的关系。
3. 有显式 `action_config` 时验证其 segment；缺失时验证预期的整段 episode
   fallback。
4. 在验证阶段检查非致命的 trace 时间维度不匹配 warning。
5. 修正 `LatentLeRobotDataset` 的路径，使其使用 `self.root / "latents"` 和
   `self.root / "trace"`，并移除对 `chunk-000` 的固定假设。
6. 计算 LIFT2 完成 action 变换后的 quantile。
7. 执行双 episode loader 验证和十 step 训练验证。

只有第 7 步通过，才能证明完整的 LIFT2 -> 4DWAM 链路在语义上有效。仅完成
HDF5 转换并不足以证明整个训练数据链路正确。
