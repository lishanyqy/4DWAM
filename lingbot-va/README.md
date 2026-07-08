<h1 align="center">4D-WAM: Infusing Spatial Awareness into World Action Model through 4D Trajectory Fields</h1>


# 📦 Model Download
- **Pretrained Checkpoints for Post-Training**

请下载以下模型以用于后训练。

| Model Name | Huggingface Repository | ModelScope Repository  | Description |
| :--- | :--- | :--- | :--- |
| lingbot-va-base &nbsp; | [🤗 robbyant/lingbot-va-base &nbsp;](https://huggingface.co/robbyant/lingbot-va-base) | [🤖 Robbyant/lingbot-va-base &nbsp;](https://modelscope.cn/models/Robbyant/lingbot-va-base)  | LingBot-VA w/ shared backbone|
| TraceAnything &nbsp; | [🤗 depth-anything/trace-anything &nbsp;](https://huggingface.co/depth-anything/trace-anything) | [🤖 depth-anything/trace-anything &nbsp;](https://modelscope.cn/models/depth-anything/trace-anything)  | Pretrained TraceAnything Model |

- **Post-Training Dataset in Simulation**

We use Lingbot-VA official dataset [🤗 robbyant/robotwin-clean-and-aug-lerobot](https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot).


# 🛠️ Quick Start

## Installation
**Requirements**
 • Python == 3.10.16
 • Pytorch == 2.8.0
 • CUDA 12.6

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install websockets einops diffusers==0.36.0 transformers==4.55.2 accelerate msgpack opencv-python matplotlib ftfy easydict
pip install flash-attn --no-build-isolation
```

For **Traceanything** Model, we require several libs, you can have a check with:
```bash
pip install -r requirements.txt
# pip install omegaconf viser
```

---

## Deploying 4D-WAM based on Lingbot-VA for Inference
Our step in inference mode is the same as original Lingbot-VA repository. Please refer to official document in [Lingbot-VA](https://github.com/robbyant/lingbot-va).

<!-- ### Standalone  Inference
```python
python inference.py
```
This processes the example data from `examples/0/` and saves visualizations to `result/`. -->



## Post-Training 4D-WAM based on Lingbot-VA

We support post-training (fine-tuning) LingBot-VA on custom robotic manipulation datasets. The training pipeline uses FSDP for distributed training and integrates with [LeRobot](https://github.com/huggingface/lerobot) dataset format.

### Additional Dependencies

On top of the base installation, post-training requires:

```bash
pip install lerobot==0.3.3 scipy wandb --no-deps
```

### Data Preparation

Download the post-training dataset from HuggingFace:

```bash
huggingface-cli download --repo-type dataset robbyant/robotwin-clean-and-aug-lerobot --local-dir /path/to/your/dataset
```


### Custom Dataset Preparation

step1到step3是lingbot-va的数据集处理的常规步骤，如果lingbot-va格式的预处理已经完成了，请直接查看step4

We provide a script for lerobot dataset generate latent style lerobot dataset, which is fulfill requirements of Lingbot-VA. Download WAN2.2 components (VAE2.2, T5Encoder) and move them in one directory before running. This step covers step1~step3.

```bash
python preprocess/extract_latents_from_lerobot.py --dataset-root [DATASET_PATH] --models-root [MODELS_PATH]
```

**Step 1: Convert your data to LeRobot format**

Follow the official [LeRobot dataset documentation](https://github.com/huggingface/lerobot/tree/v0.3.3) to convert your raw data (e.g., HDF5, video files, etc.) into the standard LeRobot dataset format. Ensure that each episode contains the required observation videos, actions, and metadata.

**Step 2: Add `action_config` field to `episodes.jsonl`**

After converting to LeRobot format, you need to modify the `meta/episodes.jsonl` file to add an `action_config` field to each line. This field describes the temporal segmentation and natural language description of the robot's actions within each episode.

Each line in `episodes.jsonl` should follow this format:

```json
{
  "episode_index": 0,
  "tasks": ["task description"],
  "length": 450,
  "action_config": [
    {
      "start_frame": 0,
      "end_frame": 450,
      "action_text": "Natural language description of the robot action in this segment.",
    }
  ]
}
```

- `start_frame` / `end_frame`: The frame range (0-indexed) of the action segment within the episode.
- `action_text`: A natural language description of what the robot does in this segment.

For episodes with a single continuous action, `start_frame` should be `0` and `end_frame` should equal the episode `length`. You can also define multiple segments per episode if your data contains sequential sub-tasks.

**Step 3: Extract video latents with Wan2.2 VAE**

LingBot-VA operates on video latent representations rather than raw pixels. You need to extract the latent features using the Wan2.2 VAE encoder and place them under the converted LeRobot dataset directory. Please refer to the [Wan-Video documentation](https://github.com/Wan-Video) for instructions on how to run the VAE encoder.

The extracted latent files should be placed under `latents/` in your dataset directory, mirroring the structure of `videos/`:

```
your_dataset/
├── videos/
│   └── chunk-000/
│       └── observation.images.cam_high/
│           ├── episode_000000.mp4
│           └── ...
├── latents/
│   └── chunk-000/
│       └── observation.images.cam_high/
│           ├── episode_000000_0_450.pth    # named as episode_{index}_{start_frame}_{end_frame}.pth
│           └── ...
└── meta/
    └── episodes.jsonl
```

Each `.pth` file is a dictionary containing the following fields:

| Key | Type | Description |
| :--- | :--- | :--- |
| `latent` | `Tensor [N, C]` (bfloat16) | Flattened VAE latent features (e.g., shape `[latent_num_frames * latent_height * latent_width, C]`) |
| `latent_num_frames` | `int` | Number of temporal frames in the latent space |
| `latent_height` | `int` | Spatial height in the latent space |
| `latent_width` | `int` | Spatial width in the latent space |
| `video_num_frames` | `int` | Number of frames in the (sampled) source video |
| `video_height` | `int` | Original video height in pixels |
| `video_width` | `int` | Original video width in pixels |
| `text_emb` | `Tensor [L, D]` (bfloat16) | Text embedding of the action description (encoded by Wan2.2 text encoder) |
| `text` | `str` | The raw action description text |
| `frame_ids` | `list[int]` | Sampled frame indices from the original episode (at target fps) |
| `start_frame` | `int` | Start frame index matching `action_config` in `episodes.jsonl` |
| `end_frame` | `int` | End frame index matching `action_config` in `episodes.jsonl` |
| `fps` | `int` | Target sampling fps used for latent extraction |
| `ori_fps` | `int` | Original fps of the episode data |

The latent file naming convention `episode_{index}_{start_frame}_{end_frame}.pth` corresponds to the `action_config` segments defined in `episodes.jsonl`. For example, an episode with `"start_frame": 0, "end_frame": 450` produces a latent file named `episode_000000_0_450.pth`.

**Step 4: Extract Traceanything latents from pretrained TraceAnything Model**

(1) Download Pretrained TraceAnything Model from Huggingface [Traceanything](https://huggingface.co/depth-anything/trace-anything)

(2) 读取视频帧转成Traceanything features，确保server安装了ffmpeg.
```bash
python preprocess/extract_trace_from_ta.py --dataset-root [DATASET_PATH] --ta-model-path [Traceanything_MODEL_DIR]
```
多卡并行
```bash
python preprocess/extract_trace_from_ta.py --dataset-root [DATA_SET_PATH] --ta-model-path [Traceanything_MODEL_DIR] --devices cuda:0,cuda:1,cuda:2
```
[DATASET_PATH]可以是单个任务的lerobot格式dataset，也可以是多任务的上级dataset目录。[Traceanything_MODEL_DIR]是目录，下面应该包含trace_anythng.pt\
⚠️ Important: 这一步必须在完成Lingbot-VA数据的预处理后进行，因为我们会根据latents中抽取到的帧进行traceanything features提取。

运行完成之后数据集目录下会出现一个trace/目录
```
your_dataset/
├── videos/
│   └── chunk-000/
│       └── observation.images.cam_high/
│           ├── episode_000000.mp4
│           └── ...
├── latents/
│   └── chunk-000/
│       └── observation.images.cam_high/
│           ├── episode_000000_0_450.pth    # named as episode_{index}_{start_frame}_{end_frame}.pth
│           └── ...
├── meta/
│   └── episodes.jsonl
├── trace/
│   └── chunk-000/
│       └── observation.images.cam_high/
│           ├── episode_000000_0_450.pth
│           └── ...
```

### Training
(1) 官方Lingbot-VA训练的命令如下
```bash
NGPU=8 bash script/run_va_posttrain.sh
```
(2) 4DWAM下训练的命令如下，读取的配置文件在wan_va/configs/config4dwam/train4dwam.py
```bash
NGPU=8 bash script/4dwam/run_posttrain.sh
```







# 🧩 Acknowledgments

This work builds upon several excellent open-source projects:

- [Wan-Video](https://github.com/Wan-Video) - Vision transformer backbone
- [MoT](https://github.com/facebookresearch/Mixture-of-Transformers) - Mixture-of-Transformers architecture
- The broader open-source computer vision and robotics communities

---

<!-- For questions, discussions, or collaborations:

- **Issues**: Open an [issue](https://github.com/robbyant/lingbot-va/issues) on GitHub
- **Email**: Contact Dr. [Qihang Zhang](https://zqh0253.github.io/) (liuhuan.zqh@antgroup.com) or Dr. [Lin Li](https://lilin-hitcrt.github.io/) (fengchang.ll@antgroup.com)  -->
