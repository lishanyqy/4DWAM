import json
from pathlib import Path

# ====== 可配置参数 ======
# 想切成多少段，就改这里
NUM_SEGMENTS = 0

HOME_PATH = Path('/cpfs01/projects-HDD/cfff-377aad6b032c_HDD/chenshuai/wenxuan')
# 输入输出文件路径
IN_PATH = Path(HOME_PATH, ".cache/huggingface/lerobot/robotwin/robotwin_multi_10_tasks/meta/episodes_bak.jsonl")
# OUT_PATH = Path(HOME_PATH, ".cache/huggingface/lerobot/robotwin/robotwin_multi_10_tasks/meta/episodes.jsonl")
OUT_PATH = Path(HOME_PATH, ".cache/huggingface/lerobot/robotwin/robotwin_multi_10_tasks/meta/episodes.jsonl")

def build_segments(length: int, num_segments: int, task_desc: str, episode_index: int):
    """
    Split episode to a number of segments.
    Generating action_config list.
    """
    if length <= 0:
        raise ValueError(f"Invalid length={length} for episode_index={episode_index}")

    step = length
    if num_segments == 0:
        # The shortest == 1
        return [
            {
                "start_frame": 0,
                "end_frame": length,
                "action_text": f"{task_desc} (single segment for short episode)",
            }
        ]
    else:
        step = length // num_segments

    segments = []
    start = 0
    for i in range(num_segments):
        if i == num_segments - 1:
            end = length  # end to length
        else:
            end = start + step

        segments.append(
            {
                "start_frame": int(start),
                "end_frame": int(end),
                "action_text": f"{task_desc}, segment {i+1}/{num_segments}.",
            }
        )
        start = end

    return segments


def main():
    if not IN_PATH.is_file():
        raise FileNotFoundError(f"Input file not found: {IN_PATH}")
    if IN_PATH == OUT_PATH:
        bak_file = Path(IN_PATH.parent,'episodes_bak.jsonl')
        # print(IN_PATH.parent)
        IN_PATH.rename(bak_file)

    if not OUT_PATH.exists():
        Path.touch(OUT_PATH)
    with IN_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            # 1. episode_index
            episode_index = data.get("episode_index", -1)

            length = data.get("length")
            if length is None:
                length = data.get("num_frames") or data.get("episode_len")
            if length is None:
                raise KeyError(
                    f"Episode {episode_index} does not have 'length'/'num_frames'/'episode_len' field."
                )

            # 3. task description: tasks[0]
            tasks = data.get("tasks", [])
            if isinstance(tasks, list) and len(tasks) > 0:
                task_desc = tasks[0]
            else:
                task_desc = f"Episode {episode_index}"

            # 4. construct action_config segment
            data["action_config"] = build_segments(int(length), NUM_SEGMENTS, task_desc, episode_index)

            # 5. write to a jsonl
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Done. New file written to: {OUT_PATH}")


if __name__ == "__main__":
    main()