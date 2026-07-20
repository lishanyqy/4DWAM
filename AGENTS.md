# Python 环境约定

- `uv` 项目和 Python 环境位于绝对路径 `/soft/wangxi/4DWAM/lingbot-va`。
- 任何 Python 相关的运行、调试、测试、依赖安装或脚本执行，都必须使用该目录下的 `uv` 环境。
- 不需要先执行 `cd`；应通过 `uv --directory /soft/wangxi/4DWAM/lingbot-va ...` 明确指定项目目录。
- 运行 Python 命令时使用 `uv --directory /soft/wangxi/4DWAM/lingbot-va run`，例如：
  - `uv --directory /soft/wangxi/4DWAM/lingbot-va run python <script.py>`
  - `uv --directory /soft/wangxi/4DWAM/lingbot-va run pytest`
- 安装或更新依赖时也必须指定该目录，例如：`uv --directory /soft/wangxi/4DWAM/lingbot-va add <package>`。
- 不要使用系统 Python、全局 `pip` 或仓库中的其他虚拟环境。
