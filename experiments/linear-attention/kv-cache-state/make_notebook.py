"""把 linear_attn_kv_cache_lab.py 按 # %% 切成 notebook 并跑出输出。

切法：# %% 之前是 helper cell；每个含 gN_* 定义的 cell 末尾追加一次调用，
这样每组的代码与输出并排；最后的 main cell 换成「断言总数」打印。
"""

import re
import pathlib

import nbformat
from nbclient import NotebookClient

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "linear_attn_kv_cache_lab.py"
DST = HERE / "linear_attn_kv_cache_lab.ipynb"

lines = SRC.read_text(encoding="utf-8").splitlines(keepends=True)
cells = []
buf = []
for line in lines:
    if line.startswith("# %%"):
        cells.append("".join(buf))
        buf = []
    else:
        buf.append(line)
cells.append("".join(buf))

nb = nbformat.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
out_cells = []
for i, src in enumerate(cells):
    if not src.strip():
        continue
    if i == 0:
        body = src
    else:
        m = re.search(r"^def (g\d+_[a-z_]+)\(\):", src, re.M)
        if m:
            body = src.rstrip() + f"\n\n{m.group(1)}()\n"
        elif "def main(" in src:
            body = src.replace('if __name__ == "__main__":\n    main()\n', "")
            body = body.rstrip() + (
                '\n\nprint(f"断言总数：{len(OK)}/{len(OK) + len(FAIL)} 条通过")\n'
                'assert not FAIL, FAIL\n'
            )
        else:
            body = src
    out_cells.append(nbformat.v4.new_code_cell(body.strip("\n")))

# 首格补一句说明（markdown），方便读者在 GitHub 上直接看到结论
md = nbformat.v4.new_markdown_cell(
    "# 线性注意力的 KV cache：状态账本 / 容量 / 分块 / 前缀复用\n\n"
    "纯 CPU numpy 合成探针，固定 seed 20261105。七个分组各自带断言，"
    "最后一格打印断言总数。配套文章见 `README.md`；`results/stdout.txt` 是同一份代码的完整输出。"
)
nb.cells = [md] + out_cells

NotebookClient(nb, timeout=600, kernel_name="python3",
               resources={"metadata": {"path": str(HERE)}}).execute()
nbformat.write(nb, open(DST, "w", encoding="utf-8"))
print(f"wrote {DST.name} with {len(nb.cells)} cells")
