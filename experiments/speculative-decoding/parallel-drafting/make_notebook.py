"""把 parallel_drafting_lab.py 按 # %% 切成 notebook 并跑出输出。

第一格是模块 docstring（转 markdown 说明），第二段是 helper；
每个含 gN_* 定义的 cell 末尾追加一次调用，让代码与输出并排；
最后的 main cell 换成「断言总数」打印。
"""

import re
import pathlib

import nbformat
from nbclient import NotebookClient

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "parallel_drafting_lab.py"
DST = HERE / "parallel_drafting_lab.ipynb"

lines = SRC.read_text(encoding="utf-8").splitlines(keepends=True)
cells, buf = [], []
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

intro = cells[0].strip().strip('"""').strip()
out_cells = [nbformat.v4.new_markdown_cell(
    "# 并行草稿与自适应验证：六个可核验的合成探针\n\n"
    + intro + "\n\n"
    "纯 CPU / numpy / 固定 seed 20261117。六组各自带断言，最后一格打印断言总数。"
    "配套文章见 `README.md`；`results/stdout.txt` 是同一份代码的完整输出。"
)]

for i, src in enumerate(cells[1:], start=1):
    if not src.strip():
        continue
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

nb.cells = out_cells

NotebookClient(nb, timeout=600, kernel_name="python3",
               resources={"metadata": {"path": str(HERE)}}).execute()
nbformat.write(nb, open(DST, "w", encoding="utf-8"))
print(f"wrote {DST.name} with {len(nb.cells)} cells")
