import json, re, pathlib, ast, sys
src = pathlib.Path(".")
def parse(fn):
    text = fn.read_text(encoding="utf-8")
    text = re.sub(r"^#@@INCLUDE (\S+)$", lambda m: (src / m.group(1)).read_text(encoding="utf-8").rstrip(), text, flags=re.M)
    cells, cur, kind = [], [], None
    def flush():
        if kind is None: return
        body = "\n".join(cur).strip("\n")
        if kind == "md":
            lines = [re.sub(r"^# ?", "", l) for l in body.split("\n")]
            cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": body})
    for line in text.split("\n"):
        if line.startswith("# %% [markdown]"): flush(); kind, cur = "md", []
        elif line.startswith("# %%"): flush(); kind, cur = "code", []
        else: cur.append(line)
    flush(); return cells
names = {"02_gemini_steam_agent": "02_gemini_steam_agent", "03_tabular_tf_tfdf": "03_tabular_tf_tfdf",
         "04_cnn": "04_cnn", "05_transfer_learning": "05_transfer_learning"}
for n in names:
    cells = parse(src / f"{n}.py")
    for c in cells:   # sintaksės patikra (Colab '!' komandas pakeičiame)
        if c["cell_type"] == "code":
            code = "\n".join("pass" if l.startswith("!") else l for l in c["source"].split("\n"))
            ast.parse(code)
    for c in cells: c["source"] = c["source"].splitlines(True)
    nb = {"cells": cells, "metadata": {"colab": {"provenance": []}, "accelerator": "GPU" if n[:2] in ("04", "05") else "None",
          "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = pathlib.Path("../notebooks") / f"{n}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out, len(cells), "cells OK")
