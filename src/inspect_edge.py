"""Column-level diagnostic used to decide which Edge-IIoTset fields to drop.

Produces: edge_columns_report.txt
Paper: the feature-engineering decisions of Section 3.3.
"""
import pandas as pd
import config as C

cfg = C.DATASETS["edge_iiot"]
print(f"读取 {cfg['csv_path']} 的前 10 万行做诊断...")
df = pd.read_csv(cfg["csv_path"], nrows=100000, low_memory=False)
n = len(df)

rows = []
for col in df.columns:
    dtype = str(df[col].dtype)
    nunique = int(df[col].nunique(dropna=True))
    uratio = round(nunique / n, 4)
    miss = round(float(df[col].isna().mean()), 3)
    numeric_ratio = round(float(pd.to_numeric(df[col], errors="coerce").notna().mean()), 2)
    sample = df[col].dropna().iloc[0] if df[col].notna().any() else "—"
    sample = str(sample)[:22]
    rows.append((col, dtype, nunique, uratio, miss, numeric_ratio, sample))

rows.sort(key=lambda r: -r[3])

hdr = f"{'列名':<26}{'dtype':<9}{'唯一数':<8}{'唯一比':<8}{'缺失':<7}{'可数值':<7}示例"
lines = [f"总列数: {df.shape[1]} | 诊断行数: {n}", "", hdr, "-" * 92]
for r in rows:
    lines.append(f"{r[0]:<26}{r[1]:<9}{r[2]:<8}{r[3]:<8}{r[4]:<7}{r[5]:<7}{r[6]}")

out = "\n".join(lines)
print(out)
with open("edge_columns_report.txt", "w", encoding="utf-8") as f:
    f.write(out)
print("\n已保存到 edge_columns_report.txt —— 把这个文件或截图发我。")
