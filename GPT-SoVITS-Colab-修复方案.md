# GPT-SoVITS Colab (T4) 训练卡死：根因分析与修复方案

> ⚠️ **本文件已被 `GPT-SoVITS-T4.ipynb` 取代**（那个是重写版，含 opencc 修复和 import 路径修复）。
> 这里仅作为根因分析参考保留。


> 结论先说：你的 Notebook 里同时存在两条**互不相容**的路线 ——
> ① 官方 WebUI 路线（0b → 0c → 1A → 1B）；
> ② 后来加的"手动绕过"路线（cell 20 / 22 / 24 / 25）。
> 第 ② 条在原理上**不可能**成功，第 ① 条又被 cell 14/15/16/22 的降级安装破坏了。
> 所以不管怎么试都会失败，并且失败得"看起来像成功"。

---

## 一、已确认的根因（附官方源码证据）

### 根因 1 ⛔ 手动 1B 用了不存在的参数 → 必然秒退
官方 `GPT_SoVITS/s1_train.py` 只接受 `-c/--config_file`：

```python
parser.add_argument("-c", "--config_file", type=str, default="configs/s1longer.yaml")
```

而你的 cell 24 写的是：

```bash
python GPT_SoVITS/s1_train.py --config_path /content/GPT-SoVITS/TEMP/tmp_s1.json
```

`--config_path` 不存在 → argparse 直接报错退出。
**这就是"很快结束 + 没有任何产物 = 静默失败"的直接原因。**

同理 cell 24 里的 `TEMP/tmp_s1.json` 是把 s2 的 JSON 复制过去当 s1 配置，
而 s1 用的是完全不同的 YAML 结构（`train/optimizer/data/model/inference`）。

### 根因 2 ⛔ 手动 1A 写的是**假数据**，训练结构上用不了
`GPT_SoVITS/module/data_utils.py` 是权威依据，`logs/<exp>/` 下训练真正需要：

| 文件 / 目录 | 由谁生成 | 是否必须 |
|---|---|---|
| `2-name2text.txt` | `1-get-text.py` | 必须，**4 列**：`name / phones / word2ph / norm_text` |
| `3-bert/*.pt` | `1-get-text.py` | 必须 |
| `4-cnhubert/*.pt` | `2-get-hubert-wav32k.py` | 必须（会被 `os.listdir`） |
| `5-wav32k/*.wav` | `2-get-hubert-wav32k.py` | 必须（会被 `os.listdir`） |
| `7-sv_cn/*.pt` | `2-get-sv.py` | v2Pro/v2ProPlus **必须**，源码里有 `assert os.path.exists(self.path7)` |
| `6-name2semantic.tsv` | `3-get-semantic.py` | 必须，首行表头 `item_name<TAB>semantic_audio`，每行是**真实** vq token 序列 |

你的 cell 22 只造了两样，而且都是错的：

```python
f1.write(f"{bname}\t{parts[3]}\n")      # 只有 2 列，官方要 4 列
f2.write(f"{bname}\t0 1 2 3\n")         # 4 个假 token，且没有表头
```

`3-bert` / `4-cnhubert` / `5-wav32k` / `7-sv_cn` 一个都没生成。
s2 训练 Dataset 第一件事就是 `os.listdir(self.path4)` + `path7` 的 assert，必然失败。

另外 cell 22 里这行是无效目录（多了中文）：

```bash
mkdir -p /content/GPT-SoVITS/logs/voice_liuyue_gpt-sovits应用/2a-org_spk
```

### 根因 3 ⛔ 环境被降级破坏（cell 14 / 15 / 16 / 22）
官方 `requirements.txt` 明确要求：

```
pytorch-lightning>=2.4
transformers>=4.51,<5
gradio<5
numpy<2.0
ctranslate2>=4.0,<5
funasr>=1.3.7
```

你的 Notebook 却做了这些事：

| cell | 做了什么 | 为什么致命 |
|---|---|---|
| 14 | conda 把 torch 降回 `2.1.2` | 与 `install.sh` 装的 torch/torchcodec 冲突 |
| 15 | `pip install --upgrade gradio` | 违反 `gradio<5` |
| 16 | `transformers==4.39.3` | 违反 `>=4.51`，1A 的 BERT 特征提取依赖新版 |
| 22 | `pytorch-lightning==1.9.5` | 违反 `>=2.4`，与 torch 2.x 不兼容 |

PL 1.9.5 与 torch 2.x 不兼容，**才**导致 cell 24 里那段 `torch.utils._pytree` 假模块补丁：

```python
mock_pytree = types.ModuleType("torch.utils._pytree")
mock_pytree.register_pytree_node = dummy_register
sys.modules["torch.utils._pytree"] = mock_pytree
```

这段补丁不是修复，而是**把真正的报错藏起来**，必须整段删除。

### 根因 4 ⛔ WebUI 是 v2Pro，但你可能没有 v2Pro 权重
当前 main 分支 `webui.py` 第 4 行：

```python
os.environ["version"] = version = "v2Pro"
```

注意：cell 17 的 `python webui.py v2` 里的 `v2` 其实是**语言参数**（`sys.argv[-1]`），
不是模型版本 —— 版本已经被硬编码成 v2Pro 了。

WebUI 启动时检查权重，缺失的会被**静默置空**（界面上不报错）：

```python
for key in pretrained_sovits_name.keys():
    if os.path.exists(pretrained_sovits_name[key]) == False:
        pretrained_sovits_name[key] = ""
```

v2Pro 需要这 6 个文件：

```
GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth
GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth
GPT_SoVITS/pretrained_models/s1v3.ckpt
GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/
GPT_SoVITS/pretrained_models/chinese-hubert-base/
GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt
```

你跳过了 "Download Model-can skip"。如果 `install.sh --source HF` 没把上面拉全，
1A 的 `3-get-semantic.py` 会抛 `FileNotFoundError(pretrained_s2G)`：

```python
if os.path.exists(pretrained_s2G):
    ...
else:
    raise FileNotFoundError(pretrained_s2G)
```

而 WebUI 是 `Popen(...)` + `wait()` 后直接回报 "finish"，
**报错只出现在跑 `python webui.py` 那个 cell 的输出流里**。
这就是"显示成功但没生成"。

> 定位 WebUI 静默失败的唯一办法：回到跑 `webui.py` 的 cell 输出里搜 `Traceback`。

### 根因 5 ⚠ 0b 切分目录名决定 list 文件名
`fasterwhisper_asr.py`：

```python
output_file_name = os.path.basename(input_folder)
output_file_path = os.path.abspath(f"{output_folder}/{output_file_name}.list")
```

list 名 = 切片目录名。WebUI 里 0c 与 1A 用的是同一个值，手敲错名字就会 1A 找不到 list。

### 附带发现：官方 0c 有个"写空文件也算成功"的坑
`fasterwhisper_asr.py` 的 `execute_asr()`：

```python
except Exception as e:
    print(e)
    traceback.print_exc()      # 单条转写失败 → 这一条被整个跳过
...
with open(output_file_path, "w") as f:
    f.write("\n".join(output))  # output 为空也照样写文件 + 打印"ASR 任务完成"
```

中文/粤语音频会走 FunASR（`only_asr`）；funasr 一旦不可用，
每条都抛异常被吞掉 → 生成一个**空的** `.list` → 界面仍显示完成。

---

## 二、修复方案（推荐：绕开 WebUI，直接跑官方脚本）

WebUI 在 Colab 上最大的问题是**把子进程报错藏进另一个 cell 的日志流**。
下面这套等价流程把每一步的报错都摆在明面上。

### 第 0 步：重置运行时
1. `Runtime → Disconnect and delete runtime`（彻底清掉被污染的 Python 环境）
2. 重跑 cell 3（挂载 Drive）→ cell 5（clone + `install.sh`）→ cell 7（condacolab，会重启）
3. 重启后重跑 cell 3 和 cell 5
4. **不要**再跑 cell 14 / 15 / 16 / 22 / 24 / 25 / 26 / 27

### 第 1 步：环境自检（必须是这些版本）

```bash
%%bash
source activate GPTSoVITS
python - <<'EOF'
import importlib, torch
for m in ["torch","torchaudio","pytorch_lightning","transformers","gradio",
          "numpy","faster_whisper","funasr","modelscope","ctranslate2","librosa","peft"]:
    try:
        mod = importlib.import_module(m)
        print("%-20s %s" % (m, getattr(mod, "__version__", "?")))
    except Exception as e:
        print("%-20s IMPORT-FAIL: %s" % (m, e))
print("cuda:", torch.cuda.is_available(), torch.version.cuda)
EOF
```

期望：`pytorch_lightning >= 2.4`、`transformers >= 4.51`、`gradio < 5`、`cuda: True`。
如果 `import kaldi` 之类失败，说明 `install.sh` 没跑完，重跑它。

### 第 2 步：预训练权重自检

```bash
%%bash
cd /content/GPT-SoVITS
for f in \
  GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth \
  GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth \
  GPT_SoVITS/pretrained_models/s1v3.ckpt \
  GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/config.json \
  GPT_SoVITS/pretrained_models/chinese-hubert-base/config.json \
  GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt \
  GPT_SoVITS/text/G2PWModel ; do
  [ -e "$f" ] && echo "OK   $f" || echo "MISS $f"
done
```

有 `MISS` 就先补下载，再往下走：

```bash
%%bash
cd /content/GPT-SoVITS && source activate GPTSoVITS
# 官方 install.sh 会在 opencc 处失败（requirements.txt 第 1 行的 --no-binary=opencc 强制源码编译），先打补丁
sed -i "/^--no-binary/d" requirements.txt
pip install -q --only-binary=opencc "opencc==1.4.2"
pip show opencc | head -2
bash install.sh --device CU126 --source HF     # 国内网络可换 --source HF-Mirror
```

> 想改回 v2（更省显存、官方 Colab notebook 原本就是 v2 时代）：
> 只要确认 `GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth`
> 与 `s1bert25hz-2kh-longer-epoch=68e-steps=502500.ckpt` 存在，
> 把下文的 `VER` 改成 `v2`，并跳过 `2-get-sv.py` 这一步即可。
> **v2 和 v2Pro 的 `logs/<exp>` 产物不能混用，改版本必须从预处理重做。**

### 第 3 步：做 ASR（官方脚本，报错可见）

```bash
%%bash
set -o pipefail
cd /content/GPT-SoVITS
source activate GPTSoVITS

SLICED="/content/drive/MyDrive/Colab/output"        # ← 0b 切分产物目录
OUT="/content/GPT-SoVITS/output/asr_opt"

python -s tools/asr/fasterwhisper_asr.py \
  -i "$SLICED" -o "$OUT" -s large-v3 -l zh -p float16
echo "exit=$?"

LIST="$OUT/$(basename "$SLICED").list"
echo "list = $LIST"
wc -l "$LIST"
head -3 "$LIST"
```

**验收**：`wc -l` 的行数 ≈ 切片数，且每行是 `绝对路径|说话人名|语言|文本`。
如果是 0 行 → 就是上面"写空 list 也算成功"的坑，往上看 traceback
（多半是 funasr / 模型下载问题，用 `HF_ENDPOINT=https://hf-mirror.com` 前缀重跑）。

### 第 4 步：1A / 1B / 1C 预处理（等价 WebUI 的 1A，但完全可见）

```bash
%%bash
set -e
cd /content/GPT-SoVITS
source activate GPTSoVITS

EXP="voice_liuyue"                                     # ← 实验名，自己定
VER="v2Pro"                                            # ← 或用 v2
BASE="/content/GPT-SoVITS"
LIST="$BASE/output/asr_opt/output.list"                # ← 上一步打印的 list 路径
OPT="$BASE/logs/$EXP"
mkdir -p "$OPT"

export inp_text="$LIST"  inp_wav_dir=""  exp_name="$EXP"  opt_dir="$OPT"
export i_part=0  all_parts=1  _CUDA_VISIBLE_DEVICES=0  is_half=True  version="$VER"
export bert_pretrained_dir="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
export cnhubert_base_dir="GPT_SoVITS/pretrained_models/chinese-hubert-base"
export sv_path="GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
export pretrained_s2G="GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth"
export s2config_path="GPT_SoVITS/configs/s2$VER.json"

echo "===== 1A-1 get-text ====="
python -s GPT_SoVITS/prepare_datasets/1-get-text.py
python - <<'EOF'
import os
opt = os.environ["opt_dir"]
src, dst = opt + "/2-name2text-0.txt", opt + "/2-name2text.txt"
open(dst, "w", encoding="utf8").write(open(src, encoding="utf8").read().strip("\n") + "\n")
os.remove(src)
print("merged ->", dst, sum(1 for _ in open(dst, encoding="utf8")), "lines")
EOF

echo "===== 1B-1 get-hubert-wav32k ====="
python -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py

if [ "$VER" = "v2Pro" ] || [ "$VER" = "v2ProPlus" ]; then
  echo "===== 1B-2 get-sv (Pro 专属) ====="
  python -s GPT_SoVITS/prepare_datasets/2-get-sv.py
fi

echo "===== 1C get-semantic ====="
python -s GPT_SoVITS/prepare_datasets/3-get-semantic.py
python - <<'EOF'
import os
opt = os.environ["opt_dir"]
src, dst = opt + "/6-name2semantic-0.tsv", opt + "/6-name2semantic.tsv"
body = open(src, encoding="utf8").read().strip("\n")
open(dst, "w", encoding="utf8").write("item_name\tsemantic_audio\n" + body + "\n")
os.remove(src)
print("merged ->", dst)
EOF

echo "===== 产物验收 ====="
for d in 3-bert 4-cnhubert 5-wav32k 7-sv_cn; do
  echo "  $d : $(ls "$OPT/$d" 2>/dev/null | wc -l) files"
done
wc -l "$OPT/2-name2text.txt" "$OPT/6-name2semantic.tsv"
head -1 "$OPT/6-name2semantic.tsv"
head -1 "$OPT/2-name2text.txt"
```

**验收标准**：`4-cnhubert`、`5-wav32k` 的文件数 = 切片数；
`7-sv_cn` 的文件数 = 切片数（v2Pro）；
`2-name2text.txt` 的每行有 **4 个 tab 列**；`6-name2semantic.tsv` 首行是表头。

### 第 5 步：训练（等价 WebUI 的 1B，参数名已修正）

```bash
%%bash
set -e
cd /content/GPT-SoVITS
source activate GPTSoVITS

python - <<'EOF'
import json, os, yaml

EXP, VER = "voice_liuyue", "v2Pro"          # ← 与上一步保持一致
ROOT = "/content/GPT-SoVITS"
exp_dir = os.path.join(ROOT, "logs", EXP)
os.makedirs(os.path.join(ROOT, "TEMP"), exist_ok=True)

# ---------- SoVITS (s2) ----------
os.makedirs(os.path.join(exp_dir, "logs_s2_" + VER), exist_ok=True)
cfg = os.path.join(ROOT, "GPT_SoVITS", "configs", "s2%s.json" % VER)
d = json.load(open(cfg, encoding="utf8"))
d["train"].update(dict(
    batch_size=4,            # T4 建议 4；显存不够改 2
    epochs=8,
    text_low_lr_rate=0.4,
    grad_ckpt=True,          # T4 省显存
    gpu_numbers="0",
    pretrained_s2G=os.path.join(ROOT, "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth"),
    pretrained_s2D=os.path.join(ROOT, "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth"),
    if_save_latest=True,
    if_save_every_weights=True,
    save_every_epoch=4,
    lora_rank=32,
))
d["model"]["version"] = VER
d["data"]["exp_dir"] = d["s2_ckpt_dir"] = exp_dir
d["save_weight_dir"] = os.path.join(ROOT, "SoVITS_weights_" + VER)
d["name"] = EXP
d["version"] = VER
json.dump(d, open(os.path.join(ROOT, "TEMP", "tmp_s2.json"), "w"), ensure_ascii=False)
print("tmp_s2.json ready")

# ---------- GPT (s1) ----------
os.makedirs(os.path.join(exp_dir, "logs_s1"), exist_ok=True)
s1 = yaml.safe_load(open(os.path.join(ROOT, "GPT_SoVITS/configs/s1longer-v2.yaml"), encoding="utf8"))
s1["train"].update(dict(batch_size=6, epochs=15, save_every_n_epoch=1))
s1["train"]["if_save_every_weights"] = True
s1["train"]["if_save_latest"] = True
s1["train"]["if_dpo"] = False
s1["train"]["half_weights_save_dir"] = os.path.join(ROOT, "GPT_weights_" + VER)
s1["train"]["exp_name"] = EXP
s1["pretrained_s1"] = os.path.join(ROOT, "GPT_SoVITS/pretrained_models/s1v3.ckpt")
s1["train_semantic_path"] = os.path.join(exp_dir, "6-name2semantic.tsv")
s1["train_phoneme_path"] = os.path.join(exp_dir, "2-name2text.txt")
s1["output_dir"] = os.path.join(exp_dir, "logs_s1_" + VER)
open(os.path.join(ROOT, "TEMP", "tmp_s1.yaml"), "w", encoding="utf8").write(
    yaml.dump(s1, default_flow_style=False, allow_unicode=True))
print("tmp_s1.yaml ready")
EOF

export _CUDA_VISIBLE_DEVICES=0 hz=25hz

echo "########## SoVITS (s2) 训练开始 ##########"
python -s GPT_SoVITS/s2_train.py --config /content/GPT-SoVITS/TEMP/tmp_s2.json

echo "########## GPT (s1) 训练开始 ##########"
python -s GPT_SoVITS/s1_train.py --config_file /content/GPT-SoVITS/TEMP/tmp_s1.yaml

echo "########## 产物 ##########"
ls -la /content/GPT-SoVITS/SoVITS_weights_v2Pro/ /content/GPT-SoVITS/GPT_weights_v2Pro/ 2>/dev/null
```

> 注意 `s2_train.py` 用 `--config`，`s1_train.py` 用 `--config_file`。这两个参数名不同，
> 你原来的 cell 24 就是在这里挂掉的。

### 第 6 步：备份回 Drive

```bash
%%bash
D="/content/drive/MyDrive/Colab/GPT_SoVITS_Models_Backup"
mkdir -p "$D"
cp -v /content/GPT-SoVITS/SoVITS_weights_v2Pro/* "$D"/ 2>/dev/null || true
cp -v /content/GPT-SoVITS/GPT_weights_v2Pro/* "$D"/ 2>/dev/null || true
ls -la "$D"
```

---

## 三、T4 (16GB) 参数建议

| 项 | 建议值 | 说明 |
|---|---|---|
| `grad_ckpt` | `True` | 省显存，v2Pro 的 `gin_channels=1024`，比 v2 大 |
| s2 `batch_size` | 4（不够就 2） | OOM 先降这个 |
| s1 `batch_size` | 6 ~ 8 | s1 是纯 Transformer，轻得多 |
| s2 `epochs` | 8 | 音色微调够用 |
| s1 `epochs` | 15 | |
| `fp16_run` | `True` | T4 支持 |

## 四、不要再做的事

1. ❌ 不要跑 cell 14（降级 torch 到 2.1.2）
2. ❌ 不要跑 cell 15（`--upgrade gradio`，会违反 `gradio<5`）
3. ❌ 不要跑 cell 16（`transformers==4.39.3`，违反 `>=4.51`）
4. ❌ 不要跑 cell 22（`pytorch-lightning==1.9.5`）
5. ❌ 不要用 cell 24 里的 `torch.utils._pytree` 假模块补丁 —— 它只是在掩盖错误
6. ❌ 不要手写 `2-name2text.txt` / `6-name2semantic.tsv` —— 它们的格式和内容都必须由官方脚本生成
7. ❌ 不要给 `s1_train.py` 传 `--config_path`，正确参数是 `--config_file`
8. ❌ 不要把 s2 的 JSON 配置拿去当 s1 的配置（s1 是 YAML，结构完全不同）

## 五、补充：WebUI 起不来（gradio / starlette 版本冲突）

你贴的报错：

```
File "gradio/routes.py", line 432, in main
    return templates.TemplateResponse(
File "starlette/templating.py", line 148, in TemplateResponse
    template = self.get_template(name)
TypeError: unhashable type: 'dict'
```

后面跟的 `ValueError: When localhost is not accessible, a shareable link must be created.`
是**连带结果**，不是第二个问题。

### 根因

`requirements.txt` 里两条依赖自己打架：

```text
gradio<5                      -> 只能装到 gradio 4.44.1（4.x 最后一版）
fastapi[standard]>=0.115.2    -> 没有上限 -> 装到最新 fastapi -> starlette 一路升到 1.x
```

- gradio 4.44.1 的 `gradio/routes.py:432` 用的是**老签名**：
  `templates.TemplateResponse("frontend/index.html", {"request": request, ...})`
- starlette 0.41 还保留兼容分支（`if isinstance(args[0], str):` → 打个 DeprecationWarning 后按老语义处理）；
  **starlette 1.x 删掉了这个分支**，`TemplateResponse(request, name, context)` 的第一个参数被当成 `Request`，
  于是 `name` 变成了那个 context dict → `self.env.get_template(dict)` → jinja2 缓存查 key → `unhashable type: 'dict'`。

行号可以一一对上：gradio 4.44.1 的 `routes.py` 第 432 行正是 `return templates.TemplateResponse(`；
starlette 1.7.0 的 `templating.py` 中 `template = self.get_template(name)` 在第 147/148 行。

### 连锁反应：`When localhost is not accessible`

`gradio/blocks.py` 的 `launch()` 里有：

```python
if _frontend and not wasm_utils.IS_WASM and not networking.url_ok(self.local_url) and not self.share:
    raise ValueError("When localhost is not accessible, a shareable link must be created. ...")
```

而 `gradio/networking.py` 的 `url_ok()` 是 `httpx.head(url)`，只认 200/401/302。
首页 `/` 因为上面那个异常返回 500 → 自检失败 → 抛这句话。**改端口 / 挂代理 / 加 share 都不解决根因。**

### 修复

```bash
pip install -q --no-deps "fastapi==0.115.14" "starlette==0.41.3"
```

- `fastapi 0.115.14` 的约束是 `starlette>=0.40,<0.47`；`0.41.3` 还带老签名兼容分支。
- `--no-deps`：这两个包的依赖（anyio / pydantic / jinja2 / typing_extensions）本来就装好了，
  带上依赖解析反而可能被 `fastapi-cli` 之类拖进冲突。
- 验证：`inspect.getsource(Jinja2Templates.TemplateResponse)` 里应当还有 `isinstance(args[0], str)`。

### 另外两个坑

1. **官方 notebook 最后那格写了 `export is_share=True`，你漏了。**
   而且 `%%bash` 起的子进程没有 IPython 调用栈，`gradio/utils.py: colab_check()` 翻不到 `google.colab`，
   所以 gradio 不会自动生成 Colab 端口代理 —— 想点开就得 `share=True`（公开链接），
   或者用 `google.colab.output.serve_kernel_port_as_iframe(9874)` 嵌进 notebook。
2. **`GPT_SoVITS/inference_cli.py` 本身是坏的**：它调用 `change_sovits_weights(...)` 却不迭代，
   而该函数是生成器 —— 等于**根本没加载 SoVITS 权重**。要命令行推理，得写
   `for _ in change_sovits_weights(path): pass`。

## 六、本机纯 CPU 推理要点（无显卡）

官方 Windows 流程（不是整合包）：

```pwsh
conda create -n GPTSoVits python=3.10 -y
conda activate GPTSoVits
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
pwsh -File install.ps1 -Device CPU -Source HF-Mirror
```

1. **版本必须对得上**：v2Pro 权重只有在带 `v2Pro` 的代码里才加载得进去
   （`config.py` 的 `pretrained_sovits_name` / `SoVITS_weight_version2root` 要有 `v2Pro` 键）。
   网上停在 v2/v3 的整合包会直接选不到模型。下载前先在包里搜 `v2Pro`。
2. **`install.ps1` 和 `install.sh` 有同一个 torchaudio 坑**：只装 `torch torchcodec`，
   而 `requirements.txt` 里是裸的 `torchaudio` → pip 会把 CPU 版 torch 换成 CUDA 版并装上新版 torchaudio。
   装完自检并锁回验证过的组合：

   ```pwsh
   python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__)"
   pip uninstall -y torch torchaudio torchcodec
   pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchaudio==2.8.0
   ```

3. **`--no-binary=opencc` 在 Windows 上同样会炸**（PyPI 有 `opencc-1.4.2-cp310-win_amd64.whl`，
   但那行强制源码编译）。删掉第一行再重跑安装脚本。
4. **不要照抄旧教程去改 `config.py`**：当前版本的 `config.py` 已经没有 `device = "cuda"` /
   `is_half = True` 这两行了，改成自动判断 —— `get_device_dtype_sm()` 在没有 CUDA 时
   直接 `return cpu, torch.float32`，`is_half = any(dtype == torch.float16 ...)`。
   纯 CPU 机器**什么都不用改**；只有「有显卡但想强制 CPU」才设 `CUDA_VISIBLE_DEVICES=""`。
5. 速度：CPU 约比显卡慢一个数量级，一句 10 秒的话几十秒是正常的；内存建议 ≥8GB。
