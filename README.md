# tools_aigc_GPT-SoVITS

用自己的声音微调 GPT-SoVITS：**Colab T4 上训练，本机推理**（纯 CPU 也能跑）。

| 文件 | 说明 |
| --- | --- |
| `GPT-SoVITS-T4.ipynb` | 主流程。从切片一路跑到出权重，每格都带自检和明确的错误提示 |
| `gsv_tts_local.py` | 本机命令行推理脚本，不启动 WebUI |
| `gsv_tts.ini` | 默认参数。填一次，以后只给文本就能合成 |
| `合成.bat` | 双击 / 拖 txt 上去就能合成，不用开终端 |
| `GPT-SoVITS-Colab-修复方案.md` | notebook 里每一处修复对应的官方 bug 和源码位置，出问题时看 |
| `RVC_Boss_GPT_SoVITS_Colab_Inference-not work.ipynb` | 官方原版 notebook，留档对照 |

---

## 第一步：在 Google Colab 训练

在 Colab 里打开 `GPT-SoVITS-T4.ipynb`（从 GitHub 打开或手动上传都行），
然后 `代码执行程序 → 更改运行时类型` 选 **T4 GPU**。

### 训练素材

- **同一个人**的干净人声：无 BGM、无混响、没有第二个人的声音插进来
- 时长 **5 分钟以上**（官方建议 5 分钟 ~ 1 小时）。低于 2 分钟也能跑完，但音色一定发飘
- 传一份到 Google Drive，路径记下来，第 1 格要填

### 单元格顺序

| # | 单元格 | 做什么 | 耗时 |
| --- | --- | --- | --- |
| 1 | `1. 基本设置` | 填 Drive 上的音频目录、实验名、版本、切片目录 | — |
| 2 | `2. 安装 conda` | **会重启，正常**。看到「session crashed」点重新连接；重连后**不要**再跑这格 | ~2 分钟 |
| 3 | `3. 重启之后`（3a→3e） | 重挂 Drive → clone 代码 → 装依赖 → 修官方两处 bug | 10~20 分钟 |
| 4 | `4. 体检` | 确认环境和预训练权重都齐 | ~1 分钟 |
| 5 | `5. 0b 语音切片` | 把整段音频切成小片 | 几分钟 |
| 6 | `6. 0c 语音识别` | 生成 `output.list` | 3~10 分钟 |
| 7 | `7. 数据集预处理` | 生成 BERT / hubert / semantic 训练特征 | 5~15 分钟 |
| 8 | `8. 训练` | SoVITS + GPT 两轮 | 40~90 分钟 |
| 9 | `9. 备份到 Google Drive` | **必跑**，Colab 随时会断线清环境 | 几分钟 |

第 10 节是推理。训练完可以直接跑 **10c** 先试听效果。

### 每步的验收点（不通过就别往下走）

- **第 6 步**：`output.list` 的**行数 ≈ 切片数**。官方脚本单条转写失败会被 `except` 吞掉，
  全部失败时会生成一个**空 list，并且照样打印「ASR 任务完成」**。行数是 0 就是失败了，往上翻 traceback。
- **第 7 步**：会打印 `3-bert` / `4-cnhubert` / `5-wav32k` / `7-sv_cn` 的数量，四个都应该等于切片数。
- **第 8 步**：产物在 `SoVITS_weights_v2Pro\` 和 `GPT_weights_v2Pro\`。
  **去目录里确认文件真的生成了**，别只看日志。
- **第 9 步**：确认 Drive 备份目录里有 `.ckpt` 和 `.pth`。

### 训练完要下载到本机的两个文件

```
GPT_weights_v2Pro\voice_liuyue-e15.ckpt        (~155 MB)
SoVITS_weights_v2Pro\voice_liuyue_e8_s176.pth  (~135 MB)
```

---

## 第二步：本机推理

### 2.1 下载整合包

```powershell
# 下到本仓库目录就行（已在 .gitignore 里，不会提交）
curl.exe -L -C - -o GPT-SoVITS-v2pro-20250604.7z "https://hf-mirror.com/lj1995/GPT-SoVITS-windows-package/resolve/main/GPT-SoVITS-v2pro-20250604.7z?download=true"
```

- 7.8 GB。断了就重跑同一条命令，`-C -` 会续传
- 官方源把域名换成 `huggingface.co` 即可（两个都实测可用）
- **别下 `...-nvidia50.7z`**（8.4 GB，是给 RTX 50 系显卡的）
- **别去 B 站搜「v2Pro 整合包」**，那些是第三方重打包的，来源和完整性没法保证

### 2.2 解压

解压到纯英文路径。压缩包 7.8 GB，解压后约 15~20 GB，磁盘留 30 GB 余量。

根目录里应该有：`webui.py`、`GPT_SoVITS\`、`runtime\`、`go-webui.bat`。

### 2.3 放脚本和权重

1. 把 `gsv_tts_local.py`、`gsv_tts.ini`、`合成.bat` 三个文件复制进整合包根目录
   （就是有 `webui.py` 的那层），再建一个 `ref\` 目录放参考音频
2. 权重放好：

```
<整合包根>\
  GPT_weights_v2Pro\voice_liuyue-e15.ckpt
  SoVITS_weights_v2Pro\voice_liuyue_e8_s176.pth
```

不给 `--gpt` / `--sovits` 时，脚本会自动挑这两个目录里**最新**的文件。

### 2.4 参考音频（音色样本）

推理要额外给**一段**参考音频，它就是"音色样本"。三个参数的分工：

| 参数 | 是什么 |
| --- | --- |
| `--ref` | 你额外准备的**一段** 3~10 秒干净人声（和训练数据没关系也行） |
| `--ref-text` | **这段音频里原封不动说的话**（不是你要合成的内容） |
| `--text` | **你要合成的内容** |

`--ref` 的要求：

- **3~10 秒，5~7 秒最稳。** 太短音色学不准；太长容易跑偏、变调，也慢
- 只有你一个人的声音，中间没插进别人的话
- 无 BGM、无混响、无电流声
- 不要剪在半句话中间
- wav / mp3 / flac 都行，采样率随便

`--ref-text` 必须和这段音频**逐字一致**（含标点、数字怎么读的），写错会明显跑调或者念糊。

#### 为什么有了权重还要参考音频？

因为它俩分工不同：**权重学的是"这个人的发音习惯和音色倾向"，参考音频是每次合成时喂进去的
"样板"**，模型照着这段样板决定这一次的音色。所以：

- 权重负责"像不像你"，参考音频负责"这一次以哪个状态为准"
- **参考音频准备一次就够了，以后一直用同一段**，不用每次都换
- 换一段不同语速/情绪的参考音频，出来会略有差异 —— 这是它比传统 TTS 灵活的地方

> 不需要 `logs\...\5-wav32k\`。那是训练中间产物，推理一行都不碰。

#### 语种

`--ref-lang` 是参考音频的语种，`--lang` 是要合成文本的语种，都默认 `zh`。
支持：`zh` 中文 / `en` 英文 / `ja` 日文 / `yue` 粤语 / `ko` 韩文 /
`zh-en` 中英混合 / `ja-en` 日英混合 / `multi` 自动识别语种。

两者**可以不一样** —— 用中文参考音频念英文句子是可以的，只是会带一点中文口音。
但 `--ref-lang` 必须和参考音频的**真实**语种一致，不一致会念飞。

#### 文本长度

没有硬上限，但**一段太长一定出问题**（重复字、漏字、突然变调）。
脚本默认 `--cut punct`，会按标点自动切句、合成完再拼起来，所以整段丢进去也行。
如果文本里几乎没有标点，用 `--cut 50字` 强制按 50 字左右断开。

#### `--list` 到底做什么

它**只提供「文件名 → 原文」的映射，不提供音频**。
脚本读 `output.list` 时只取文件名那一段，前面的绝对路径直接丢掉，
所以 `/content/drive/MyDrive/...` 这种路径它**不会**去访问，本地也找不到。

因此 `--ref-dir` + `--list` 的用法是：

- 你先把**切片 wav 文件**放到本地某个目录（`--ref-dir`）
- 脚本从那个目录里挑一段 3~10 秒的
- 再用它的文件名去 `output.list` 里查出原文，自动当 `--ref-text`

好消息：你之前那张 `output.list` 里写的是
`/content/drive/MyDrive/Colab/output/...`，说明**切片是写在 Google Drive 上的**，
大概率还在。用 Google Drive 桌面版同步下来，或网页端把 `MyDrive\Colab\output` 整个下载下来，
就能直接复用：

```powershell
.\runtime\python.exe gsv_tts_local.py --root . --version v2Pro `
  --ref-dir "G:\My Drive\Colab\output" `
  --list "G:\My Drive\Colab\output\output.list" `
  --text "今天天气不错，我们一起出去走走吧。"
```

懒得找也没关系，自己剪一段 5~7 秒的 + 手写 `--ref-text` 效果完全一样。
### 2.5 跑

```powershell
cd <整合包根>
.\runtime\python.exe gsv_tts_local.py --root . --version v2Pro `
  --ref ".\ref\my7s.wav" `
  --ref-text "那段音频里原封不动说的话" `
  --text "今天天气不错，我们一起出去走走吧。"
```

注意用的是整合包自带的 `runtime\python.exe`，**不是**系统里的 `python`。
产物默认在 `<整合包根>\tts_out\`。

CPU 上第一次加载要读 BERT + hubert + 两个权重，等 20~60 秒是正常的；
之后每合成一句 10 秒的话还要再花 20~60 秒。慢是正常的，不是卡死。

### 2.6 配置一次，以后只给文本

把 `gsv_tts.ini` 里的 `root` / `version` / `ref` / `ref-text` 填好之后，日常合成只需要给文本。
**命令行上写的参数永远覆盖 ini**，所以临时想换参考音频直接加 `--ref` 就行。

三种日常用法：

**A. 一整段文案 —— 一行一句存成 txt，一次合成**

```powershell
.\runtime\python.exe gsv_tts_local.py --text-file 文案.txt
```

`文案.txt` 用 UTF-8 保存（记事本的"UTF-8"选项），一行一句。
长文案建议一行 10~50 字、按自然停顿断句，不要一整段塞一行。

**合成前脚本会打印「待合成 N 句」和前 3 句的内容**，扫一眼就能确认喂进去的是对的文本。

产物取决于 `merge`：

- `merge = true`（ini 里默认）：全部合完拼成**一个** `tts_out\merged_xxx.wav`，段间留 0.25 秒静音
- `merge = false`：每句一个 `001_xxx.wav` / `002_xxx.wav`，方便挑出哪句念坏了

**B. 双击 `合成.bat` —— 直接合成剪贴板里的文字**

最快的路径：复制文案 → 双击 `合成.bat` → 出音频。双击时不带参数，会自动读 Windows 剪贴板，
一个换行算一句。

> ⚠️ 双击时读的是**剪贴板当前内容**。如果你刚复制的是日志、代码或别的东西，它就会念那个。
> 合成前先看脚本打印的「待合成 N 句」确认一下。

**C. 把 txt 拖到 `合成.bat` 上**

等价于 A，不用开终端。

也可以直接给一句话，或者读剪贴板：

```powershell
.\runtime\python.exe gsv_tts_local.py --text "今天天气不错"
.\runtime\python.exe gsv_tts_local.py --clip
```

想换某一轮权重，命令行覆盖即可，ini 不用改：

```powershell
.\runtime\python.exe gsv_tts_local.py --text-file 文案.txt `
  --gpt ".\GPT_weights_v2Pro\voice_liuyue-e10.ckpt"
```

### 2.7 常用参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--gpt` / `--sovits` | 自动挑最新 | 指定某一轮权重 |
| `--ref-dir` + `--list` | — | 给切片目录 + `output.list`，参考音频和原文全自动 |
| `--text-file` | — | 批量合成，一行一句（UTF-8） |
| `--ref-lang` / `--lang` | zh / zh | 参考音频语种 / 合成文本语种 |
| `--cut` | punct | 长文本切句：punct / 50字 / 4句 / 中文句号 / 英文句号 / none |
| `--merge` | 见 ini | 把所有句子拼成一个 wav（ini 里默认 `merge = true`） |
| `--temperature` | 1.0 | 越大越随机。念得太平→1.1~1.2；重复漏字→0.8~0.9 |
| `--top-k` | 15 | 调大到 20~30 通常更稳 |
| `--seed` | -1 | 固定随机种子，方便对比参数改动 |
| `--speed` | 1.0 | 语速 |
| `--out` | `<root>\tts_out` | 输出目录 |

批量试不同 GPT 权重（想找出哪一轮最稳时特别省事）：

```powershell
.\runtime\python.exe gsv_tts_local.py --root . --version v2Pro `
  --ref-dir ".\ref" --text-file 句子.txt `
  --gpt ".\GPT_weights_v2Pro\voice_liuyue-e10.ckpt"
```

权重怎么挑：GPT 先试 `e15`；语调发飘、重复字、突然变调就换 `e10` 或 `e8`。SoVITS 用 `e8_s176`。

---

## 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 报「没找到 soundfile —— 你多半没有 activate 仓库的 conda 环境」 | Python 用错了 | 用 `.\runtime\python.exe` |
| 报「不知道 GPT-SoVITS 仓库在哪」 | 没给 `--root`，又不在仓库目录里 | 加 `--root "D:\GPT-SoVITS-v2pro"` |
| 报「不像 GPT-SoVITS 仓库」 | `--root` 指错了层 | 要指到同时有 `webui.py` 和 `GPT_SoVITS\` 的那层 |
| 找不到 GPT / SoVITS 权重 | 权重没放对目录 | 放进 `GPT_weights_v2Pro\` / `SoVITS_weights_v2Pro\` |
| 音色不像本人 / 念糊 / 重复字 / 突然变调 | 参考音频太长太脏，或原文和音频不一致 | 换 5~7 秒干净片段；原文逐字核对；temperature 降到 0.8~0.9 |
| 报缺 ffmpeg | 整合包自带，正常不会遇到 | 确认没把整合包目录拆散 |
| 第一次跑卡很久 | CPU 加载模型慢 | 等，别 Ctrl+C |
| 有显卡但想强制走 CPU | — | 加 `--cpu`，**不要去改 `config.py`** |
| 生成出一堆 wav 而不是一个 | 文本是多行，且 `merge = false` | ini 里改成 `merge = true`，或加 `--merge` |
| 合成出来的内容完全不是想要的文案 | `--clip` 读到的是剪贴板里别的东西（比如刚复制的日志） | 看合成前的「待合成」预览，重新复制正确的文案 |
| 双击 `合成.bat` 一闪而过 | 没放在整合包根目录 | 和 `gsv_tts_local.py` 放在同一层，也就是有 `webui.py` 的那层 |
| `--clip` 报读不到剪贴板 | 剪贴板是空的，或系统里没有 powershell | 先复制一段文字；或改用 `--text-file` |
| 改了 `gsv_tts.ini` 没生效 | 命令行上写了同一个参数（命令行优先） | 把命令行上那个参数去掉 |

> 网上流传的「打开 `config.py`，把 `device = "cuda"` 改成 `"cpu"`、`is_half = True` 改成 `False`」
> 是**旧版**攻略。当前版本 `config.py` 里**没有这两行**了，改成自动判断：
> 没有 CUDA 时直接返回 `cpu, torch.float32`。纯 CPU 机器什么都不用改。

---

## 附：Colab 额度用完之后

训练成果只要第 9 步备份过，就不会丢。以后想再回 Colab：

第 1 格 → 3a 挂 Drive → 3b clone → 3c 装依赖 → 3d → 3e
→ **跳过第 4~8 步**（不用重新切片、ASR、预处理、训练）
→ 跑 10-0（把 Drive 上的权重恢复到 `/content`）→ 10c 合成。

注意新会话的 `/content` 是空的，切片和 `logs\` 都没了。`output.list` 只能提供
「文件名 → 原文」的映射，音频本身不在的话查不出东西，所以 10c 得手打 `--ref-text`，
或者先把一段参考音频传到 Drive。

---

## 附：源码 + conda 路线（不推荐，仅供参考）

整合包已经够用了，这条路只在你想自己控制代码版本时才走。Windows 上它有两个坑：
`requirements.txt` 第一行 `--no-binary=opencc` 会触发源码编译（要 MSVC），
而 `install.ps1` 的 CPU 分支只装 torch，不装 torchaudio。

```powershell
conda create -n GPTSoVits python=3.10 -y
conda activate GPTSoVits
git clone https://github.com/RVC-Boss/GPT-SoVITS.git D:\GPT-SoVITS
cd D:\GPT-SoVITS

# 先删掉那行 --no-binary=opencc，否则 opencc 要现场编译
(Get-Content requirements.txt) | Where-Object { $_ -notmatch '^--no-binary' } | Set-Content requirements.txt

pwsh -File install.ps1 -Device CPU -Source HF-Mirror
```

装完自检：

```powershell
python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__)"
python -c "import torchaudio; w, sr = torchaudio.load(r'D:\GPT-SoVITS\ref\my7s.wav'); print(w.shape, sr)"
```

不通过就锁回配套的 CPU 版（和云端 3e 步同一个版本）：

```powershell
pip uninstall -y torch torchaudio torchcodec
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 torchaudio==2.8.0
```

之后同样用 `python gsv_tts_local.py --root . --version v2Pro ...`。



