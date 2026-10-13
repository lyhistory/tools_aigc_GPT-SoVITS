#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gsv_tts_local.py — 不启动 WebUI，直接在命令行用 GPT-SoVITS 合成语音（本机 / 云端通用）

它不复制任何官方代码：内部调用的就是 GPT_SoVITS/inference_webui.py 里
和 WebUI「合成语音」按钮**完全相同**的那几个函数
（change_gpt_weights / change_sovits_weights / get_tts_wav）。

前置条件：本机已有装好的 GPT-SoVITS 仓库
    conda create -n GPTSoVits python=3.10 -y
    conda activate GPTSoVits
    git clone https://github.com/RVC-Boss/GPT-SoVITS.git
    cd GPT-SoVITS
    pwsh -File install.ps1 -Device CPU -Source HF-Mirror

最小用法（权重自动去 GPT_weights_<version> / SoVITS_weights_<version> 里找最新的）：

    python gsv_tts_local.py --root "D:/GPT-SoVITS" ^
        --ref "D:/ref/我的5秒参考.wav" ^
        --ref-text "那段音频里原封不动说的话" ^
        --text "今天天气不错，我们一起出去走走吧。"

只给一个切片目录，参考音频和参考原文都自动搞定（原文从 list 里按文件名查）：

    python gsv_tts_local.py --root "D:/GPT-SoVITS" ^
        --ref-dir "D:/colab_slices" --list "D:/colab_slices/output.list" ^
        --text "今天天气不错"

批量合成（一行一句；试「哪一轮 GPT 权重最稳」时特别省事）：

    python gsv_tts_local.py --root ... --ref-dir ... --list ... --text-file 句子.txt

换权重 / 调参数：

    --gpt   .../GPT_weights_v2Pro/voice_liuyue-e15.ckpt
    --sovits .../SoVITS_weights_v2Pro/voice_liuyue_e8_s176.pth
    --temperature 1.0  --top-k 15  --top-p 1.0  --speed 1.0  --seed 42
    --cpu          # 机器上有显卡但想强制用 CPU
    --out 目录      # 默认 <root>/tts_out
"""
import argparse
import os
import sys
import time


def _fix_console():
    """Windows 控制台默认不是 UTF-8，print 中文会直接 UnicodeEncodeError"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_fix_console()

LANG_ALIAS = {
    "zh": "all_zh", "cn": "all_zh", "中文": "all_zh",
    "en": "en", "英文": "en",
    "ja": "all_ja", "jp": "all_ja", "日文": "all_ja",
    "yue": "all_yue", "粤语": "all_yue",
    "ko": "all_ko", "韩文": "all_ko",
    "zh-en": "zh", "中英混合": "zh",
    "ja-en": "ja", "日英混合": "ja",
    "multi": "auto", "多语种混合": "auto",
}

# 长文本切句方式。官方 get_tts_wav 的 how_to_cut 参数默认是「不切」，
# 也就是整段当成一句去合成 —— 段落一长就容易重复、漏字、变调。
# 这里默认改成「按标点符号切」，并按官方 i18n 的键取值（必须走 iw.i18n() 才能和内部比较对上）。
CUT_ALIAS = {
    "punct": "按标点符号切", "标点": "按标点符号切", "按标点符号切": "按标点符号切",
    "50": "凑50字一切", "50字": "凑50字一切", "凑50字一切": "凑50字一切",
    "4": "凑四句一切", "4句": "凑四句一切", "凑四句一切": "凑四句一切",
    "zh": "按中文句号。切", "中文句号": "按中文句号。切", "按中文句号。切": "按中文句号。切",
    "en": "按英文句号.切", "英文句号": "按英文句号.切", "按英文句号.切": "按英文句号.切",
    "none": "不切", "不切": "不切",
}

REPO_MARKERS = ["webui.py", "GPT_SoVITS"]


def die(msg):
    print("")
    print("!! " + msg)
    print("")
    raise SystemExit(1)


def build_parser():
    ap = argparse.ArgumentParser(
        description="GPT-SoVITS 命令行推理（不启动 WebUI）",
    )
    ap.add_argument("--root", help="GPT-SoVITS 仓库根目录（也可以设环境变量 GSV_ROOT）")
    ap.add_argument("--version", default="v2Pro", help="权重版本，默认 v2Pro；决定去哪个权重目录找最新的")
    ap.add_argument("--gpt", help="GPT 权重 .ckpt（不给就用权重目录里最新的）")
    ap.add_argument("--sovits", help="SoVITS 权重 .pth（不给就用权重目录里最新的）")
    ap.add_argument("--ref", help="参考音频，3~10 秒干净人声")
    ap.add_argument("--ref-dir", help="参考音频目录；没给 --ref 时自动在这里挑一段 3~10 秒的")
    ap.add_argument("--ref-text", help="参考音频里原封不动说的话（写错会明显跑调）")
    ap.add_argument(
        "--ref-trim", nargs=2, type=float, metavar=("START", "END"),
        help="参考音频不在 3~10 秒时，先剪出第 START~END 秒再用（单位秒，比如 --ref-trim 0 9）",
    )
    ap.add_argument("--list", dest="list_path", help="output.list；没给 --ref-text 时用它按文件名查原文")
    ap.add_argument("--ref-lang", default="zh", help="参考音频语种，默认 zh")
    ap.add_argument("--lang", default="zh", help="要合成文本的语种，默认 zh")
    ap.add_argument("--text", action="append", default=[], help="要合成的文本，可以给多次")
    ap.add_argument("--text-file", help="一行一句的文本文件（UTF-8），和 --text 会合并")
    ap.add_argument("--out", help="输出目录，默认 <root>/tts_out")
    ap.add_argument("--temperature", type=float, default=1.0, help="越大越随机，WebUI 默认 1.0")
    ap.add_argument("--top-k", type=int, default=15, help="WebUI 默认 15")
    ap.add_argument("--top-p", type=float, default=1.0, help="WebUI 默认 1.0")
    ap.add_argument("--speed", type=float, default=1.0, help="语速")
    ap.add_argument("--seed", type=int, default=-1, help="-1 表示随机；固定它便于复现")
    ap.add_argument("--cpu", action="store_true", help="强制用 CPU（有显卡时才有意义）")
    ap.add_argument("--half", action="store_true", help="强制半精度；CPU 上别加")
    ap.add_argument("--merge", action="store_true", help="把所有句子拼成一个 wav（段间留 0.25 秒静音）")
    ap.add_argument("--config", help="ini 配置文件；默认用脚本旁边的 gsv_tts.ini")
    ap.add_argument("--clip", action="store_true", help="把 Windows 剪贴板里的文字当作要合成的内容（一行一句）")
    ap.add_argument(
        "--cut", default="punct",
        help="长文本怎么切句：punct/标点(默认) | 50字 | 4句 | 中文句号 | 英文句号 | none(不切)",
    )
    return ap


def parse_args():
    ap = build_parser()
    return ap.parse_args(), ap


def apply_config(parser, args, path):
    """把 ini 里的值填进来。优先级：命令行 > ini > 代码默认值。

    只覆盖「命令行上没显式写过」的参数，所以临时想换参考音频照样能加 --ref 覆盖。
    ini 可以不带 section 头，直接一行一个 key = value；# 开头是注释。
    key 写命令行参数名即可（ref / ref-text / ref-dir / list / text-file / version ...），
    内部名（比如 list_path）也认。
    """
    if not path or not os.path.exists(path):
        return None
    import configparser
    with open(path, encoding="utf-8-sig") as f:
        txt = f.read()
    if "[" not in txt:
        txt = "[gsv]\n" + txt
    cp = configparser.ConfigParser(interpolation=None)
    cp.optionxform = lambda s: s.strip().lower().replace("-", "_")
    try:
        cp.read_string(txt)
    except Exception as e:
        die("配置文件读不了: %s\n   %r" % (path, e))
    sec = cp[cp.sections()[0]]

    # --list 这种参数命令行名字和内部名字不一样（list -> list_path），
    # 直接用内部名字去查会漏掉，所以这里建一张 命令行名 -> 内部名 的表。
    opt2dest = {}
    for act in parser._actions:
        for s in act.option_strings:
            opt2dest[s.lstrip("-").replace("-", "_")] = act.dest

    def dest_of(key):
        return opt2dest.get(key, key)

    provided = set()
    for tok in sys.argv[1:]:
        if tok.startswith("--"):
            n = tok[2:].split("=", 1)[0].replace("-", "_")
            provided.add(n)
            provided.add(opt2dest.get(n, n))

    used = []
    for k, v in sec.items():
        dest = dest_of(k)
        if k in provided or dest in provided:
            continue
        if not hasattr(args, dest):
            die(
                "配置文件里的 %s 不是有效参数。\n   能写的 key：\n     %s"
                % (k, "  ".join(sorted(opt2dest)))
            )
        cur = getattr(args, dest)
        if isinstance(cur, list):
            continue
        try:
            if isinstance(cur, bool):
                nv = str(v).strip().lower() in ("1", "true", "yes", "on", "y")
            elif isinstance(cur, int):
                nv = int(str(v).strip())
            elif isinstance(cur, float):
                nv = float(str(v).strip())
            else:
                nv = v
        except Exception:
            die("配置文件里 %s = %s 解析不了" % (k, v))
        setattr(args, dest, nv)
        used.append(dest)
    return used

def text_from_clipboard():
    """读 Windows 剪贴板（复制一段文案直接合成）。"""
    import subprocess
    cmd = "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-Clipboard -Raw"
    for exe in ("powershell", "pwsh"):
        try:
            r = subprocess.run(
                [exe, "-NoProfile", "-NonInteractive", "-Command", cmd],
                capture_output=True, timeout=20,
            )
        except FileNotFoundError:
            continue
        except Exception as e:
            die("读剪贴板失败: %r" % (e,))
        if r.returncode == 0:
            return r.stdout.decode("utf-8-sig", "replace")
    die("读不到剪贴板：没找到 powershell，或者剪贴板是空的。")


def guess_root(script_dir):
    for cand in [script_dir, os.getcwd()]:
        if all(os.path.exists(os.path.join(cand, m)) for m in REPO_MARKERS):
            return cand
    return None


def newest(d, ext):
    if not os.path.isdir(d):
        return None
    fs = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(ext)]
    return max(fs, key=os.path.getmtime) if fs else None


def safe_name(s, n=12):
    """把一段文本变成能当文件名的短标签（只留字母数字和汉字）"""
    keep = []
    for ch in str(s):
        if ch.isalnum():
            keep.append(ch)
        if len(keep) >= n:
            break
    return "".join(keep)


def merge_segments(segs, gap_sec=0.25):
    """把 [(sr, wav), ...] 按顺序拼成一段，段间补一点静音，不然听起来会赶"""
    import numpy as np
    sr0 = segs[0][0]
    parts = []
    for k, (sr, wav) in enumerate(segs):
        if k:
            parts.append(np.zeros(int(gap_sec * sr0), dtype=wav.dtype))
        parts.append(wav)
    return sr0, np.concatenate(parts)


def load_name2text(list_path):
    """output.list 每行是 路径|说话人|语种|文本；官方下游脚本都只取 basename"""
    name2text = {}
    if not list_path:
        return name2text
    if not os.path.exists(list_path):
        print("!! 找不到 list 文件: %s（不影响，只是 --ref-text 得手打）" % list_path)
        return name2text
    with open(list_path, encoding="utf-8-sig") as f:
        for line in f.read().strip("\n").split("\n"):
            part = line.split("|", 3)
            if len(part) == 4:
                name2text[os.path.basename(part[0])] = part[3].strip()
    print("[list] 读到 %d 条「文件名 -> 原文」：%s" % (len(name2text), list_path))
    return name2text


# soundfile 底层是 libsndfile，它认得这些格式
SF_EXTS = (".wav", ".flac", ".ogg", ".oga", ".aiff", ".aif", ".au", ".w64", ".caf", ".mp3", ".opus")
# 参考音频允许出现的格式（不在这两个里的也会试着读一下）
AUDIO_EXTS = SF_EXTS + (".m4a", ".m4b", ".aac", ".mp4", ".m4v", ".wma", ".amr", ".3gp", ".mov", ".webm", ".mkv")


def find_ffmpeg(root):
    """找 ffmpeg。整合包通常放在 runtime\\，也允许放根目录或直接用 PATH 里的。"""
    import shutil
    for c in [
        os.path.join(root, "ffmpeg.exe"),
        os.path.join(root, "ffmpeg"),
        os.path.join(root, "runtime", "ffmpeg.exe"),
        os.path.join(root, "runtime", "ffmpeg"),
        os.path.join(root, "runtime", "Library", "bin", "ffmpeg.exe"),
    ]:
        if os.path.exists(c):
            return c
    for name in ("ffmpeg.exe", "ffmpeg"):
        p = shutil.which(name)
        if p:
            return p
    return None


def audio_duration(path):
    """取音频时长。soundfile 读不了 m4a 之类，失败了就退回 torchaudio。"""
    try:
        import soundfile as sf
        return sf.info(path).duration
    except Exception:
        pass
    try:
        import torchaudio
        info = torchaudio.info(path)
        return info.num_frames / float(info.sample_rate)
    except Exception:
        return None


def norm_trim(v):
    """把 --ref-trim 的值统一成 (开始秒, 结束秒)。

    命令行写法是 --ref-trim 0 9（argparse 给一个 list），
    ini 里写 ref-trim = 0 9（configparser 给一个字符串），两种都兼容。
    """
    if v is None:
        return None
    if isinstance(v, str):
        import re
        parts = [p for p in re.split(r"[\s,，]+", v.strip()) if p]
    else:
        parts = [str(x) for x in v]
    if len(parts) != 2:
        die("ref-trim 要写两个数（开始秒 结束秒），比如 0 9；现在写的是 %r" % (v,))
    try:
        a, b = float(parts[0]), float(parts[1])
    except ValueError:
        die("ref-trim 的两个值必须是秒数，比如 0 9；现在写的是 %r" % (v,))
    if a < 0 or b <= a:
        die("ref-trim 的范围不对：开始 %g、结束 %g（结束要大于开始，且都不能是负数）" % (a, b))
    return a, b


def to_wav(path, root, trim=None):
    """参考音频必须是 libsndfile 读得了的格式。

    Windows 录音默认就是 m4a，libsndfile 完全不认（报 Format not recognised）。
    这里用 ffmpeg 转一份 32kHz 单声道 PCM wav 到系统临时目录，原文件不动。
    传 trim=(开始秒, 结束秒) 时顺便只取这一段（官方只接受 3~10 秒的参考音频）。
    返回 (真正要用的路径, 是否转换过)。
    """
    ext = os.path.splitext(path)[1].lower()
    if trim is None and ext in SF_EXTS:
        try:
            import soundfile as sf
            sf.info(path)
            return path, False
        except Exception:
            pass
    ff = find_ffmpeg(root)
    if not ff:
        die(
            "参考音频 %s 的格式 soundfile 读不了（%s 这类需要先转成 wav），\n"
            "   而这个仓库里也没找到 ffmpeg。检查一下 %s\\runtime\\ffmpeg.exe 在不在，\n"
            "   或者自己转一份 wav 再用 --ref 指过去。"
            % (os.path.basename(path), (ext or "?").lstrip("."), root)
        )
    import subprocess
    import tempfile
    tag = safe_name(os.path.splitext(os.path.basename(path))[0]) or "ref"
    if trim:
        tag += "_%g_%g" % trim
    dst = os.path.join(tempfile.gettempdir(), "gsv_ref_%s.wav" % tag)
    cmd = [ff, "-y", "-loglevel", "error", "-i", path]
    if trim:
        cmd += ["-ss", "%g" % trim[0], "-to", "%g" % trim[1]]
    cmd += ["-vn", "-ac", "1", "-ar", "32000", "-c:a", "pcm_s16le", dst]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
        err = (r.stderr or b"").decode("utf-8", "replace").strip()
        die("用 ffmpeg 转换参考音频失败：\n   %s\n   %s" % (ff, err or ("退出码 %s" % r.returncode)))
    return dst, True


def pick_ref(ref_dir, name2text, sf):
    """挑一段 3~10 秒的 wav；有对应原文的优先"""
    if not os.path.isdir(ref_dir):
        return None
    cands = []
    for f in sorted(os.listdir(ref_dir)):
        if os.path.splitext(f)[1].lower() not in AUDIO_EXTS:
            continue
        p = os.path.join(ref_dir, f)
        dur = audio_duration(p)
        if dur is None:
            continue
        if 3.0 <= dur <= 10.0:
            cands.append((0 if f in name2text else 1, abs(dur - 7.0), p))
    return min(cands)[2] if cands else None


def main():
    args, _parser = parse_args()
    cfg_path = args.config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "gsv_tts.ini"
    )
    cfg_used = None
    if os.path.exists(cfg_path):
        cfg_used = apply_config(_parser, args, cfg_path)
    elif args.config:
        die("找不到配置文件: %s" % cfg_path)
    if args.clip:
        args.text = list(args.text) + [
            ln.strip() for ln in text_from_clipboard().splitlines() if ln.strip()
        ]

    cut_key = CUT_ALIAS.get(str(args.cut).strip().lower()) or CUT_ALIAS.get(str(args.cut).strip())
    if cut_key is None:
        die(
            "--cut 不支持 %r。可选：punct(默认) / 50字 / 4句 / 中文句号 / 英文句号 / none\n"
            "   一段文本不切句的话，长了容易重复、漏字、变调。" % (args.cut,)
        )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = args.root or os.environ.get("GSV_ROOT") or guess_root(script_dir)
    if not root:
        die(
            "不知道 GPT-SoVITS 仓库在哪。加 --root \"D:/GPT-SoVITS\"（指向仓库根目录，\n"
            "   就是里面有 webui.py 和 GPT_SoVITS 文件夹的那层），或者设环境变量 GSV_ROOT。"
        )
    root = os.path.abspath(root)
    if not all(os.path.exists(os.path.join(root, m)) for m in REPO_MARKERS):
        die("--root 指向的目录不像 GPT-SoVITS 仓库（里面得同时有 webui.py 和 GPT_SoVITS）: %s" % root)

    out = os.path.abspath(args.out) if args.out else os.path.join(root, "tts_out")
    ref = os.path.abspath(args.ref) if args.ref else None
    ref_dir = os.path.abspath(args.ref_dir) if args.ref_dir else None
    list_path = os.path.abspath(args.list_path) if args.list_path else None
    text_file = os.path.abspath(args.text_file) if args.text_file else None
    gpt = os.path.abspath(args.gpt) if args.gpt else None
    sovits = os.path.abspath(args.sovits) if args.sovits else None
    os.makedirs(out, exist_ok=True)

    gpt = gpt or newest(os.path.join(root, "GPT_weights_" + args.version), ".ckpt")
    sovits = sovits or newest(os.path.join(root, "SoVITS_weights_" + args.version), ".pth")
    if not gpt or not os.path.exists(gpt):
        die(
            "找不到 GPT 权重。把 .ckpt 放进 %s，或用 --gpt 指过去。"
            % os.path.join(root, "GPT_weights_" + args.version)
        )
    if not sovits or not os.path.exists(sovits):
        die(
            "找不到 SoVITS 权重。把 .pth 放进 %s，或用 --sovits 指过去。"
            % os.path.join(root, "SoVITS_weights_" + args.version)
        )

    texts = list(args.text)
    if text_file:
        if not os.path.exists(text_file):
            die("找不到文本文件: %s" % text_file)
        with open(text_file, encoding="utf-8-sig") as f:
            texts += [ln.strip() for ln in f if ln.strip()]
    if not texts:
        texts = ["今天天气不错，我们一起出去走走吧。"]

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["version"] = args.version
    if args.cpu:
        os.environ["is_half"] = "False"
    elif args.half:
        os.environ["is_half"] = "True"

    for sub in ["", "GPT_SoVITS/BigVGAN", "tools", "tools/asr", "GPT_SoVITS", "tools/uvr5"]:
        p = os.path.join(root, sub)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)
    os.chdir(root)

    print("=" * 70)
    print("仓库      :", root)
    print("版本      :", args.version)
    print("GPT 权重  :", gpt)
    print("SoVITS权重:", sovits)
    print("切句方式  :", cut_key)
    if cfg_used is not None:
        print("配置文件  :", cfg_path, "->", ", ".join(cfg_used) or "(没有可用项)")
    print("输出目录  :", out)
    print("=" * 70)

    try:
        import soundfile as sf
    except Exception:
        die(
            "没找到 soundfile —— 你多半没有 activate 仓库的 conda 环境。\n"
            "   先跑:  conda activate GPTSoVits"
        )

    name2text = load_name2text(list_path)

    if ref and not os.path.exists(ref):
        die("--ref 指的文件不存在: %s" % ref)
    if not ref and ref_dir:
        ref = pick_ref(ref_dir, name2text, sf)
        if ref:
            print("[参考音频] 从 %s 里自动挑了一段" % ref_dir)
    if not ref:
        die(
            "没有参考音频。用 --ref <一段 3~10 秒的干净人声>，\n"
            "   或者 --ref-dir <切片目录> 让它自己挑一段。"
        )

    ref_orig = ref
    ref_text = args.ref_text or name2text.get(os.path.basename(ref_orig))
    if not ref_text:
        if not args.list_path:
            why = "这次**没有**读到 --list，所以没法按文件名查原文。"
        elif not name2text:
            why = "--list 指向的文件没读到任何条目，检查一下路径和文件内容：\n   %s" % list_path
        else:
            why = (
                "--list 读到了 %d 条，但没有任何一行的文件名和它相同。\n"
                "   list 里前 3 条是：\n     %s"
                % (len(name2text), "\n     ".join(list(name2text)[:3]))
            )
        die(
            "拿不到 %s 的原文。用 --ref-text \"那段音频里原封不动说的话\" 手动给，\n"
            "   或者用 --list <output.list> 让它按文件名查。\n"
            "   注意：参考原文必须和音频逐字一致（标点、数字都要对上），否则会明显跑调。\n\n"
            "   具体原因：%s" % (os.path.basename(ref), why)
        )

    trim = norm_trim(getattr(args, "ref_trim", None))
    ref, ref_converted = to_wav(ref_orig, root, trim)
    if trim:
        print("[参考音频] 已按 ref-trim 只取第 %g~%g 秒（原文件没动）" % trim)
    if ref_converted:
        print("[参考音频] %s 这个格式 soundfile 读不了，已用 ffmpeg 转成 wav（原文件没动）"
              % os.path.basename(ref_orig))
    dur = audio_duration(ref)
    print("参考音频  : %s（%s）" % (ref, ("%.2f 秒" % dur) if dur is not None else "时长未知"))
    if dur is not None and not (3.0 <= dur <= 10.0):
        extra = ""
        if trim is None:
            extra = (
                "\n   剪法一（写进 gsv_tts.ini，以后一直生效）：ref-trim = 0 9\n"
                "   剪法二（只在这次命令行上用）：--ref-trim 0 9\n"
                "   数字是秒，意思是「取第 0 秒到第 9 秒」，换成你想要的那一段。\n"
                "   剪完记得把 ref-text 改成这 9 秒里实际说的话（必须逐字一致）。\n"
                "   更省事的办法（不用自己剪）：\n"
                "     --ref-dir .\\ref\\training_result\\ --list .\\ref\\training_result\\output.list\n"
                "   让脚本从训练切片里自动挑一段 3~10 秒的，原文也能自动查出来。"
            )
        die(
            "参考音频是 %.2f 秒，官方硬性要求 3~10 秒，否则合成时会直接报\n"
            "   「参考音频在3~10秒范围外，请更换！」。%s" % (dur, extra)
        )
    print("参考原文  :", ref_text)
    print("待合成    : %d 句" % len(texts))
    for k, tx in enumerate(texts[:3], 1):
        print("            %d) %s%s" % (k, tx[:36], "…" if len(tx) > 36 else ""))
    if len(texts) > 3:
        print("            …… 后面还有 %d 句" % (len(texts) - 3))

    try:
        import GPT_SoVITS.inference_webui as iw
        from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav
    except Exception as e:
        die(
            "导入仓库推理代码失败: %r\n"
            "   先在仓库目录跑一次 install.ps1 / install.sh，并确认已经 conda activate GPTSoVits" % (e,)
        )

    def lang_key(alias):
        key = str(alias).strip()
        code = LANG_ALIAS.get(key.lower(), LANG_ALIAS.get(key, key))
        for k, v in iw.dict_language.items():
            if v == code:
                return k
        die("语种 %s（=%s）这个版本不支持，支持的有: %s" % (alias, code, list(iw.dict_language.values())))

    prompt_lang = lang_key(args.ref_lang)
    text_lang = lang_key(args.lang)
    how_to_cut = iw.i18n(cut_key)

    t0 = time.time()
    change_gpt_weights(gpt)
    print("[1/3] GPT 权重加载完成 %.1fs" % (time.time() - t0))

    t0 = time.time()
    # 官方 change_sovits_weights 有两个坑：
    #   1) 它是生成器（yield），不迭代就等于完全没加载。
    #   2) 它末尾那个 yield 会引用 prompt_text_update / text_update /
    #      visible_sample_steps 等变量，而这些只在
    #      "prompt_language is not None and text_language is not None" 分支里赋值。
    #      WebUI 调用时总会带上这两个语种，所以只有命令行调用会踩到
    #      UnboundLocalError: local variable 'prompt_text_update' referenced before assignment。
    # 这里照 WebUI 的调法把两个语种一起传进去，两个坑一起躲开。
    for _ in change_sovits_weights(sovits, prompt_lang, text_lang):
        pass
    print("[2/3] SoVITS 权重加载完成 %.1fs" % (time.time() - t0))

    print("[3/3] 开始合成")
    done = []
    for i, text in enumerate(texts, 1):
        if args.seed is not None and args.seed >= 0:
            iw.set_seed(args.seed)
        t0 = time.time()
        try:
            results = list(
                get_tts_wav(
                    ref_wav_path=ref,
                    prompt_text=ref_text,
                    prompt_language=prompt_lang,
                    text=text,
                    text_language=text_lang,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    speed=args.speed,
                    how_to_cut=how_to_cut,
                )
            )
        except Exception as e:
            print("  %d/%d 失败: %r" % (i, len(texts), e))
            import traceback
            traceback.print_exc()
            continue
        sr, wav = results[-1]
        done.append((sr, wav))
        secs = len(wav) / float(sr)
        if args.merge:
            print("  %d/%d  %.1fs  %.2f 秒" % (i, len(texts), time.time() - t0, secs))
        else:
            stem = safe_name(text)
            fn = ("%03d_%s.wav" % (i, stem)) if stem else ("%03d.wav" % i)
            path = os.path.join(out, fn)
            sf.write(path, wav, sr)
            print("  %d/%d  %.1fs  %.2f 秒音频 -> %s"
                  % (i, len(texts), time.time() - t0, secs, path))

    print("")
    if args.merge:
        if not done:
            die("一句都没合成成功，没有可拼接的内容。")
        sr, wav = merge_segments(done)
        stem = safe_name(texts[0]) or "all"
        path = os.path.join(out, "merged_%s.wav" % stem)
        sf.write(path, wav, sr)
        print("拼接完成：%d 段 -> %s（共 %.2f 秒）"
              % (len(done), path, len(wav) / float(sr)))
    elif len(done) > 1:
        print("产物是 %d 个片段，都在 %s" % (len(done), out))
        print("想拼成一个文件：加 --merge，或者在 gsv_tts.ini 里写一行 merge = true")
    else:
        print("全部结束，产物在:", out)


if __name__ == "__main__":
    main()










