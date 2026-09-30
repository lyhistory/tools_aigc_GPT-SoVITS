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

REPO_MARKERS = ["webui.py", "GPT_SoVITS"]


def die(msg):
    print("")
    print("!! " + msg)
    print("")
    raise SystemExit(1)


def parse_args():
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
    return ap.parse_args()


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


def load_name2text(list_path):
    """output.list 每行是 路径|说话人|语种|文本；官方下游脚本都只取 basename"""
    name2text = {}
    if not list_path:
        return name2text
    if not os.path.exists(list_path):
        print("!! 找不到 list 文件: %s（不影响，只是 --ref-text 得手打）" % list_path)
        return name2text
    with open(list_path, encoding="utf8") as f:
        for line in f.read().strip("\n").split("\n"):
            part = line.split("|", 3)
            if len(part) == 4:
                name2text[os.path.basename(part[0])] = part[3].strip()
    print("[list] 读到 %d 条「文件名 -> 原文」：%s" % (len(name2text), list_path))
    return name2text


def pick_ref(ref_dir, name2text, sf):
    """挑一段 3~10 秒的 wav；有对应原文的优先"""
    if not os.path.isdir(ref_dir):
        return None
    cands = []
    for f in sorted(os.listdir(ref_dir)):
        if not f.lower().endswith(".wav"):
            continue
        p = os.path.join(ref_dir, f)
        try:
            dur = sf.info(p).duration
        except Exception:
            continue
        if 3.0 <= dur <= 10.0:
            cands.append((0 if f in name2text else 1, abs(dur - 7.0), p))
    return min(cands)[2] if cands else None


def main():
    args = parse_args()

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
        with open(text_file, encoding="utf8") as f:
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

    ref_text = args.ref_text or name2text.get(os.path.basename(ref))
    if not ref_text:
        die(
            "拿不到 %s 的原文。用 --ref-text \"那段音频里原封不动说的话\" 手动给，\n"
            "   或者用 --list <output.list> 让它按文件名查。\n"
            "   注意：参考原文必须和音频逐字一致（标点、数字都要对上），否则会明显跑调。"
            % os.path.basename(ref)
        )

    print("参考音频  : %s（%.2f 秒）" % (ref, sf.info(ref).duration))
    print("参考原文  :", ref_text)
    print("待合成    : %d 句" % len(texts))

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

    t0 = time.time()
    change_gpt_weights(gpt)
    print("[1/3] GPT 权重加载完成 %.1fs" % (time.time() - t0))

    t0 = time.time()
    for _ in change_sovits_weights(sovits):
        pass
    print("[2/3] SoVITS 权重加载完成 %.1fs" % (time.time() - t0))

    print("[3/3] 开始合成")
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
                )
            )
        except Exception as e:
            print("  %d/%d 失败: %r" % (i, len(texts), e))
            import traceback
            traceback.print_exc()
            continue
        sr, wav = results[-1]
        path = os.path.join(out, "%03d_%d.wav" % (i, int(time.time())))
        sf.write(path, wav, sr)
        print("  %d/%d  %.1fs  %.2f 秒音频 -> %s"
              % (i, len(texts), time.time() - t0, len(wav) / float(sr), path))

    print("")
    print("全部结束，产物在:", out)


if __name__ == "__main__":
    main()
