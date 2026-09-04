#!/usr/bin/env python3
"""
Karaoke Generator (batch, AMD/NVIDIA/CPU) — forced-alignment edition
--------------------------------------------------------------------
Вход:  аудиофайл + текст (одна строка = одна строка субтитра)
Выход: mp4 с минусовкой и караоке-подсветкой слов (fill-эффект)

Главное отличие этой версии — ТОЧНЫЕ тайминги слов через
wav2vec2 forced alignment (torchaudio MMS_FA). Мы больше не гадаем,
где Whisper поставил границы: мы берём ТВОЙ текст и находим, где
именно в аудио звучит каждое слово. Работает даже на медленном
оперном вокале, где Whisper-тайминги «плывут».

Forced alignment требует латиницы. Для языков на латинице (en, de,
es, it, fr, pt, nl, ...) работает напрямую. Для кириллицы и прочего
используется старый Whisper-метод как фолбэк (--timing whisper
форсирует его).

Использование:
    python karaoke_gen.py single song.mp3 lyrics.txt -o out.mp4 -l en
    python karaoke_gen.py batch ./songs --out ./out -l en
    python karaoke_gen.py batch ./metal --out ./out --demucs-model htdemucs_ft
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


# латинские языки, где forced alignment работает без романизации
LATIN_LANGS = {"en", "de", "es", "it", "fr", "pt", "nl", "pl", "sv",
               "da", "no", "fi", "id", "ca", "ro", "cs", "hr", "sk"}


# ---------- Устройство ----------
def get_device():
    import torch
    if torch.cuda.is_available():
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("    GPU не найден, использую CPU (будет медленно)")
    return "cpu"


# ---------- Demucs ----------
def separate_vocals(audio_path: Path, work_dir: Path, device: str, model: str):
    cmd = ["demucs", "-n", model, "--two-stems=vocals", "-o", str(work_dir),
           "-d", "cuda" if device == "cuda" else "cpu", str(audio_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err = result.stderr.strip() or result.stdout.strip() or "(нет вывода)"
        raise RuntimeError(f"Demucs упал (код {result.returncode}):\n{err}")
    stem_dir = work_dir / model / audio_path.stem
    instrumental = stem_dir / "no_vocals.wav"
    vocals = stem_dir / "vocals.wav"
    if not instrumental.exists() or not vocals.exists():
        raise RuntimeError(f"Demucs завершился, но файлы не найдены в {stem_dir}")
    return instrumental, vocals


# ========== СПОСОБ 1: forced alignment (wav2vec2 / MMS_FA) ==========
_FA_BUNDLE = None
_FA_MODEL = None
_FA_TOKENIZER = None
_FA_ALIGNER = None


def load_forced_aligner(device: str):
    """Загружает MMS_FA один раз (кэшируется в глобалах)."""
    global _FA_BUNDLE, _FA_MODEL, _FA_TOKENIZER, _FA_ALIGNER
    if _FA_MODEL is not None:
        return
    import torchaudio
    print("    Загружаю forced-aligner (wav2vec2 MMS_FA)...")
    _FA_BUNDLE = torchaudio.pipelines.MMS_FA
    _FA_MODEL = _FA_BUNDLE.get_model().to(device)
    _FA_TOKENIZER = _FA_BUNDLE.get_tokenizer()
    _FA_ALIGNER = _FA_BUNDLE.get_aligner()


def _normalize_for_fa(word: str) -> str:
    """
    Нормализация под MMS_FA: строчные латинские буквы + апостроф.
    Возвращает '' если после чистки ничего не осталось.
    """
    w = word.lower().replace("’", "'")
    w = re.sub(r"[^a-z']", "", w)
    return w


def align_forced(vocals_path: Path, lyrics_lines: list, device: str):
    """
    Forced alignment всего текста на аудио. Возвращает список строк с
    точными таймингами каждого слова прямо из wav2vec2.

    lyrics_lines — список исходных строк текста.
    """
    import torch
    import torchaudio
    import soundfile as sf

    load_forced_aligner(device)

    # читаем аудио через soundfile (torchaudio.load в 2.9 требует torchcodec,
    # который тянет CUDA-библиотеки — на ROCm недоступны). soundfile уже стоит.
    data, sr = sf.read(str(vocals_path), dtype="float32", always_2d=True)
    # data: (samples, channels) → (channels, samples)
    waveform = torch.from_numpy(data.T)
    # в моно
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # ресемпл под частоту модели
    target_sr = _FA_BUNDLE.sample_rate
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    # строим плоский список слов, помним номер строки и нормализованную форму
    flat_words = []       # исходные слова (для показа)
    flat_norm = []        # нормализованные (для токенайзера)
    word_to_line = []     # индекс строки
    for li, line in enumerate(lyrics_lines):
        for w in line.split():
            norm = _normalize_for_fa(w)
            if not norm:
                # слово вроде "—" или чисто пунктуация — пропускаем в выравнивании,
                # но сохраняем для отображения, привяжем позже интерполяцией
                continue
            flat_words.append(w)
            flat_norm.append(norm)
            word_to_line.append(li)

    if not flat_norm:
        raise RuntimeError("После нормализации не осталось слов для выравнивания")

    # прогон через модель (эмиссия — на GPU, быстро)
    with torch.inference_mode():
        emission, _ = _FA_MODEL(waveform.to(device))
        # forced_align НЕ реализован на ROCm/CUDA-бэкенде — только CPU.
        # Переносим эмиссию на CPU и выравниваем там (это лёгкий алгоритм,
        # не нейросеть — на CPU быстро). Aligner и токены тоже на CPU.
        emission_cpu = emission[0].cpu()
        token_spans = _FA_ALIGNER(emission_cpu, _FA_TOKENIZER(flat_norm))

    # длительность одного фрейма эмиссии в секундах
    ratio = waveform.shape[1] / emission.shape[1] / target_sr

    # для каждого слова — время из спанов
    word_times = []
    for spans in token_spans:
        t0 = spans[0].start * ratio
        t1 = spans[-1].end * ratio
        word_times.append((t0, t1))

    # раскладываем обратно по строкам
    lines_out = []
    for li, line_text in enumerate(lyrics_lines):
        idxs = [k for k, wl in enumerate(word_to_line) if wl == li]
        if not idxs:
            # строка целиком выпала из выравнивания (только пунктуация и т.п.)
            continue
        first_ws = word_times[idxs[0]][0]
        last_we = word_times[idxs[-1]][1]
        line_start = max(0.0, first_ws - 0.15)
        line_end = last_we + 0.3

        words_info = []
        prev_flip = line_start
        for k in idxs:
            ws, we = word_times[k]
            k_cs = max(1, int(round((ws - prev_flip) * 100)))
            words_info.append({"text": flat_words[k], "ws": ws, "we": we,
                               "k": k_cs, "mapped": True})
            prev_flip = ws

        # SANITY-CHECK: forced alignment иногда "растягивает" слово на
        # инструментальные проигрыши / бэквокал. Если одно слово занимает
        # больше 40% строки И больше 2 секунд — переходим на равномерное
        # распределение.
        line_dur_cs = int((line_end - line_start) * 100)
        need_fallback = False
        if len(words_info) > 1 and line_dur_cs > 100:
            max_k = max(w["k"] for w in words_info)
            if max_k > line_dur_cs * 0.4 and max_k > 200:
                need_fallback = True
        if need_fallback:
            per_word = max(15, line_dur_cs // (len(words_info) + 1))
            words_info[0]["k"] = 15
            for w in words_info[1:]:
                w["k"] = per_word

        lines_out.append({"text": line_text, "start": line_start,
                          "end": line_end, "words": words_info,
                          "fallback": need_fallback})

    # снимаем overlap: жёсткий clamp, конец предыдущей до начала следующей
    for i in range(len(lines_out) - 1):
        ns = lines_out[i + 1]["start"]
        if lines_out[i]["end"] > ns - 0.02:
            lines_out[i]["end"] = ns - 0.02

    return lines_out


# ========== СПОСОБ 2: Whisper + fuzzy (фолбэк для кириллицы) ==========
def load_whisper(model_size: str, device: str):
    print(f"    Загружаю Whisper '{model_size}' на {device}...")
    import whisper
    return whisper.load_model(model_size, device=device)


def whisper_words(model, vocals_path: Path, language: str, device: str):
    result = model.transcribe(str(vocals_path), language=language,
                              word_timestamps=True, fp16=(device == "cuda"),
                              verbose=False)
    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            t = w.get("word", "").strip()
            if t and "start" in w and "end" in w:
                words.append({"text": t, "start": w["start"], "end": w["end"]})
    return words


def _norm(w):
    return re.sub(r"[^\w]", "", w.lower())


def _lev_ratio(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    m, n = len(a), len(b)
    if m < n:
        a, b, m, n = b, a, n, m
    prev = list(range(n + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * n
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (0 if ca == cb else 1))
        prev = cur
    return 1.0 - prev[n] / max(m, n)


def _align_dp(text_words, wh_words, thr=0.65):
    n, m = len(text_words), len(wh_words)
    tn = [_norm(w) for w in text_words]
    wn = [_norm(w["text"]) for w in wh_words]
    NEG = -1e9
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    dirn = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] - 0.05
        dirn[i][0] = "T"
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] - 0.05
        dirn[0][j] = "W"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = _lev_ratio(tn[i - 1], wn[j - 1])
            match = score[i - 1][j - 1] + s if s >= thr else NEG
            st = score[i - 1][j] - 0.05
            sw = score[i][j - 1] - 0.05
            best = max(match, st, sw)
            score[i][j] = best
            dirn[i][j] = "M" if best == match else ("T" if best == st else "W")
    mapping = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        d = dirn[i][j]
        if d == "M":
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif d == "T":
            i -= 1
        else:
            j -= 1
    return mapping


def align_whisper_fuzzy(model, vocals_path, lyrics_lines, language, device):
    wh = whisper_words(model, vocals_path, language, device)
    if not wh:
        return []
    text_words, word_to_line = [], []
    for li, line in enumerate(lyrics_lines):
        for w in line.split():
            text_words.append(w)
            word_to_line.append(li)
    mapping = _align_dp(text_words, wh)

    wt = [None] * len(text_words)
    for i, wi in enumerate(mapping):
        if wi is not None:
            wt[i] = (wh[wi]["start"], wh[wi]["end"])
    # интерполяция по START соседей
    for i in range(len(wt)):
        if wt[i] is not None:
            continue
        left = right = None
        for j in range(i - 1, -1, -1):
            if wt[j] is not None:
                left = (j, wt[j]); break
        for j in range(i + 1, len(wt)):
            if wt[j] is not None:
                right = (j, wt[j]); break
        if left and right:
            lj, (ls, _) = left; rj, (rs, _) = right
            gap = rs - ls
            if gap <= 0:
                s = ls + 0.3 * (i - lj); wt[i] = (s, s + 0.3)
            elif gap > 5.0:
                s = min(ls + 0.4 * (i - lj), rs - 0.3); wt[i] = (s, s + 0.3)
            else:
                t = (i - lj) / (rj - lj); s = ls + gap * t; wt[i] = (s, s + 0.3)
        elif left:
            s = left[1][0] + 0.4 * (i - left[0]); wt[i] = (s, s + 0.3)
        elif right:
            s = max(0.0, right[1][0] - 0.4 * (right[0] - i)); wt[i] = (s, s + 0.3)
        else:
            wt[i] = (0.0, 0.3)

    lines_out = []
    for li, line_text in enumerate(lyrics_lines):
        idxs = [i for i, wl in enumerate(word_to_line) if wl == li]
        if not idxs:
            continue
        line_start = max(0.0, wt[idxs[0]][0] - 0.15)
        line_end = wt[idxs[-1]][1] + 0.3
        words_info = []
        prev = line_start
        for i in idxs:
            ws, we = wt[i]
            words_info.append({"text": text_words[i], "ws": ws, "we": we,
                               "k": max(1, int(round((ws - prev) * 100))),
                               "mapped": mapping[i] is not None})
            prev = ws
        # fallback для развалившихся строк
        total_k = sum(w["k"] for w in words_info)
        dur_cs = int((line_end - line_start) * 100)
        need = False
        if dur_cs > 100 and total_k < dur_cs * 0.3 and len(words_info) > 1:
            need = True
        if not need and len(words_info) > 1 and dur_cs > 100:
            mk = max(w["k"] for w in words_info)
            if mk > dur_cs * 0.5 and mk > 200:
                need = True
        if need:
            per = max(15, dur_cs // (len(words_info) + 1))
            words_info[0]["k"] = 15
            for w in words_info[1:]:
                w["k"] = per
        lines_out.append({"text": line_text, "start": line_start,
                          "end": line_end, "words": words_info, "fallback": need})

    for i in range(len(lines_out) - 1):
        ns = lines_out[i + 1]["start"]
        if lines_out[i]["end"] > ns - 0.02:
            lines_out[i]["end"] = ns - 0.02
    return lines_out


# ---------- ASS ----------
# Три стиля:
# Active   — активная строка сверху, y=500, с fill-эффектом (белый→жёлтый)
# Preview  — следующая строка снизу, y=640, ПОЛУПРОЗРАЧНАЯ (не жёлтая, без fill)
# Notes    — ♪ ♫ ♪ на месте активной (y=500) во время длинных пауз
ASS_HEADER = """[Script Info]
Title: Karaoke
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Active,Arial,80,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,5,2,5,0,0,0,1
Style: Preview,Arial,60,&H80FFFFFF,&H80FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,3,2,5,0,0,0,1
Style: Notes,Arial,100,&H80FFFFFF,&H80FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,3,2,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def fmt_time(t):
    t = round(t, 2)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t - h * 3600 - m * 60
    return f"{h}:{m:02d}:{s:05.2f}"


def write_ass(lines, path):
    """
    Упрощённая логика показа субтитров:

    - Preview[j] следующей строки висит СНИЗУ от появления предыдущей Active
      (или от начала аудио для первой) до появления своей Active. То есть
      всегда виден "что будет петься следующим", без ограничений по времени.

    - Active показывается СВЕРХУ с \\kf (плавная заливка).

    - Notes (♪ ♫ ♪) показываются только ПОСЛЕ последней Active строки —
      финальный проигрыш где следующей уже нет.

    В любой момент времени на экране: либо Active+Preview, либо только
    Preview (в паузе перед следующей), либо только Notes (в финале).
    """
    NOTES_TEXT = "♪ ♫ ♪"
    POS_ACTIVE = "{\\pos(960,500)}"
    POS_PREVIEW = "{\\pos(960,640)}"
    POS_NOTES = "{\\pos(960,500)}"

    def active_event(l):
        ktext = " ".join(f"{{\\kf{w['k']}}}{w['text']}" for w in l["words"])
        return (f"Dialogue: 0,{fmt_time(l['start'])},{fmt_time(l['end'])},"
                f"Active,,0,0,0,,{POS_ACTIVE}{ktext}")

    def preview_event(start, end, text):
        return (f"Dialogue: 0,{fmt_time(start)},{fmt_time(end)},"
                f"Preview,,0,0,0,,{POS_PREVIEW}{text}")

    def notes_event(start, end):
        return (f"Dialogue: 0,{fmt_time(start)},{fmt_time(end)},"
                f"Notes,,0,0,0,,{POS_NOTES}{NOTES_TEXT}")

    events = []
    if not lines:
        path.write_text(ASS_HEADER, encoding="utf-8")
        return

    # Active-события для каждой строки — обычная жёлтая заливка сверху
    for l in lines:
        events.append(active_event(l))

    # Preview-события для каждой строки — висит снизу от появления
    # ПРЕДЫДУЩЕЙ Active (или от 0 для первой) до появления СВОЕЙ Active
    for j, l in enumerate(lines):
        preview_start = lines[j - 1]["start"] if j > 0 else 0.0
        preview_end = l["start"]
        if preview_end > preview_start:
            events.append(preview_event(preview_start, preview_end, l["text"]))

    # Notes после последней Active — финальный проигрыш.
    # Ставим на 10 минут вперёд, ffmpeg -shortest обрежет по концу аудио.
    last = lines[-1]
    events.append(notes_event(last["end"], last["end"] + 600))

    path.write_text(ASS_HEADER + "\n".join(events), encoding="utf-8")


def render_video(instrumental, subs, output, w=1920, h=1080):
    subs_arg = str(subs).replace("\\", "/").replace(":", r"\:")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r=30",
                    "-i", str(instrumental),
                    "-vf", f"subtitles='{subs_arg}'",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", str(output)],
                   check=True)


# ---------- Обработка одной пары ----------
def process_track(audio, lyrics, output, work_dir, device, language,
                  demucs_model, timing_mode, whisper_model,
                  use_ass=None, instrumental_path=None):
    """
    audio            — аудиофайл (обязательно если instrumental_path не указан)
    lyrics           — текст песни (обязательно если use_ass не указан)
    use_ass          — путь к готовому .ass, пропустить Whisper/forced alignment
    instrumental_path — путь к готовому инструменталу WAV, пропустить Demucs
    """
    t0 = time.time()

    # === Шаг 1: инструментал (Demucs или готовый) ===
    if instrumental_path:
        print(f"    [1] Инструментал: {instrumental_path.name} (готовый, Demucs пропущен)")
        instrumental = instrumental_path
        vocals = None
    else:
        print(f"    [1] Demucs ({demucs_model})...", end=" ", flush=True)
        instrumental, vocals = separate_vocals(audio, work_dir, device, demucs_model)
        print(f"{time.time()-t0:.1f}s")

    # === Шаг 2: субтитры (alignment или готовый .ass) ===
    lines = None  # если используем готовый .ass — не нужны для отчёта
    if use_ass:
        print(f"    [2] Субтитры: {use_ass.name} (готовые, alignment пропущен)")
        subs_path = use_ass
        method = "прегенерированный .ass"
    else:
        lyrics_lines = [ln.strip() for ln in
                        lyrics.read_text(encoding="utf-8").splitlines() if ln.strip()]

        use_fa = (timing_mode == "forced" or
                  (timing_mode == "auto" and language in LATIN_LANGS))

        t1 = time.time()
        if use_fa:
            print(f"    [2] Forced alignment...", end=" ", flush=True)
            try:
                lines = align_forced(vocals, lyrics_lines, device)
                method = "forced-align"
            except Exception as e:
                print(f"\n    ⚠ forced alignment не удался ({e}), падаю на Whisper")
                if whisper_model[0] is None:
                    whisper_model[0] = load_whisper(whisper_model[1], device)
                lines = align_whisper_fuzzy(whisper_model[0], vocals, lyrics_lines,
                                            language, device)
                method = "whisper-fuzzy (fallback)"
        else:
            print(f"    [2] Whisper alignment...", end=" ", flush=True)
            if whisper_model[0] is None:
                whisper_model[0] = load_whisper(whisper_model[1], device)
            lines = align_whisper_fuzzy(whisper_model[0], vocals, lyrics_lines,
                                        language, device)
            method = "whisper-fuzzy"
        print(f"{time.time()-t1:.1f}s [{method}]")

        subs_path = work_dir / f"{audio.stem}.ass"
        write_ass(lines, subs_path)

    # === Шаг 3: рендер видео (всегда) ===
    output.parent.mkdir(parents=True, exist_ok=True)
    render_video(instrumental, subs_path, output)

    # Отчёт
    if lines is not None:
        fb = sum(1 for l in lines if l.get("fallback"))
        extra = f", {fb} строк fallback" if fb else ""
        print(f"    ✓ {output.name} ({len(lines)} строк{extra}, всего {time.time()-t0:.1f}s)")
    else:
        print(f"    ✓ {output.name} (всего {time.time()-t0:.1f}s)")


# ---------- Сбор пар ----------
def find_pairs(audio_dir, lyrics_dir):
    exts = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
    pairs, missing = [], []
    for a in sorted(Path(audio_dir).iterdir()):
        if a.suffix.lower() not in exts:
            continue
        l = Path(lyrics_dir) / f"{a.stem}.txt"
        if l.exists():
            pairs.append((a, l))
        else:
            missing.append(a.name)
    return pairs, missing


# ---------- Main ----------
def cmd_single(args):
    if not args.use_ass and not args.lyrics:
        sys.exit("Нужен либо файл текста (позиционный аргумент lyrics), "
                 "либо --use-ass с готовым .ass")
    device = get_device()
    args.work_dir.mkdir(exist_ok=True)
    whisper_model = [None, args.model]
    process_track(args.audio, args.lyrics, args.output, args.work_dir,
                  device, args.language, args.demucs_model, args.timing,
                  whisper_model,
                  use_ass=args.use_ass,
                  instrumental_path=args.instrumental)


def cmd_batch(args):
    audio_dir = args.audio_dir or args.folder
    lyrics_dir = args.lyrics_dir or args.folder
    if not audio_dir or not lyrics_dir:
        sys.exit("Укажи folder или --audio-dir + --lyrics-dir")
    pairs, missing = find_pairs(audio_dir, lyrics_dir)
    if missing:
        print(f"⚠  Без текста ({len(missing)}): {', '.join(missing[:5])}"
              + ("..." if len(missing) > 5 else ""))
    if not pairs:
        sys.exit("Пар не найдено")
    print(f"Найдено пар: {len(pairs)}")
    todo = []
    for a, l in pairs:
        out = args.out / f"{a.stem}.mp4"
        if out.exists() and not args.force:
            print(f"  ⏭  {a.stem}.mp4 готов, пропускаю")
        else:
            todo.append((a, l, out))
    if not todo:
        print("Всё готово ✓")
        return
    print(f"К обработке: {len(todo)}\n")

    device = get_device()
    args.work_dir.mkdir(exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    whisper_model = [None, args.model]

    t_start = time.time()
    failed = []
    for i, (audio, lyrics, output) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {audio.name}")
        try:
            process_track(audio, lyrics, output, args.work_dir, device,
                          args.language, args.demucs_model, args.timing,
                          whisper_model,
                          use_ass=None, instrumental_path=None)
        except Exception as e:
            print(f"    ✗ ОШИБКА: {e}")
            failed.append(audio.name)
        print()
    print(f"═══ Готово: {len(todo)-len(failed)}/{len(todo)} "
          f"за {(time.time()-t_start)/60:.1f} мин ═══")
    if failed:
        print(f"Ошибки: {', '.join(failed)}")


def main():
    p = argparse.ArgumentParser(description="Караоке с forced alignment")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("-l", "--language", default="ru")
        sp.add_argument("-m", "--model", default="large-v3",
                        choices=["tiny", "base", "small", "medium",
                                 "large-v2", "large-v3"],
                        help="Модель Whisper (для fallback-режима)")
        sp.add_argument("--demucs-model", default="htdemucs",
                        choices=["htdemucs", "htdemucs_ft", "htdemucs_6s",
                                 "mdx_extra", "mdx_extra_q"])
        sp.add_argument("--timing", default="auto",
                        choices=["auto", "forced", "whisper"],
                        help="auto=forced для латиницы иначе whisper; "
                             "forced=всегда forced align; whisper=всегда Whisper")
        sp.add_argument("--work-dir", type=Path, default=Path("karaoke_work"))

    sp1 = sub.add_parser("single")
    sp1.add_argument("audio", type=Path)
    sp1.add_argument("lyrics", type=Path, nargs="?",
                     help="Файл текста (не нужен если указан --use-ass)")
    sp1.add_argument("-o", "--output", type=Path, default=Path("karaoke.mp4"))
    sp1.add_argument("--use-ass", type=Path, default=None,
                     help="Готовый .ass файл — пропустить Whisper/forced alignment")
    sp1.add_argument("--instrumental", type=Path, default=None,
                     help="Готовый инструментал WAV — пропустить Demucs")
    add_common(sp1)
    sp1.set_defaults(func=cmd_single)

    sp2 = sub.add_parser("batch")
    sp2.add_argument("folder", nargs="?", type=Path)
    sp2.add_argument("--audio-dir", type=Path)
    sp2.add_argument("--lyrics-dir", type=Path)
    sp2.add_argument("--out", type=Path, default=Path("karaoke_out"))
    sp2.add_argument("-f", "--force", action="store_true")
    add_common(sp2)
    sp2.set_defaults(func=cmd_batch)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"Ошибка внешней команды: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nПрервано", file=sys.stderr)
        sys.exit(130)
