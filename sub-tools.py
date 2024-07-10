# -*- coding: utf-8 -*-
import argparse
import logging
import re
import subprocess as sp
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations, groupby
from pathlib import Path

import chardet

STYLE_DEFAULT = """Style: Default,GenYoMin TW H,23,&H00AAE2E6,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,85,100,0.1,0,1,1,3,2,30,30,15,1
Style: ENG,GenYoMin TW B,11,&H003CA8DC,&H000000FF,&H00000000,&H00000000,1,0,0,0,90,100,0,0,1,1,2,2,30,30,10,1
Style: JPN,GenYoMin JP B,15,&H003CA8DC,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,2,2,30,30,10,1"""
STYLE_2_EN = "{\\rENG}"
STYLE_2_JP = "{\\rJPN}"
STYLE_EN = "Style: Default,Verdana,18,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,90,100,0,0,1,0.3,3,2,30,30,20,1"
ARGS = ""
LIST_LANG = [
    "eng",
    "zho",
    "chi",
    "jpn",
]  # LIST_LANG  需要提取的字幕语言的ISO639代码列表
EFFECT = "{\\blur3}"
logger = logging.getLogger("sub_tools")
executor = ThreadPoolExecutor(max_workers=32)


class CustomFormatter(logging.Formatter):
    reset = "\x1b[0m"
    format = "%(levelname)s - %(message)s"
    debug_format = "%(thread)d - %(asctime)s - %(levelname)s - %(message)s \n(%(filename)s:%(lineno)d)"
    FORMATS = {
        logging.DEBUG: (grey := "\x1b[38;20m") + debug_format + reset,
        logging.INFO: (green := "\x1b[32m") + format + reset,
        logging.WARNING: (yellow := "\x1b[33;20m") + debug_format + reset,
        logging.ERROR: (red := "\x1b[31;20m") + debug_format + reset,
        logging.CRITICAL: (bold_red := "\x1b[31;1m") + format + reset,
    }

    def format(self, record):
        return logging.Formatter(self.FORMATS.get(record.levelno)).format(record)


class SRT:
    sub = namedtuple("sub", ["begin", "end", "content", "beginTime", "endTime"])

    def __init__(self, content) -> None:
        self.content = content

    @classmethod
    def fromFile(cls, file: Path, escape=" "):
        def time(rawtime):
            hour, minute, second, millisecond = map(int, re.split(r"[:,]", rawtime))
            return millisecond + 1000 * (second + (60 * (minute + 60 * hour)))

        def process(line: list[str]):
            begin, end = line[0].strip().split(" --> ")
            return SRT.sub(begin, end, [escape.join(line[1:])], time(begin), time(end))

        regex = re.compile(r"\r?\n\r?\n\d+\r?\n")
        return cls([process(x.splitlines()) for x in regex.split(read_file(file)[2:])])

    def merge_with(self, srt: Path, time_shift=1000):
        logger.debug(f"Starting merge with {srt}")

        def time_merge(content1: list[SRT.sub], content2: list[SRT.sub]):
            merged_content = []
            while content1 and content2:
                if (
                    abs(content1[0].beginTime - content2[0].beginTime) <= time_shift
                    or abs(content1[0].endTime - content2[0].endTime) <= time_shift
                ):
                    content1[0] = content1[0]._replace(
                        content=content1[0].content + content2.pop(0).content
                    )
                    continue
                if content1[0].beginTime < content2[0].beginTime:
                    merged_content.append(content1.pop(0))
                else:  # content1[0].beginTime > content2[0].beginTime
                    merged_content.append(content2.pop(0))
            merged_content.extend([*content1, *content2])
            return merged_content

        isCJK = lambda x: "\u4e00" <= x <= "\u9fa5"
        all_text = lambda y: "".join([x.content[0] for x in y])
        cjk_percentage = lambda z: sum(map(isCJK, (t := all_text(z)))) / (len(t) + 1)
        content1, content2 = self.content, srt.content
        if cjk_percentage(content1) < cjk_percentage(content2):
            content1, content2 = content2, content1
        logger.debug(f"Merge {len(content1)} lines with {len(content2)} lines")
        return SRT(time_merge(content1, content2))

    def save_as(self, file: Path):
        try:
            output = [
                "\n".join(
                    [
                        str(i),
                        f"{line.begin} --> {line.end}",
                        *map(str, line.content),
                        "",
                    ]
                )
                for i, line in enumerate(self.content, start=1)
            ]
            file.write_text("\n".join(output), encoding="utf-8")
            logger.info(f"Successfully wrote merged file: {file}")
        except Exception as e:
            logger.error(f"Failed to write merged file {file}: {str(e)}")
            logger.debug("Content causing the error:")
            for i, line in enumerate(
                self.content[:5]
            ):  # Log first 5 lines for debugging
                logger.debug(f"Line {i}: {line}")


class ASS:
    RE_ENG = re.compile(r"[\W\sA-Za-z0-9_\u00A0-\u03FF]+")

    def __init__(self, styles, events):
        self.styles = styles
        self.events = events

    @classmethod
    def from_ASS(cls, file: Path):
        # styles = []
        events = []
        for line in [x for x in read_file(file).splitlines()]:
            # styles += [cls.Style(line)] if line.startswith("Style:") else []
            events += [cls.Event(line)] if line.startswith("Dialogue:") else []
        return cls([], events)

    @classmethod
    def from_SRT(cls, file: Path):
        def rm_style(l):
            l = re.sub(r"<([ubi])>", r"{\\\1}", l)
            l = re.sub(r"</([ubi])>", r"{\\\1}", l)
            l = re.sub(r'<font\s+color="?(\w*?)"?>|</font>', "", l)
            return l

        re_time = re.compile(r"\d*(\d:\d{2}:\d{2}),(\d{2})\d")
        ftime = lambda x: re_time.sub(r"\1.\2", x)
        events = [
            cls.Event.fromSrt(ftime(x.begin), ftime(x.end), rm_style(x.content[0]))
            for x in SRT.fromFile(file, escape="\\N").content
        ]
        return cls([], events)

    def save(self, file: Path):
        output = """[Script Info]
ScriptType: v4.00+
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"""
        output += "\n".join(map(str, self.styles))
        output += """\n[Events]
Format: Layer, Start, End, Style, Actor, MarginL, MarginR, MarginV, Effect, Text\n"""
        output += "\n".join(map(str, self.events))
        file.write_text(output, encoding="utf-8")

    def update(self):
        self.styles = [self.Style(style) for style in self.get_style().splitlines()]
        second_style = self.get_2nd_style() if len(self.styles) > 1 else ""
        [event.update_style(second_style) for event in self.events]
        return self

    def is_eng_only(self, text):
        return self.RE_ENG.fullmatch(text) != None

    def _text(self):
        return "".join([event.text for event in self.events])

    def _text_2(self):
        re_1line = re.compile(r"^.*?(//N)?")
        return "".join([re_1line.sub("", event.text) for event in self.events])

    class Style:
        def __init__(self, line: str):
            (
                self.name,
                self.fontname,
                self.fontsize,
                self.primarycolour,
                self.secondarycolour,
                self.outlinecolour,
                self.backcolour,
                self.bold,
                self.italic,
                self.underline,
                self.strikeout,
                self.scalex,
                self.scaley,
                self.spacing,
                self.angle,
                self.borderstyle,
                self.outline,
                self.shadow,
                self.alignment,
                self.marginl,
                self.marginr,
                self.marginv,
                self.encoding,
            ) = line.split(":")[1].split(",")

        def __str__(self):
            return f"Style: {self.name},{self.fontname},{self.fontsize},{self.primarycolour},{self.secondarycolour},{self.outlinecolour},{self.backcolour},{self.bold},{self.italic},{self.underline},{self.strikeout},{self.scalex},{self.scaley},{self.spacing},{self.angle},{self.borderstyle},{self.outline},{self.shadow},{self.alignment},{self.marginl},{self.marginr},{self.marginv},{self.encoding}"

    class Event:
        def __init__(self, line: str):
            (
                self.layer,
                self.start,
                self.end,
                self.style,
                self.actor,
                self.marginl,
                self.marginr,
                self.marginv,
                self.effect,
                self.text,
            ) = (
                line.split(":", 1)[1].strip().split(",", 9)
            )

        @classmethod
        def fromSrt(cls, start_time, end_time, text):
            return cls(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}")

        @staticmethod
        def has_jap(x):
            return re.search(r"[\u3040-\u30f0]", x) != None

        @staticmethod
        def has_cjk(x):
            return re.search(r"[\u4e00-\u9fa5]", x) != None

        def update_style(self, second_style: str) -> None:
            self.text = re.sub(r"\{\\[(fn)rb](.*?)\}", "", self.text)
            self.text = self.text.replace("\\N", "\\N" + second_style + EFFECT)
            self.style = "Default"
            self.text = EFFECT + self.text
            if not self.has_cjk(self.text):
                self.text = second_style + self.text
            return

        def __str__(self):
            return f"Dialogue: {self.layer},{self.start},{self.end},{self.style},{self.actor},{self.marginl},{self.marginr},{self.marginv},{self.effect},{self.text}"

    def get_style(self):
        return STYLE_EN if self.is_eng_only(self._text()) else STYLE_DEFAULT

    def get_2nd_style(self):
        return (
            STYLE_2_JP
            if self.Event.has_jap(txt := self._text_2())
            else STYLE_2_EN if self.is_eng_only("", txt) else ""
        )


def is_exist(f: Path) -> bool:
    result = f.is_file() and not ARGS.force
    logger.debug(f"Checking if {f} exists: {result}")
    return result


def read_file(file: Path) -> str:
    return file.read_text(encoding=chardet.detect(file.read_bytes())["encoding"])


def merge_SRTs(files: list[Path]):
    logger.debug(f"Entering merge_SRTs with {len(files)} files")
    stem = lambda file: file.with_suffix("").with_suffix("").with_suffix("")
    len_suffixes = lambda x: len(x.suffixes)

    def merge(file1: Path, file2: Path):
        logger.debug(f"Attempting to merge {file1.name} and {file2.name}")
        if len(file2.suffixes) >= 3 and file1.suffixes[-2] == file2.suffixes[-2]:
            logger.debug(f"Skipping merge due to suffix condition")
            return
        if is_exist(new_file := file1.with_suffix("".join(file2.suffixes[-3:]))):
            logger.debug(f"Skipping merge due to existing file: {new_file}")
            return
        logger.info(f"merging:\n{file1.name}\n&\n{file2.name}\nas\n{new_file.name}")
        SRT.fromFile(file1).merge_with(SRT.fromFile(file2)).save_as(new_file)
        SRT_to_ASS(new_file)

    sorted_files = sorted([x for x in files if len(x.suffixes) < 5], key=len_suffixes)
    logger.debug(f"Sorted files: {sorted_files}")
    for x, group in groupby(sorted_files, key=stem):
        logger.debug(f"Processing group with stem: {x}")
        group_list = list(group)
        logger.debug(f"Group contents: {group_list}")
        for combo in combinations(group_list, 2):
            logger.debug(f"Submitting combination: {combo}")
            executor.submit(merge, *combo)

    logger.debug("Waiting for executor to complete all tasks")
    executor.shutdown(wait=True)
    logger.debug("Executor shutdown complete")


def SRT_to_ASS(file: Path) -> None:
    if not is_exist(new_file := file.with_suffix(".ass")):
        logger.info(f"Convert to ASS: {file.stem}\n")
        ASS.from_SRT(file).update().save(new_file)


def update_ASS_style(file: Path) -> None:
    logger.info(f"Updating style: {file.name}")
    ASS.from_ASS(file).update().save(file)


def extract_subs_MKV(files: list[Path]) -> None:
    SubInfo = namedtuple("SubInfo", ["index", "codec", "lang"])
    sp_run_quiet = lambda cmd: sp.run(cmd, stderr=sp.DEVNULL, stdout=sp.DEVNULL)

    def extract_fname(file: Path, sub: SubInfo, ext) -> Path:
        return file.with_suffix(f".track{sub.index}.{sub.lang}.{ext}")

    def extract(sub: SubInfo, ext) -> Path:
        if is_exist(out_sub := extract_fname(file, sub, ext)):
            return Path()
        sp_run_quiet(f'ffmpeg -y -i "{file}" -map 0:{sub.index} "{out_sub}" -an -vn')
        return out_sub

    for file in files:
        logger.info(f"extracting: {file.name}")
        probe_cmd = f'ffprobe "{file}" -select_streams s -show_entries stream=index:stream_tags=language:stream=codec_name -v quiet -print_format csv'
        probe = sp.check_output(probe_cmd).decode("utf-8").splitlines()
        subs = [SubInfo._make(sub.split(",")[1:]) for sub in probe]
        logger.debug(subs)
        for sub in subs:
            if "ass" in sub.codec:
                update_ASS_style(extract(sub, "ass"))
            elif "subrip" in sub.codec and sub.lang in LIST_LANG:
                SRT_to_ASS(extract(sub, "srt"))
        merge_SRTs(list(file.rglob(f"{file.stem}*.srt")))


def main():
    def load_args() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "file",
            help="files, default all files in current folder",
            nargs="*",
            default=".",
        )
        parser.add_argument(
            "-r", "--recurse", help="process all .srt/.ass", action="store_true"
        )
        parser.add_argument(
            "-f",
            "--force",
            help="force operation and overwrite existing files",
            action="store_true",
        )
        parser.add_argument(
            "-v", "--verbose", help="show debug information", action="store_true"
        )
        parser.add_argument(
            "-q", "--quite", help="show less information", action="store_true"
        )
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            "-u", "--update-ass", help="update .ass style", action="store_true"
        )
        group.add_argument("-m", "--merge-srt", help="merge srts", action="store_true")
        group.add_argument(
            "-e",
            "--extract-sub",
            help="extract subtitles from .mkv",
            action="store_true",
        )
        global ARGS
        ARGS = parser.parse_args()
        logger.debug(ARGS)

    def init_logger() -> None:
        logger.setLevel(logging.INFO)
        if ARGS.verbose:
            logger.setLevel(logging.DEBUG)
        if ARGS.quite:
            logger.setLevel(logging.ERROR)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(CustomFormatter())
        logger.addHandler(ch)

    def get_files() -> list[Path]:
        glob = lambda paths, pattern: sum([list(p.glob(pattern)) for p in paths], [])
        paths = [Path(x).resolve() for x in ARGS.file]
        if ARGS.recurse:
            paths += glob(paths, "**")
        if ARGS.update_ass:
            paths += glob(paths, "*.ass")
        elif ARGS.extract_sub:
            paths += glob(paths, "*.mkv")
        else:
            paths += glob(paths, "*.srt")
        paths = [x for x in paths if x.is_file()]
        logger.debug(paths)
        logger.info(f"found {len(paths)} files")
        return paths

    load_args()
    init_logger()
    files = get_files()
    if ARGS.update_ass:
        executor.map(update_ASS_style, files)
    elif ARGS.extract_sub:
        extract_subs_MKV(files)
    elif ARGS.merge_srt:
        logger.debug("Calling merge_SRTs")
        merge_SRTs(files)
        logger.debug("merge_SRTs completed")
        logger.debug("Waiting for all tasks to complete")
        executor.shutdown(wait=True)
        logger.debug("All tasks completed")
    else:
        executor.map(SRT_to_ASS, files)
    return


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
