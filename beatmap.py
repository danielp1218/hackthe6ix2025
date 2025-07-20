import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import re
import json


# === Data Classes ===

@dataclass
class General:
    AudioFilename: str = ""
    AudioLeadIn: int = 0
    AudioHash: Optional[str] = None
    PreviewTime: int = -1
    Countdown: int = 1
    SampleSet: str = "Normal"
    StackLeniency: float = 0.7
    Mode: int = 0
    LetterboxInBreaks: int = 0
    StoryFireInFront: int = 1
    UseSkinSprites: int = 0
    AlwaysShowPlayfield: int = 0
    OverlayPosition: str = "NoChange"
    SkinPreference: Optional[str] = None
    EpilepsyWarning: int = 0
    CountdownOffset: int = 0
    SpecialStyle: int = 0
    WidescreenStoryboard: int = 0
    SamplesMatchPlaybackRate: int = 0


@dataclass
class Editor:
    Bookmarks: List[int] = field(default_factory=list)
    DistanceSpacing: float = 1.0
    BeatDivisor: int = 4
    GridSize: int = 4
    TimelineZoom: float = 1.0


@dataclass
class Metadata:
    Title: str = ""
    TitleUnicode: str = ""
    Artist: str = ""
    ArtistUnicode: str = ""
    Creator: str = ""
    Version: str = ""
    Source: str = ""
    Tags: List[str] = field(default_factory=list)
    BeatmapID: int = 0
    BeatmapSetID: int = 0


@dataclass
class Difficulty:
    HPDrainRate: float = 5.0
    CircleSize: float = 5.0  # Number of columns in Mania
    OverallDifficulty: float = 5.0
    ApproachRate: float = 5.0
    SliderMultiplier: float = 1.4
    SliderTickRate: float = 1.0


@dataclass
class Event:
    type: str
    startTime: int
    params: List[str]


@dataclass
class TimingPoint:
    time: int
    beatLength: float
    meter: int
    sampleSet: int
    sampleIndex: int
    volume: int
    uninherited: int
    effects: int


@dataclass
class Colour:
    r: int
    g: int
    b: int


@dataclass
class Colours:
    colours: Dict[str, Colour] = field(default_factory=dict)


@dataclass
class HitSample:
    normalSet: int = 0
    additionSet: int = 0
    index: int = 0
    volume: int = 0
    filename: str = ""

    @staticmethod
    def from_string(sample_str: str):
        parts = sample_str.split(":")
        while len(parts) < 5:
            parts.append("")
        return HitSample(
            normalSet=int(parts[0]),
            additionSet=int(parts[1]),
            index=int(parts[2]),
            volume=int(parts[3]),
            filename=parts[4],
        )


@dataclass
class HitObject:
    x: int
    y: int
    time: int
    type: int
    hitSound: int
    column: int
    is_hold: bool = False
    end_time: Optional[int] = None
    hitSample: HitSample = field(default_factory=HitSample)


@dataclass
class Beatmap:
    format_version: int
    general: General
    editor: Editor
    metadata: Metadata
    difficulty: Difficulty
    events: List[Event]
    timing_points: List[TimingPoint]
    colours: Colours
    hit_objects: List[HitObject]


# === Parsing Functions ===

def parse_key_value_section(lines: List[str]) -> Dict[str, str]:
    return {
        key.strip(): value.strip()
        for line in lines if (':' in line or '=' in line)
        for key, value in [line.split(":", 1) if ":" in line else line.split("=", 1)]
    }


def parse_events(lines: List[str]) -> List[Event]:
    return [
        Event(parts[0], int(parts[1]), parts[2:])
        for line in lines if (parts := line.split(",")) and len(parts) >= 2
    ]


def parse_timing_points(lines: List[str]) -> List[TimingPoint]:
    return [
        TimingPoint(
            int(parts[0]), float(parts[1]), int(parts[2]), int(parts[3]),
            int(parts[4]), int(parts[5]), int(parts[6]), int(parts[7])
        ) for line in lines if (parts := line.split(",")) and len(parts) >= 8
    ]


def parse_colours(lines: List[str]) -> Colours:
    colour_dict = {}
    for line in lines:
        if ':' not in line:
            continue
        key, value = line.split(":", 1)
        rgb = list(map(int, value.strip().split(",")))
        if len(rgb) == 3:
            colour_dict[key.strip()] = Colour(*rgb)
    return Colours(colour_dict)


def parse_hit_object(line: str, column_count: int) -> HitObject:
    parts = line.strip().split(",")
    x, y, time, type_, hitSound = map(int, parts[:5])
    x = math.floor(x * 4 / 512)

    # Determine column index (clamped)
    column = max(0, min(int(x * column_count / 512), column_count - 1))

    is_hold = (type_ & 128) != 0
    end_time = None
    hit_sample = HitSample()

    if is_hold:
        # Format: ...,endTime:hitSample
        if len(parts) >= 6 and ":" in parts[5]:
            end_time_str, sample_str = parts[5].split(":", 1)
            try:
                end_time = int(end_time_str)
            except ValueError:
                end_time = None
            hit_sample = HitSample.from_string(sample_str)
        elif len(parts) >= 6:
            try:
                end_time = int(parts[5])
            except ValueError:
                pass
    else:
        if len(parts) >= 6 and ":" in parts[-1]:
            hit_sample = HitSample.from_string(parts[-1])

    return HitObject(
        x=x,
        y=y,
        time=time,
        type=type_,
        hitSound=hitSound,
        column=column,
        is_hold=is_hold,
        end_time=end_time,
        hitSample=hit_sample
    )


# === Section Cast Helpers ===

def _cast_general(key: str, val: str):
    cast_map = {
        "AudioLeadIn": int, "PreviewTime": int, "Countdown": int,
        "StackLeniency": float, "Mode": int, "LetterboxInBreaks": int,
        "StoryFireInFront": int, "UseSkinSprites": int, "AlwaysShowPlayfield": int,
        "CountdownOffset": int, "SpecialStyle": int, "WidescreenStoryboard": int,
        "SamplesMatchPlaybackRate": int
    }
    return cast_map.get(key, str)(val)


def _cast_editor(key: str, val: str):
    if key == "Bookmarks":
        return list(map(int, val.split(",")))
    return float(val) if "." in val else int(val)


def _cast_metadata(key: str, val: str):
    if key in {"BeatmapID", "BeatmapSetID"}:
        return int(val)
    elif key == "Tags":
        return val.split()
    return val


def _cast_difficulty(key: str, val: str):
    return float(val)


# === Main Parser ===

def parse_osu_file(file_path: str) -> Beatmap:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]

    match = re.match(r"osu file format v(\d+)", lines[0])
    if not match:
        raise ValueError(f"Invalid file format line: {lines[0]}")
    format_version = int(match.group(1))

    sections = {}
    current_section = None
    section_lines = []

    for line in lines[1:]:
        if line.startswith("[") and line.endswith("]"):
            if current_section:
                sections[current_section] = section_lines
            current_section = line[1:-1]
            section_lines = []
        elif line != "":
            section_lines.append(line)
    if current_section:
        sections[current_section] = section_lines

    general = General(
        **{k: _cast_general(k, v) for k, v in parse_key_value_section(sections.get("General", [])).items()})
    editor = Editor(**{k: _cast_editor(k, v) for k, v in parse_key_value_section(sections.get("Editor", [])).items()})
    metadata = Metadata(
        **{k: _cast_metadata(k, v) for k, v in parse_key_value_section(sections.get("Metadata", [])).items()})
    difficulty = Difficulty(
        **{k: _cast_difficulty(k, v) for k, v in parse_key_value_section(sections.get("Difficulty", [])).items()})
    events = parse_events(sections.get("Events", []))
    timing_points = parse_timing_points(sections.get("TimingPoints", []))
    colours = parse_colours(sections.get("Colours", []))

    column_count = int(difficulty.CircleSize)
    hit_objects = [parse_hit_object(line, column_count) for line in sections.get("HitObjects", [])]

    return Beatmap(
        format_version=format_version,
        general=general,
        editor=editor,
        metadata=metadata,
        difficulty=difficulty,
        events=events,
        timing_points=timing_points,
        colours=colours,
        hit_objects=hit_objects
    )


# === Optional: Save to JSON ===

def parse_osu_file_to_json(file_path: str, output_path: str):
    beatmap = parse_osu_file(file_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(beatmap, f, default=lambda o: o.__dict__, indent=4, ensure_ascii=False)


# === Example usage ===
parse_osu_file_to_json('exit.osu', 'death_piano.json')
