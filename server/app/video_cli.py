import argparse
import sys
from pathlib import Path

from app.errors import PipelineError
from app.video import video_to_photos


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从视频中提取帧并清洗为重建可用照片")
    parser.add_argument("--video", required=True, type=Path, help="输入视频路径")
    parser.add_argument("--out", required=True, type=Path, help="输出照片目录")
    parser.add_argument("--max-frames", type=int, default=120, help="最大抽取帧数")
    parser.add_argument("--blur-threshold", type=float, default=100.0, help="模糊阈值")
    parser.add_argument("--min-kept", type=int, default=30, help="至少保留帧数")
    args = parser.parse_args(argv)

    if not args.video.is_file():
        print(f"视频文件不存在: {args.video}", file=sys.stderr)
        return 2

    try:
        result = video_to_photos(
            args.video,
            args.out,
            max_frames=args.max_frames,
            blur_threshold=args.blur_threshold,
            min_kept=args.min_kept,
        )
    except PipelineError as e:
        print(e.message_zh, file=sys.stderr)
        return 1

    print(
        f"保留 {len(result.kept)} 张"
        f"（丢弃模糊 {len(result.dropped_blurry)}、重复 {len(result.dropped_dupes)}）"
        f"，输出目录: {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
