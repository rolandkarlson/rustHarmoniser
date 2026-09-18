#!/usr/bin/env python3
"""Idempotently install rustHarmoniser's AbletonMCP clip writer extension."""
from __future__ import print_function

import argparse
import ast
import datetime
import os
import shutil
import sys


IMPORT_MARKER = "# rustHarmoniser clip writer import"
CAPABILITY_MARKER = "# rustHarmoniser clip writer capability"
DISPATCH_MARKER = "# rustHarmoniser clip writer dispatch"
TIMEOUT_MARKER = "# rustHarmoniser clip writer timeout"


def patched_source(source):
    installed_fragments = (
        IMPORT_MARKER + "\nfrom .harmonizer_clips import write_harmonizer_clips\n",
        CAPABILITY_MARKER + '\n    "write_harmonizer_clips",',
        CAPABILITY_MARKER + '\n                                 "write_harmonizer_clips",',
        DISPATCH_MARKER + '\n                            result = write_harmonizer_clips(self._song, params)',
        TIMEOUT_MARKER + '\n                long_running_commands["write_harmonizer_clips"] = 30.0',
    )
    if any(marker in source for marker in (IMPORT_MARKER, CAPABILITY_MARKER, DISPATCH_MARKER, TIMEOUT_MARKER)):
        if all(fragment in source for fragment in installed_fragments):
            ast.parse(source, filename="AbletonMCP/__init__.py")
            return source
        raise RuntimeError("refusing incomplete rustHarmoniser installer markers")
    required = [
        "from _Framework.ControlSurface import ControlSurface\n",
        '    "clear_notes_from_clip",\n]',
        '                                 "clear_notes_from_clip",\n',
        '                        elif command_type == "clear_notes_from_clip":\n'
        '                            track_index = params.get("track_index", 0)\n'
        '                            clip_index = params.get("clip_index", 0)\n'
        '                            result = self._clear_notes_from_clip(track_index, clip_index)\n',
        '                long_running_commands = {"create_audio_clip": 60.0}\n',
    ]
    missing = [marker for marker in required if source.count(marker) != 1]
    if missing:
        raise RuntimeError("refusing unknown AbletonMCP structure; expected unique patch markers")
    source = source.replace(
        required[0],
        required[0] + IMPORT_MARKER + "\nfrom .harmonizer_clips import write_harmonizer_clips\n",
    )
    source = source.replace(
        required[1],
        '    "clear_notes_from_clip",\n    ' + CAPABILITY_MARKER + '\n    "write_harmonizer_clips",\n]',
    )
    source = source.replace(
        required[2],
        '                                 "clear_notes_from_clip",\n                                 ' + CAPABILITY_MARKER + '\n                                 "write_harmonizer_clips",\n',
    )
    source = source.replace(
        required[3],
        required[3] +
        '                        elif command_type == "write_harmonizer_clips":\n'
        '                            ' + DISPATCH_MARKER + '\n'
        '                            result = write_harmonizer_clips(self._song, params)\n',
    )
    source = source.replace(
        required[4],
        required[4] + '                ' + TIMEOUT_MARKER + '\n'
        '                long_running_commands["write_harmonizer_clips"] = 30.0\n',
    )
    # The installed bridge is a large file and indentation is executable
    # structure in Python. Refuse to return or install an invalid patch.
    ast.parse(source, filename="AbletonMCP/__init__.py")
    return source


def install(target_dir, dry_run=False, check=False):
    init_path = os.path.join(target_dir, "__init__.py")
    module_src = os.path.join(os.path.dirname(__file__), "harmonizer_clips.py")
    module_dst = os.path.join(target_dir, "harmonizer_clips.py")
    with open(init_path, "r", encoding="utf-8") as handle:
        original = handle.read()
    updated = patched_source(original)
    module_matches = False
    if os.path.exists(module_dst):
        with open(module_src, "rb") as left, open(module_dst, "rb") as right:
            module_matches = left.read() == right.read()
    source_matches = updated == original
    if check:
        return source_matches and module_matches
    if dry_run:
        print("init: %s" % ("unchanged" if source_matches else "would patch"))
        print("module: %s" % ("unchanged" if module_matches else "would install"))
        return True
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not source_matches:
        shutil.copy2(init_path, init_path + ".rustharmoniser-backup-" + stamp)
    if os.path.exists(module_dst) and not module_matches:
        shutil.copy2(module_dst, module_dst + ".rustharmoniser-backup-" + stamp)
    module_temp = module_dst + ".rustharmoniser-new"
    shutil.copy2(module_src, module_temp)
    os.replace(module_temp, module_dst)
    if not source_matches:
        temp = init_path + ".rustharmoniser-new"
        with open(temp, "w", encoding="utf-8") as handle:
            handle.write(updated)
        os.replace(temp, init_path)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        okay = install(args.target_dir, args.dry_run, args.check)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.check and not okay:
        print("rustHarmoniser clip writer is not installed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
