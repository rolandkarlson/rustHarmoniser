"""Validated, in-place Session clip replacement for rustHarmoniser.

This module is deliberately independent of ControlSurface so its planning and
mutation behavior can be tested with small Live-like stubs.  The caller must
run ``write_harmonizer_clips`` on Live's main thread.
"""
from __future__ import absolute_import, print_function, unicode_literals

import math


MAX_CLIPS = 64
MAX_NOTES_PER_CLIP = 100000
MAX_NOTES_PER_BATCH = 250000
MAX_CLIP_LENGTH = 1000000.0


def _number(value, label, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a number" % label)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("%s must be finite" % label)
    if minimum is not None and value < minimum:
        raise ValueError("%s must be at least %s" % (label, minimum))
    if maximum is not None and value > maximum:
        raise ValueError("%s must be at most %s" % (label, maximum))
    return value


def _integer(value, label, minimum=None, maximum=None):
    number = _number(value, label, minimum, maximum)
    if number != int(number):
        raise ValueError("%s must be an integer" % label)
    return int(number)


def _note_spec_factory():
    try:
        import Live
        return Live.Clip.MidiNoteSpecification
    except (ImportError, AttributeError):
        return None


def preflight_harmonizer_clips(song, params, note_spec_factory=None):
    """Validate the entire batch and select APIs before any clip is changed."""
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    clips = params.get("clips", [])
    if not isinstance(clips, list) or not clips or len(clips) > MAX_CLIPS:
        raise ValueError("clips must contain between 1 and %d routes" % MAX_CLIPS)

    source = params.get("source")
    source_key = None
    if source is not None:
        if not isinstance(source, dict):
            raise ValueError("source must be an object or null")
        source_key = (
            _integer(source.get("track_index"), "source.track_index", 0),
            _integer(source.get("clip_index"), "source.clip_index", 0),
        )

    factory = note_spec_factory if note_spec_factory is not None else _note_spec_factory()
    prepared = []
    total_notes = 0
    destinations = set()
    for route_index, route in enumerate(clips):
        label = "clips[%d]" % route_index
        if not isinstance(route, dict):
            raise ValueError("%s must be an object" % label)
        track_index = _integer(route.get("track_index"), label + ".track_index", 0)
        clip_index = _integer(route.get("clip_index"), label + ".clip_index", 0)
        key = (track_index, clip_index)
        if key in destinations:
            raise ValueError("duplicate destination track %d clip %d" % key)
        if source_key == key:
            raise ValueError("output destination collides with source track %d clip %d" % key)
        destinations.add(key)
        if track_index >= len(song.tracks):
            raise ValueError("%s track index is out of range" % label)
        track = song.tracks[track_index]
        if not bool(getattr(track, "has_midi_input", False)):
            raise ValueError("%s target is not a MIDI track" % label)
        if clip_index >= len(track.clip_slots):
            raise ValueError("%s clip index is out of range" % label)
        slot = track.clip_slots[clip_index]
        existing = slot.clip if slot.has_clip else None
        if existing is not None and not bool(getattr(existing, "is_midi_clip", False)):
            raise ValueError("%s contains an audio clip" % label)
        if existing is not None and bool(getattr(existing, "is_recording", False)):
            raise ValueError("%s is recording" % label)

        length = _number(route.get("length"), label + ".length", 0.0001, MAX_CLIP_LENGTH)
        name = route.get("name", "")
        if not isinstance(name, str) or len(name) > 128:
            raise ValueError("%s.name must be a string of at most 128 characters" % label)
        notes = route.get("notes", [])
        if not isinstance(notes, list) or len(notes) > MAX_NOTES_PER_CLIP:
            raise ValueError("%s.notes exceeds the event limit" % label)
        total_notes += len(notes)
        if total_notes > MAX_NOTES_PER_BATCH:
            raise ValueError("clip batch exceeds the total event limit")

        live_notes = []
        legacy_notes = []
        for note_index, note in enumerate(notes):
            nlabel = "%s.notes[%d]" % (label, note_index)
            if not isinstance(note, dict):
                raise ValueError("%s must be an object" % nlabel)
            pitch = _integer(note.get("pitch"), nlabel + ".pitch", 0, 127)
            start = _number(note.get("start_time"), nlabel + ".start_time", 0, length)
            duration = _number(note.get("duration"), nlabel + ".duration", 0.000001, MAX_CLIP_LENGTH)
            if start + duration > length + 0.000001:
                raise ValueError("%s extends beyond the clip length" % nlabel)
            velocity = _integer(note.get("velocity"), nlabel + ".velocity", 0, 127)
            mute = note.get("mute", False)
            if not isinstance(mute, bool):
                raise ValueError("%s.mute must be boolean" % nlabel)
            probability = _number(note.get("probability", 1.0), nlabel + ".probability", 0, 1)
            if factory is not None:
                live_notes.append(factory(
                    pitch=pitch,
                    start_time=start,
                    duration=duration,
                    velocity=velocity,
                    mute=mute,
                    probability=probability,
                    velocity_deviation=0.0,
                    release_velocity=64.0,
                ))
            legacy_notes.append((pitch, start, duration, velocity, mute))

        if factory is None:
            raise ValueError("Live MidiNoteSpecification is unavailable")
        existing_note_ids = tuple()
        if existing is not None:
            if not callable(getattr(existing, "add_new_notes", None)):
                raise ValueError("target clip has no modern note-write API")
            getter = getattr(existing, "get_all_notes_extended", None)
            remover = getattr(existing, "remove_notes_by_id", None)
            if not callable(getter) or not callable(remover):
                raise ValueError("target clip has no safe all-notes clear API")
            # Read every target during preflight. If any target cannot enumerate
            # all notes, no earlier target has been touched yet.
            existing_note_ids = tuple(note.note_id for note in getter())
        prepared.append({
            "track_index": track_index,
            "clip_index": clip_index,
            "slot": slot,
            "existing": existing,
            "length": length,
            "name": name,
            "live_notes": tuple(live_notes),
            "legacy_notes": tuple(legacy_notes),
            "factory": factory,
            "existing_note_ids": existing_note_ids,
        })
    return prepared


def _clear_all_notes(clip, note_ids):
    if note_ids:
        clip.remove_notes_by_id(note_ids)
    return len(note_ids)


def _set_clip_extent(clip, length):
    # start_marker must move first because Live rejects an end marker before it.
    clip.start_marker = 0.0
    clip.loop_start = 0.0
    old_end = float(getattr(clip, "end_marker", getattr(clip, "length", length)))
    if length >= old_end:
        clip.end_marker = length
        clip.loop_end = length
    else:
        clip.loop_end = length
        clip.end_marker = length


def write_harmonizer_clips(song, params, note_spec_factory=None):
    """Preflight every route, then replace clips sequentially without playback.

    A runtime failure can leave earlier clips written.  The returned per-clip
    statuses make that partial outcome explicit; callers must never retry an
    uncertain batch automatically.
    """
    prepared = preflight_harmonizer_clips(song, params, note_spec_factory)
    statuses = []
    for index, item in enumerate(prepared):
        try:
            slot = item["slot"]
            created = not slot.has_clip
            if created:
                slot.create_clip(item["length"])
            clip = slot.clip
            if not bool(getattr(clip, "is_midi_clip", False)):
                raise RuntimeError("created target is not a MIDI clip")
            if created:
                if not callable(getattr(clip, "add_new_notes", None)):
                    raise RuntimeError("created clip has no modern note-write API")
                cleared = 0
            else:
                cleared = _clear_all_notes(clip, item["existing_note_ids"])
            _set_clip_extent(clip, item["length"])
            clip.name = item["name"]
            if item["live_notes"]:
                clip.add_new_notes(item["live_notes"])
            statuses.append({
                "track_index": item["track_index"],
                "clip_index": item["clip_index"],
                "status": "written",
                "created": created,
                "cleared_count": cleared,
                "note_count": len(item["legacy_notes"]),
            })
        except Exception as exc:
            statuses.append({
                "track_index": item["track_index"],
                "clip_index": item["clip_index"],
                "status": "error",
                "message": str(exc),
            })
            for pending in prepared[index + 1:]:
                statuses.append({
                    "track_index": pending["track_index"],
                    "clip_index": pending["clip_index"],
                    "status": "not_attempted",
                })
            return {"complete": False, "clips": statuses}
    return {"complete": True, "clips": statuses}
