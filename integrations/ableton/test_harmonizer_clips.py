import importlib.util
import ast
import pathlib
import tempfile
import unittest


HERE = pathlib.Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("harmonizer_clips", HERE / "harmonizer_clips.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NoteId:
    def __init__(self, note_id):
        self.note_id = note_id


class MidiSpec:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Clip:
    def __init__(self, length=8.0, midi=True, note_ids=(), fail_add=False):
        self.is_midi_clip = midi
        self.is_recording = False
        self.length = length
        self.start_marker = 0.0
        self.loop_start = 0.0
        self._loop_end = length
        self._end_marker = length
        self.name = "Old"
        self.identity = object()
        self.note_ids = list(note_ids)
        self.added = []
        self.fail_add = fail_add
        self.marker_writes = []

    @property
    def loop_end(self):
        return self._loop_end

    @loop_end.setter
    def loop_end(self, value):
        self.marker_writes.append(("loop_end", value))
        self._loop_end = value

    @property
    def end_marker(self):
        return self._end_marker

    @end_marker.setter
    def end_marker(self, value):
        if value < self.start_marker:
            raise ValueError("end before start")
        self.marker_writes.append(("end_marker", value))
        self._end_marker = value

    def get_all_notes_extended(self):
        return [NoteId(i) for i in self.note_ids]

    def remove_notes_by_id(self, ids):
        self.note_ids = [i for i in self.note_ids if i not in ids]

    def add_new_notes(self, notes):
        if self.fail_add:
            raise RuntimeError("simulated write failure")
        self.added.extend(notes)


class Slot:
    def __init__(self, clip=None):
        self.clip = clip
        self.has_clip = clip is not None
        self.create_count = 0

    def create_clip(self, length):
        self.create_count += 1
        self.clip = Clip(length=length)
        self.has_clip = True


class Track:
    def __init__(self, slots, midi=True):
        self.clip_slots = slots
        self.has_midi_input = midi


class Song:
    def __init__(self, tracks):
        self.tracks = tracks


def note(pitch=60, start=0.0, duration=1.0, velocity=0):
    return {"pitch": pitch, "start_time": start, "duration": duration,
            "velocity": velocity, "mute": False, "probability": 1.0}


def route(track, slot, length=4.0, notes=None):
    return {"track_index": track, "clip_index": slot, "name": "Generated",
            "length": length, "notes": [] if notes is None else notes}


class ClipWriterTests(unittest.TestCase):
    def test_replaces_existing_clip_in_place_and_clears_all_note_ids(self):
        clip = Clip(note_ids=(-4, 10, 99))
        identity = clip.identity
        song = Song([Track([Slot(clip)])])
        result = MODULE.write_harmonizer_clips(song, {"clips": [route(0, 0, notes=[note()])]}, MidiSpec)
        self.assertTrue(result["complete"])
        self.assertIs(song.tracks[0].clip_slots[0].clip.identity, identity)
        self.assertEqual(clip.note_ids, [])
        self.assertEqual(len(clip.added), 1)
        self.assertEqual(clip.added[0].velocity, 0)
        self.assertEqual(clip.added[0].velocity_deviation, 0.0)

    def test_resize_orders_are_safe_for_shorter_and_longer_clips(self):
        short = Clip(length=8.0)
        long = Clip(length=2.0)
        song = Song([Track([Slot(short), Slot(long)])])
        MODULE.write_harmonizer_clips(song, {"clips": [route(0, 0, 3.0), route(0, 1, 7.0)]}, MidiSpec)
        self.assertEqual(short.marker_writes[-2:], [("loop_end", 3.0), ("end_marker", 3.0)])
        self.assertEqual(long.marker_writes[-2:], [("end_marker", 7.0), ("loop_end", 7.0)])

    def test_empty_slot_creates_one_clip_and_empty_notes_are_valid(self):
        slot = Slot()
        result = MODULE.write_harmonizer_clips(Song([Track([slot])]), {"clips": [route(0, 0)]}, MidiSpec)
        self.assertTrue(result["complete"])
        self.assertEqual(slot.create_count, 1)
        self.assertEqual(slot.clip.added, [])

    def test_preflight_rejects_audio_recording_and_unsupported_clear_without_mutation(self):
        good = Clip(note_ids=(1,))
        unsupported = Clip(note_ids=(2,))
        unsupported.get_all_notes_extended = None
        song = Song([Track([Slot(good), Slot(unsupported)])])
        with self.assertRaisesRegex(ValueError, "clear API"):
            MODULE.write_harmonizer_clips(song, {"clips": [route(0, 0), route(0, 1)]}, MidiSpec)
        self.assertEqual(good.note_ids, [1])
        audio = Clip(midi=False)
        with self.assertRaisesRegex(ValueError, "audio"):
            MODULE.preflight_harmonizer_clips(Song([Track([Slot(audio)])]), {"clips": [route(0, 0)]}, MidiSpec)
        recording = Clip(); recording.is_recording = True
        with self.assertRaisesRegex(ValueError, "recording"):
            MODULE.preflight_harmonizer_clips(Song([Track([Slot(recording)])]), {"clips": [route(0, 0)]}, MidiSpec)

    def test_runtime_error_reports_partial_batch_and_stops(self):
        clips = [Clip(), Clip(fail_add=True), Clip()]
        song = Song([Track([Slot(c) for c in clips])])
        result = MODULE.write_harmonizer_clips(song, {"clips": [route(0, 0, notes=[note()]), route(0, 1, notes=[note()]), route(0, 2, notes=[note()])]}, MidiSpec)
        self.assertFalse(result["complete"])
        self.assertEqual([s["status"] for s in result["clips"]], ["written", "error", "not_attempted"])
        self.assertEqual(clips[2].name, "Old")

    def test_preflight_rejects_duplicate_source_collision_and_bad_note_before_mutation(self):
        clips = [Clip(note_ids=(1,)), Clip(note_ids=(2,))]
        song = Song([Track([Slot(c) for c in clips])])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE.preflight_harmonizer_clips(song, {"clips": [route(0, 0), route(0, 0)]}, MidiSpec)
        with self.assertRaisesRegex(ValueError, "collides"):
            MODULE.preflight_harmonizer_clips(song, {"source": {"track_index": 0, "clip_index": 0}, "clips": [route(0, 0)]}, MidiSpec)
        bad = route(0, 1, notes=[note(pitch=128)])
        with self.assertRaisesRegex(ValueError, "pitch"):
            MODULE.write_harmonizer_clips(song, {"clips": [route(0, 0), bad]}, MidiSpec)
        self.assertEqual(clips[0].note_ids, [1])


class InstallerTests(unittest.TestCase):
    def test_installer_patches_known_bridge_and_is_idempotent(self):
        install_spec = importlib.util.spec_from_file_location("install", HERE / "install.py")
        install = importlib.util.module_from_spec(install_spec)
        install_spec.loader.exec_module(install)
        fixture = pathlib.Path("/tmp/harmonizer-ableton-init-before.py")
        if not fixture.exists():
            self.skipTest("installed AbletonMCP snapshot is unavailable")
        source = fixture.read_text()
        patched = install.patched_source(source)
        ast.parse(patched)
        self.assertIn('"write_harmonizer_clips"', patched)
        self.assertIn('long_running_commands["write_harmonizer_clips"] = 30.0', patched)
        self.assertEqual(install.patched_source(patched), patched)
        with self.assertRaisesRegex(RuntimeError, "unknown"):
            install.patched_source("unknown bridge")

if __name__ == "__main__":
    unittest.main()
