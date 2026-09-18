//! Planning for a finite, configurable 3–8 voice fugal exposition.
//!
//! The planner owns form and entry identity. The harmonizer still chooses the
//! free counterpoint, but receives a separate constraint set so an imported
//! note never has to overload `Note::muted` or another exported MIDI field.

use crate::contour::Contours;
use crate::harmonizer::gen_voice;
use crate::model::{Config, FugueTransformation, Note, DEFAULT_START_NOTES};
use std::collections::HashMap;
use std::fmt;

const EPS: f64 = 0.001;
pub const MIN_FUGUE_VOICES: usize = 3;
pub const MAX_FUGUE_VOICES: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FugueError(pub String);

impl fmt::Display for FugueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.0) }
}

impl std::error::Error for FugueError {}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct NoteKey {
    channel: i32,
    start_tick: i64,
}

/// Exact pitch obligations for planned subject/answer notes. Timing, duration
/// and separate repeated-note attacks remain on the corresponding `Note`.
#[derive(Clone, Debug, Default)]
pub struct FixedNoteConstraints(HashMap<NoteKey, i32>);

impl FixedNoteConstraints {
    fn insert(&mut self, note: &Note) -> Result<(), FugueError> {
        let key = NoteKey { channel: note.channel, start_tick: tick(note.start) };
        if self.0.insert(key, note.pitch).is_some() {
            return Err(FugueError(format!(
                "fugue plan has two notes in voice {} at beat {:.4}", note.channel, note.start
            )));
        }
        Ok(())
    }

    pub fn pitch_for(&self, note: &Note) -> Option<i32> {
        self.0.get(&NoteKey { channel: note.channel, start_tick: tick(note.start) }).copied()
    }

    pub fn len(&self) -> usize { self.0.len() }
    pub fn is_empty(&self) -> bool { self.0.is_empty() }
}

fn tick(start: f64) -> i64 { (start * 10_000.0).round() as i64 }

#[derive(Debug)]
pub struct FuguePlan {
    pub notes: Vec<Note>,
    pub fixed: FixedNoteConstraints,
    pub entry_spacing: f64,
    pub entries: Vec<FugueEntry>,
    pub timeline: FugueTimeline,
}

impl FuguePlan {
    /// Fit every protected entry pitch to the already-created HarmonizerState's
    /// per-voice, per-bar Schillinger pitch classes. Each sustained note uses
    /// its onset bar for its full duration; timing and articulation never move.
    pub fn fit_to_schillinger(
        &mut self,
        progression: &[Vec<Vec<i32>>],
    ) -> Result<(), FugueError> {
        if progression.is_empty() {
            return Err(FugueError("cannot fit fugue entries without Schillinger voices".into()));
        }

        let old_fixed = self.fixed.clone();
        let mut fitted = FixedNoteConstraints::default();
        for entry in &mut self.timeline.entries {
            for note in &mut entry.protected_notes {
                note.pitch = fit_pitch_at(note, progression)?;
                fitted.insert(note)?;
            }
        }

        for note in &mut self.notes {
            if old_fixed.pitch_for(note).is_some() {
                note.pitch = fitted.pitch_for(note).ok_or_else(|| {
                    FugueError(format!(
                        "fitted fugue timeline lost voice {} at beat {:.4}",
                        note.channel, note.start
                    ))
                })?;
            }
        }

        for (preparation, approach) in &mut self.timeline.approaches {
            let target = NoteKey {
                channel: preparation.channel,
                start_tick: tick(approach.target_start),
            };
            approach.target_pitch = fitted.0.get(&target).copied().ok_or_else(|| {
                FugueError(format!(
                    "fitted fugue timeline lost approach target in voice {} at beat {:.4}",
                    preparation.channel, approach.target_start
                ))
            })?;
        }
        self.fixed = fitted;
        Ok(())
    }
}

fn fit_pitch_at(note: &Note, progression: &[Vec<Vec<i32>>]) -> Result<i32, FugueError> {
    let voice = usize::try_from(note.channel).ok();
    let bars = voice.and_then(|voice| progression.get(voice)).ok_or_else(|| {
        FugueError(format!(
            "cannot fit fugue voice {} because the Schillinger progression has {} voices",
            note.channel,
            progression.len()
        ))
    })?;
    if bars.is_empty() {
        return Err(FugueError(format!(
            "cannot fit fugue voice {} without Schillinger bars",
            note.channel
        )));
    }
    let bar_number = (note.start / 4.0).floor() as i32;
    let bar = bar_number.rem_euclid(bars.len() as i32) as usize;
    let allowed: Vec<i32> = bars[bar].iter().map(|p| p.rem_euclid(12)).collect();
    (0_i32..=127)
        .filter(|pitch| allowed.contains(&(*pitch).rem_euclid(12)))
        .min_by_key(|pitch| ((pitch - note.pitch).abs(), *pitch))
        .ok_or_else(|| {
            FugueError(format!(
                "cannot fit fugue voice {} at bar {} because its Schillinger harmony is empty",
                note.channel, bar_number
            ))
        })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FugueEntry {
    pub channel: i32,
    pub start: f64,
    pub span: f64,
    pub transpose: i32,
    pub invert: bool,
    pub time_scale: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FugueEntryRole {
    Subject,
    Answer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FugueMoment {
    Subject,
    Answer,
    Free,
    Rest,
}

#[derive(Clone, Debug)]
pub struct FugueTimelineEntry {
    pub form: FugueEntry,
    pub role: FugueEntryRole,
    pub is_return: bool,
    /// Actual protected occupancy. Without returns, trailing source-clip rest
    /// remains free exactly as in the original exposition planner.
    pub occupied_end: f64,
    pub protected_notes: Vec<Note>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FugueApproach {
    pub target_pitch: i32,
    pub target_start: f64,
}

#[derive(Clone, Debug, Default)]
pub struct FugueTimeline {
    pub entries: Vec<FugueTimelineEntry>,
    approaches: HashMap<NoteKey, FugueApproach>,
}

impl FugueTimeline {
    pub fn moment_at(&self, channel: i32, at: f64) -> FugueMoment {
        if let Some(entry) = self.entries.iter().find(|entry| {
            entry.form.channel == channel
                && at >= entry.form.start - EPS
                && at < entry.occupied_end - EPS
        }) {
            let sounding = entry.protected_notes.iter().any(|note| {
                note.start <= at + EPS && note.start + note.duration > at + EPS
            });
            if sounding {
                return match entry.role {
                    FugueEntryRole::Subject => FugueMoment::Subject,
                    FugueEntryRole::Answer => FugueMoment::Answer,
                };
            }
            return FugueMoment::Rest;
        }
        let first_start = self.entries.iter()
            .filter(|entry| entry.form.channel == channel)
            .map(|entry| entry.form.start)
            .min_by(|a, b| a.partial_cmp(b).unwrap());
        if first_start.map_or(false, |start| at < start - EPS) {
            FugueMoment::Rest
        } else {
            FugueMoment::Free
        }
    }

    /// Protected notes that begin later but sound before `note` releases.
    /// The caller computes this once per generated input note, outside its
    /// candidate-pitch loop.
    pub fn future_overlaps(&self, note: &Note) -> Vec<Note> {
        let end = note.start + note.duration;
        self.entries.iter().flat_map(|entry| entry.protected_notes.iter())
            .filter(|future| {
                future.channel != note.channel
                    && future.start > note.start + EPS
                    && future.start < end - EPS
                    && future.start + future.duration > note.start + EPS
            })
            .copied()
            .collect()
    }

    pub fn approach_for(&self, note: &Note) -> Option<FugueApproach> {
        self.approaches.get(&NoteKey {
            channel: note.channel,
            start_tick: tick(note.start),
        }).copied()
    }
}

/// Make the scoring controls appropriate for a multi-part imitative texture.
/// Tonal center, progression and user search budget stay configurable; chordal
/// completion pressure is relaxed while voice independence is strengthened.
pub fn scoring_config(config: &Config) -> Config {
    let mut c = config.clone();
    c.soprano_voice = 0;
    c.bass_voice = (c.fugue_voice_count - 1).max(0);
    c.use_floor = false;
    c.use_ceiling = false;
    c.use_resolve = false;
    c.chord_templates.clear();
    c.interval_exists_in_harmony = 0.0;
    c.root_doubling_weight = 0.0;
    c.chord_quality_weight *= 0.25;
    c.root_position_weight *= 0.5;
    c.same_note_bonus = c.same_note_bonus.min(0.4);
    c.melody_force = c.melody_force.max(0.6);
    c.melody_force_contour = None;
    c.consecutive_octav_fift = c.consecutive_octav_fift.max(6.0);
    c.no_crossing = c.no_crossing.max(10.0);
    c.min_voices_changed = -1;
    c.max_voices_changed = -1;
    c.cadence_weight = 0.0;
    c.loop_wrap_weight = 0.0;
    c.phrase_echo_weight = 0.0;
    c
}

/// Plan middle-voice subject, upper-voice real answer, then successively lower
/// subject/answer entries:
///
/// * voice 1: subject at its imported pitch;
/// * voice 0: answer transposed by `fugue_answer_interval`;
/// * voice 2 and later: alternating subject/answer entries in lower registers.
///
/// Each part rests until its entry and begins generated free counterpoint only
/// after finishing the protected entry.
pub fn plan_exposition(
    imported: &[Note],
    source_length: f64,
    config: &Config,
    contours: &Contours,
) -> Result<FuguePlan, FugueError> {
    let voice_count = validate_voice_count(config)?;
    let subject = validate_subject(imported, source_length)?;
    if !config.fugue_entry_spacing.is_finite() || config.fugue_entry_spacing < 0.0 {
        return Err(FugueError("fugue entry spacing must be finite and non-negative (zero = automatic)".into()));
    }
    let spacing = if config.fugue_entry_spacing > EPS {
        config.fugue_entry_spacing
    } else {
        source_length
    };
    if !spacing.is_finite() || spacing <= EPS {
        return Err(FugueError("fugue entry spacing must be positive and finite".into()));
    }

    let clip_beats = config.pl.checked_mul(4)
        .and_then(|bars| bars.checked_mul(config.render_length))
        .ok_or_else(|| FugueError("render length is too large".into()))?;
    let clip_len = clip_beats as f64;
    if clip_len <= EPS {
        return Err(FugueError("render length must leave a positive fugue clip".into()));
    }
    let subject_end = subject.iter().map(|n| n.start + n.duration).fold(0.0, f64::max);
    let last_entry_start = (voice_count - 1) as f64 * spacing;
    let initial_required = if config.fugue_repeat_subject {
        last_entry_start + source_length
    } else {
        last_entry_start + subject_end
    };
    if initial_required > clip_len + EPS {
        return Err(FugueError(format!(
            "{} fugue entries need {:.2} beats, but the render is {:.2}; increase Render Len or reduce Entry Gap",
            voice_count, initial_required, clip_len
        )));
    }

    let channels = entry_channels(voice_count);
    let transposes: Vec<i32> = channels.iter().enumerate()
        .map(|(ordinal, &channel)| voice_transpose(channel, ordinal, voice_count, config))
        .collect::<Result<_, _>>()?;
    let mut entries: Vec<FugueEntry> = channels.iter().enumerate().map(|(ordinal, &channel)| {
        FugueEntry {
            channel,
            start: ordinal as f64 * spacing,
            span: source_length,
            transpose: transposes[ordinal],
            invert: false,
            time_scale: 1.0,
        }
    }).collect();

    if config.fugue_repeat_subject {
        validate_transform_cycle(&config.fugue_transform_cycle, &subject)?;
        let mut available = vec![0.0; voice_count];
        for entry in &entries {
            available[entry.channel as usize] = entry.start + entry.span;
        }
        let mut cursor = (voice_count as f64 * spacing).max(last_entry_start + source_length);
        let mut index = 0usize;
        loop {
            let transform = config.fugue_transform_cycle[index % config.fugue_transform_cycle.len()];
            let channel = channels[index % channels.len()];
            let span = source_length * transform.time_scale;
            let start = cursor.max(available[channel as usize]);
            if start + span > clip_len + EPS {
                break;
            }
            entries.push(FugueEntry {
                channel,
                start,
                span,
                transpose: transposes[entry_ordinal(channel, voice_count)],
                invert: transform.invert,
                time_scale: transform.time_scale,
            });
            available[channel as usize] = start + span;
            cursor = start + spacing * transform.time_scale;
            index += 1;
        }
    }

    let mut notes = Vec::new();
    let mut fixed = FixedNoteConstraints::default();
    let mut protected_by_entry = Vec::with_capacity(entries.len());

    for entry in &entries {
        let protected = transform_subject(&subject, *entry)?;
        for note in &protected {
            fixed.insert(note)?;
            notes.push(*note);
        }
        protected_by_entry.push(protected);
    }

    // The first three channels retain the historical RNG consumption order.
    for channel in channels {
        let base = config.start_notes.get(channel as usize).copied()
            .or_else(|| DEFAULT_START_NOTES.get(channel as usize).copied())
            .unwrap_or_else(|| {
                let ordinal = entry_ordinal(channel, voice_count);
                (subject[0].pitch + transposes[ordinal])
                    .clamp(24, 90)
            });
        let generated = gen_voice(base, &config.voice_rhythm, &[0], channel, 1, config, contours);
        if config.fugue_repeat_subject {
            let mut blocks: Vec<(f64, f64)> = entries.iter()
                .filter(|e| e.channel == channel)
                .map(|e| (e.start, e.start + e.span))
                .collect();
            blocks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            if let Some(first) = blocks.first_mut() {
                first.0 = 0.0;
            }
            for note in generated {
                notes.extend(subtract_blocks(note, &blocks));
            }
        } else {
            // Preserve the original exposition behavior exactly while repeats
            // are disabled: the part joins after its protected notes finish.
            let entry = entries.iter().find(|e| e.channel == channel).unwrap();
            let entry_end = entry.start + subject_end;
            notes.extend(generated.into_iter().filter_map(|mut n| {
                let end = n.start + n.duration;
                if end <= entry_end + EPS { return None; }
                if n.start < entry_end {
                    n.start = entry_end;
                    n.duration = end - entry_end;
                }
                Some(n)
            }));
        }
    }

    notes.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap()
        .then_with(|| a.channel.cmp(&b.channel)));
    let timeline = build_timeline(
        &entries,
        protected_by_entry,
        &notes,
        &fixed,
        voice_count,
        config.fugue_repeat_subject,
        subject_end,
    );
    Ok(FuguePlan { notes, fixed, entry_spacing: spacing, entries, timeline })
}

fn build_timeline(
    entries: &[FugueEntry],
    protected_by_entry: Vec<Vec<Note>>,
    notes: &[Note],
    fixed: &FixedNoteConstraints,
    voice_count: usize,
    repeat_enabled: bool,
    subject_end: f64,
) -> FugueTimeline {
    let mut timeline = FugueTimeline::default();
    for (index, (&form, protected_notes)) in entries.iter().zip(protected_by_entry).enumerate() {
        let ordinal = entry_ordinal(form.channel, voice_count);
        timeline.entries.push(FugueTimelineEntry {
            form,
            role: if ordinal % 2 == 0 { FugueEntryRole::Subject } else { FugueEntryRole::Answer },
            is_return: index >= voice_count,
            occupied_end: if repeat_enabled {
                form.start + form.span
            } else {
                form.start + subject_end
            },
            protected_notes,
        });
    }

    for entry in timeline.entries.iter().filter(|entry| entry.is_return) {
        let Some(target) = entry.protected_notes.iter()
            .min_by(|a, b| a.start.partial_cmp(&b.start).unwrap())
        else { continue };
        let preparation = notes.iter().filter(|note| {
            note.channel == entry.form.channel
                && fixed.pitch_for(note).is_none()
                && note.start < entry.form.start - EPS
                && note.start + note.duration <= entry.form.start + EPS
        }).max_by(|a, b| {
            (a.start + a.duration).partial_cmp(&(b.start + b.duration)).unwrap()
        });
        if let Some(note) = preparation {
            timeline.approaches.entry(NoteKey {
                channel: note.channel,
                start_tick: tick(note.start),
            }).or_insert(FugueApproach {
                target_pitch: target.pitch,
                target_start: target.start,
            });
        }
    }
    timeline
}

pub fn validate_voice_count(config: &Config) -> Result<usize, FugueError> {
    let count = usize::try_from(config.fugue_voice_count).unwrap_or(0);
    if !(MIN_FUGUE_VOICES..=MAX_FUGUE_VOICES).contains(&count) {
        return Err(FugueError(format!(
            "fugue voice count must be between {MIN_FUGUE_VOICES} and {MAX_FUGUE_VOICES}"
        )));
    }
    for (name, weight) in [
        ("subject anticipation", config.fugue_subject_anticipation),
        ("entry approach", config.fugue_entry_approach),
    ] {
        if !weight.is_finite() || weight < 0.0 {
            return Err(FugueError(format!("fugue {name} weight must be finite and non-negative")));
        }
    }
    Ok(count)
}

fn entry_channels(voice_count: usize) -> Vec<i32> {
    let mut channels = Vec::with_capacity(voice_count);
    channels.extend([1, 0, 2]);
    channels.extend((3..voice_count).map(|channel| channel as i32));
    channels
}

fn entry_ordinal(channel: i32, voice_count: usize) -> usize {
    entry_channels(voice_count).iter().position(|&c| c == channel).unwrap()
}

fn automatic_register_shift(channel: usize) -> i32 {
    if channel < 2 { 0 } else { -12 * ((channel as i32 + 1) / 2) }
}

/// Register shift for one output channel. Three-voice configurations keep the
/// historical bass field authoritative. With more voices, an explicit vector
/// value wins and a missing/short vector uses the octave-spread default.
pub fn voice_register_shift(channel: usize, voice_count: usize, config: &Config) -> i32 {
    if voice_count == 3 && channel == 2 {
        config.fugue_bass_subject_shift
    } else {
        config.fugue_voice_register_shifts.get(channel).copied()
            .unwrap_or_else(|| automatic_register_shift(channel))
    }
}

fn voice_transpose(channel: i32, ordinal: usize, voice_count: usize, config: &Config) -> Result<i32, FugueError> {
    let register = voice_register_shift(channel as usize, voice_count, config);
    if ordinal % 2 == 1 {
        register.checked_add(config.fugue_answer_interval)
            .ok_or_else(|| FugueError("fugue answer and register transposition overflowed".into()))
    } else {
        Ok(register)
    }
}

fn validate_transform_cycle(
    cycle: &[FugueTransformation],
    subject: &[Note],
) -> Result<(), FugueError> {
    if cycle.is_empty() {
        return Err(FugueError("repeat subject needs at least one transformation step".into()));
    }
    for transform in cycle {
        if !transform.time_scale.is_finite()
            || ![0.5, 1.0, 2.0, 4.0].iter().any(|v| (transform.time_scale - v).abs() < 1e-9)
        {
            return Err(FugueError("subject time scale must be 0.5, 1, 2 or 4".into()));
        }
        for note in subject {
            let duration = note.duration * transform.time_scale;
            if !duration.is_finite() || duration <= EPS {
                return Err(FugueError("transformed subject notes must have a positive duration".into()));
            }
            if duration > 32.0 + EPS {
                return Err(FugueError("transformed subject notes may not sustain for more than 32 beats".into()));
            }
        }
    }
    Ok(())
}

fn transform_subject(subject: &[Note], entry: FugueEntry) -> Result<Vec<Note>, FugueError> {
    let axis = subject[0].pitch as i64;
    subject.iter().map(|source| {
        let raw = if entry.invert {
            axis * 2 - source.pitch as i64
        } else {
            source.pitch as i64
        };
        let pitch = raw + entry.transpose as i64;
        if !(0..=127).contains(&pitch) {
            return Err(FugueError(format!(
                "fugue transformation puts MIDI pitch {pitch} outside 0..127"
            )));
        }
        let duration = source.duration * entry.time_scale;
        if !duration.is_finite() || duration <= EPS || duration > 32.0 + EPS {
            return Err(FugueError("transformed subject duration must be positive and at most 32 beats".into()));
        }
        let mut note = *source;
        note.pitch = pitch as i32;
        note.start = entry.start + source.start * entry.time_scale;
        note.duration = duration;
        note.channel = entry.channel;
        note.muted = 0;
        Ok(note)
    }).collect()
}

fn subtract_blocks(note: Note, blocks: &[(f64, f64)]) -> Vec<Note> {
    let mut segments = vec![(note.start, note.start + note.duration)];
    for &(block_start, block_end) in blocks {
        let mut next = Vec::new();
        for (start, end) in segments {
            if end <= block_start + EPS || start >= block_end - EPS {
                next.push((start, end));
                continue;
            }
            if start < block_start - EPS {
                next.push((start, block_start));
            }
            if end > block_end + EPS {
                next.push((block_end, end));
            }
        }
        segments = next;
    }
    segments.into_iter().filter_map(|(start, end)| {
        if end - start <= EPS { return None; }
        let mut fragment = note;
        fragment.start = start;
        fragment.duration = end - start;
        Some(fragment)
    }).collect()
}

fn validate_subject(imported: &[Note], source_length: f64) -> Result<Vec<Note>, FugueError> {
    if !source_length.is_finite() || source_length <= EPS {
        return Err(FugueError("the imported subject clip must have a positive finite length".into()));
    }
    let mut notes: Vec<Note> = imported.iter().copied().filter(|n| n.muted == 0).collect();
    if notes.is_empty() {
        return Err(FugueError("the imported subject has no unmuted notes".into()));
    }
    for n in &notes {
        if !n.start.is_finite() || !n.duration.is_finite() || n.start < 0.0 || n.duration <= EPS {
            return Err(FugueError("subject notes need finite, non-negative starts and positive durations".into()));
        }
        if !(0..=127).contains(&n.pitch) {
            return Err(FugueError(format!("subject MIDI pitch {} is outside 0..127", n.pitch)));
        }
        if n.start + n.duration > source_length + EPS {
            return Err(FugueError(format!(
                "subject note at beat {:.3} extends past the {:.3}-beat source clip",
                n.start, source_length
            )));
        }
        if n.duration > 32.0 + EPS {
            return Err(FugueError("subject notes may not sustain for more than 32 beats".into()));
        }
    }
    notes.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap()
        .then_with(|| a.pitch.cmp(&b.pitch)));
    for pair in notes.windows(2) {
        if pair[1].start < pair[0].start + pair[0].duration - EPS {
            return Err(FugueError(format!(
                "the imported fugue subject is polyphonic near beat {:.3}; use one note at a time",
                pair[1].start
            )));
        }
    }
    Ok(notes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config {
        let mut c = Config::default();
        c.main_pitch = 0;
        c.render_length = 2;
        c
    }

    fn subject() -> Vec<Note> {
        vec![
            Note::new(60, 0.0, 1.0, 90, 0, 0),
            Note::new(60, 1.0, 0.5, 75, 0, 0),
            Note::new(62, 1.5, 2.5, 80, 0, 0),
        ]
    }

    #[test]
    fn plans_tonic_answer_tonic_with_exact_articulation_and_rests() {
        let c = config();
        let contours = Contours::from_config(&c);
        let p = plan_exposition(&subject(), 4.0, &c, &contours).unwrap();
        assert_eq!(p.entry_spacing, 4.0);
        for (channel, offset, shift) in [(1, 0.0, 0), (0, 4.0, 7), (2, 8.0, -12)] {
            let entry: Vec<_> = p.notes.iter().filter(|n| n.channel == channel
                && n.start >= offset - EPS && n.start < offset + 4.0 - EPS).collect();
            assert_eq!(entry.len(), 3);
            assert_eq!(entry.iter().map(|n| n.pitch).collect::<Vec<_>>(), vec![60 + shift, 60 + shift, 62 + shift]);
            assert_eq!(entry.iter().map(|n| n.start).collect::<Vec<_>>(), vec![offset, offset + 1.0, offset + 1.5]);
            assert_eq!(entry.iter().map(|n| n.duration).collect::<Vec<_>>(), vec![1.0, 0.5, 2.5]);
            assert!(!p.notes.iter().any(|n| n.channel == channel && n.start < offset - EPS));
        }
        assert_eq!(p.fixed.len(), 9);
    }

    #[test]
    fn rejects_polyphonic_and_invalid_subjects() {
        let c = config();
        let contours = Contours::from_config(&c);
        let mut poly = subject();
        poly.push(Note::new(67, 0.5, 1.0, 90, 0, 0));
        assert!(plan_exposition(&poly, 4.0, &c, &contours).unwrap_err().0.contains("polyphonic"));
        assert!(plan_exposition(&[], 4.0, &c, &contours).is_err());
        assert!(plan_exposition(&subject(), 0.0, &c, &contours).is_err());

        let mut bad_spacing = c.clone();
        bad_spacing.fugue_entry_spacing = f64::NAN;
        assert!(plan_exposition(&subject(), 4.0, &bad_spacing, &contours).is_err());
        bad_spacing.fugue_entry_spacing = -1.0;
        assert!(plan_exposition(&subject(), 4.0, &bad_spacing, &contours).is_err());

        let mut too_short = c.clone();
        too_short.render_length = 0;
        assert!(plan_exposition(&subject(), 4.0, &too_short, &contours).is_err());

        let mut bad_answer = c.clone();
        bad_answer.fugue_answer_interval = 100;
        assert!(plan_exposition(&subject(), 4.0, &bad_answer, &contours).is_err());
    }

    #[test]
    fn ignores_muted_source_notes_during_validation() {
        let c = config();
        let contours = Contours::from_config(&c);
        let mut notes = subject();
        let mut muted_overlap = Note::new(72, 0.25, 2.0, 100, 1, 0);
        muted_overlap.muted = 1;
        notes.push(muted_overlap);
        let p = plan_exposition(&notes, 4.0, &c, &contours).unwrap();
        assert_eq!(p.fixed.len(), 9);
    }

    #[test]
    fn fugue_profile_enforces_three_voice_roles_and_clears_chord_texture_controls() {
        let mut c = config();
        c.soprano_voice = 9;
        c.bass_voice = 12;
        c.melody_force = 0.0;
        c.melody_force_contour = Some(vec![vec![0.0; 8]; 16]);
        c.chord_templates = vec![crate::model::ChordTemplate::Bare(vec![0, 4, 7])];
        c.interval_exists_in_harmony = 99.0;
        let f = scoring_config(&c);
        assert_eq!((f.soprano_voice, f.bass_voice), (0, 2));
        assert!(f.melody_force >= 0.6);
        assert!(f.melody_force_contour.is_none());
        assert!(f.chord_templates.is_empty());
        assert_eq!(f.interval_exists_in_harmony, 0.0);
    }

    #[test]
    fn plans_four_five_and_eight_voice_expositions_with_alternating_answers() {
        for count in [4, 5, 8] {
            let mut c = config();
            c.fugue_voice_count = count;
            let contours = Contours::from_config(&c);
            let p = plan_exposition(&subject(), 4.0, &c, &contours).unwrap();
            let expected_channels: Vec<i32> = [1, 0, 2].into_iter()
                .chain(3..count)
                .collect();
            assert_eq!(p.entries.len(), count as usize);
            assert_eq!(p.entries.iter().map(|e| e.channel).collect::<Vec<_>>(), expected_channels);
            assert_eq!(p.fixed.len(), subject().len() * count as usize);
            for (ordinal, entry) in p.entries.iter().enumerate() {
                let register = automatic_register_shift(entry.channel as usize);
                let expected = register + if ordinal % 2 == 1 { c.fugue_answer_interval } else { 0 };
                assert_eq!(entry.transpose, expected);
                assert_eq!(entry.start, ordinal as f64 * 4.0);
                assert!(!p.notes.iter().any(|n| n.channel == entry.channel && n.start < entry.start - EPS));
                assert_eq!(
                    p.timeline.entries[ordinal].role,
                    if ordinal % 2 == 0 { FugueEntryRole::Subject } else { FugueEntryRole::Answer },
                );
            }
            let scored = scoring_config(&c);
            assert_eq!(scored.soprano_voice, 0);
            assert_eq!(scored.bass_voice, count - 1);
        }
    }

    #[test]
    fn register_vector_overrides_wide_defaults_and_three_voice_bass_stays_legacy() {
        let mut c = config();
        c.fugue_voice_register_shifts = vec![2, 1, -10, -30];
        c.fugue_bass_subject_shift = -15;
        assert_eq!(voice_register_shift(2, 3, &c), -15);
        assert_eq!(voice_register_shift(3, 4, &c), -30);
        c.fugue_voice_register_shifts.clear();
        assert_eq!(voice_register_shift(3, 4, &c), -24);
        assert_eq!(voice_register_shift(7, 8, &c), -48);
    }

    #[test]
    fn rejects_voice_counts_outside_three_through_eight() {
        for count in [i32::MIN, -1, 0, 2, 9, i32::MAX] {
            let mut c = config();
            c.fugue_voice_count = count;
            assert!(plan_exposition(&subject(), 4.0, &c, &Contours::from_config(&c))
                .unwrap_err().0.contains("between 3 and 8"));
        }
        let mut c = config();
        c.fugue_subject_anticipation = -1.0;
        assert!(validate_voice_count(&c).unwrap_err().0.contains("anticipation"));
        c.fugue_subject_anticipation = 0.0;
        c.fugue_entry_approach = f64::NAN;
        assert!(validate_voice_count(&c).unwrap_err().0.contains("entry approach"));
    }

    #[test]
    fn timeline_distinguishes_roles_free_time_and_real_entry_rests() {
        let c = config();
        let rested = vec![Note::new(60, 1.0, 1.0, 90, 0, 0)];
        let p = plan_exposition(&rested, 4.0, &c, &Contours::from_config(&c)).unwrap();
        assert_eq!(p.timeline.moment_at(1, 0.5), FugueMoment::Rest);
        assert_eq!(p.timeline.moment_at(1, 1.5), FugueMoment::Subject);
        // Repeat-off compatibility: the source clip's trailing rest is free
        // after the last protected note, as it was before timelines existed.
        assert_eq!(p.timeline.moment_at(1, 2.5), FugueMoment::Free);
        assert_eq!(p.timeline.moment_at(0, 3.5), FugueMoment::Rest);
        assert_eq!(p.timeline.moment_at(0, 5.5), FugueMoment::Answer);

        let sustaining = Note::new(65, 7.0, 2.5, 80, 0, 1);
        let overlaps = p.timeline.future_overlaps(&sustaining);
        assert_eq!(overlaps.len(), 1);
        assert_eq!((overlaps[0].channel, overlaps[0].start), (2, 9.0));
    }

    #[test]
    fn timeline_marks_returns_and_uses_first_sounding_note_after_leading_rest() {
        let mut c = config();
        c.fugue_repeat_subject = true;
        c.voice_rhythm_patterns = vec![vec![0.5]; 3];
        let rested = vec![Note::new(60, 1.0, 1.0, 90, 0, 0)];
        let p = plan_exposition(&rested, 4.0, &c, &Contours::from_config(&c)).unwrap();
        let first_return = p.timeline.entries.iter().find(|entry| entry.is_return).unwrap();
        assert_eq!(first_return.role, FugueEntryRole::Subject);
        assert!(first_return.form.invert);
        assert_eq!(first_return.form.start, 12.0);
        assert_eq!(first_return.protected_notes[0].start, 13.0);
        let preparation = p.notes.iter().find(|note| {
            note.channel == first_return.form.channel
                && p.fixed.pitch_for(note).is_none()
                && p.timeline.approach_for(note).is_some()
        }).unwrap();
        assert_eq!(preparation.start + preparation.duration, 12.0);
        assert_eq!(p.timeline.approach_for(preparation).unwrap().target_start, 13.0);
    }

    fn fixed_entry_notes<'a>(plan: &'a FuguePlan, entry: &FugueEntry) -> Vec<&'a Note> {
        plan.notes.iter().filter(|note| {
            note.channel == entry.channel
                && note.start >= entry.start - EPS
                && note.start < entry.start + entry.span - EPS
                && plan.fixed.pitch_for(note).is_some()
        }).collect()
    }

    #[test]
    fn repeat_cycle_inverts_and_stretches_from_the_original_subject() {
        let mut c = config();
        c.fugue_repeat_subject = true;
        let contours = Contours::from_config(&c);
        let p = plan_exposition(&subject(), 4.0, &c, &contours).unwrap();
        assert_eq!(p.entries.iter().map(|e| e.start).collect::<Vec<_>>(), vec![0.0, 4.0, 8.0, 12.0, 16.0, 24.0]);
        assert_eq!(p.entries.iter().map(|e| e.channel).collect::<Vec<_>>(), vec![1, 0, 2, 1, 0, 2]);

        let inverted = fixed_entry_notes(&p, &p.entries[3]);
        assert_eq!(inverted.iter().map(|n| n.pitch).collect::<Vec<_>>(), vec![60, 60, 58]);
        assert_eq!(inverted.iter().map(|n| n.start).collect::<Vec<_>>(), vec![12.0, 13.0, 13.5]);

        let stretched_original = fixed_entry_notes(&p, &p.entries[4]);
        assert_eq!(stretched_original.iter().map(|n| n.pitch).collect::<Vec<_>>(), vec![67, 67, 69]);
        assert_eq!(stretched_original.iter().map(|n| n.start).collect::<Vec<_>>(), vec![16.0, 18.0, 19.0]);
        assert_eq!(stretched_original.iter().map(|n| n.duration).collect::<Vec<_>>(), vec![2.0, 1.0, 5.0]);

        let stretched_inversion = fixed_entry_notes(&p, &p.entries[5]);
        assert_eq!(stretched_inversion.iter().map(|n| n.pitch).collect::<Vec<_>>(), vec![48, 48, 46]);
        assert_eq!(stretched_inversion.iter().map(|n| n.start).collect::<Vec<_>>(), vec![24.0, 26.0, 27.0]);
        assert_eq!(p.fixed.len(), subject().len() * p.entries.len());
    }

    #[test]
    fn wider_repeat_rotation_reaches_added_voices_and_protects_their_blocks() {
        let mut c = config();
        c.fugue_voice_count = 4;
        c.fugue_repeat_subject = true;
        c.render_length = 3;
        let p = plan_exposition(&subject(), 4.0, &c, &Contours::from_config(&c)).unwrap();
        assert_eq!(p.entries.iter().take(8).map(|e| e.channel).collect::<Vec<_>>(),
            vec![1, 0, 2, 3, 1, 0, 2, 3]);
        assert_eq!(p.entries.iter().take(8).map(|e| e.start).collect::<Vec<_>>(),
            vec![0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 28.0, 36.0]);
        let added_repeat = &p.entries[7];
        assert_eq!((added_repeat.transpose, added_repeat.invert, added_repeat.time_scale), (-17, false, 1.0));
        for note in p.notes.iter().filter(|n| n.channel == 3
            && n.start < added_repeat.start + added_repeat.span - EPS
            && n.start + n.duration > added_repeat.start + EPS)
        {
            assert!(p.fixed.pitch_for(note).is_some());
        }
    }

    #[test]
    fn old_snapshot_defaults_to_three_voices_without_a_register_vector() {
        let mut value = serde_json::to_value(Config::default()).unwrap();
        let object = value.as_object_mut().unwrap();
        object.remove("fugue_voice_count");
        object.remove("fugue_voice_register_shifts");
        object.remove("fugue_fit_to_scale");
        let loaded: Config = serde_json::from_value(value).unwrap();
        assert_eq!(loaded.fugue_voice_count, 3);
        assert!(loaded.fugue_voice_register_shifts.is_empty());
        assert!(!loaded.fugue_fit_to_scale);
    }

    #[test]
    fn scale_fit_updates_every_protected_representation_at_note_onsets() {
        let mut c = config();
        c.fugue_repeat_subject = true;
        c.voice_rhythm_patterns = vec![vec![0.5]; 3];
        let source = vec![
            Note::new(61, 1.0, 1.0, 91, 0, 0),
            Note::new(61, 2.0, 0.5, 79, 0, 0),
            Note::new(64, 2.5, 1.5, 84, 0, 0),
        ];
        let mut plan = plan_exposition(&source, 4.0, &c, &Contours::from_config(&c)).unwrap();
        let before: Vec<_> = plan
            .timeline
            .entries
            .iter()
            .flat_map(|entry| entry.protected_notes.iter())
            .map(|n| (n.channel, n.start, n.duration, n.velocity))
            .collect();

        // Deliberately vary pitch classes by both voice and bar. Voice 1/bar 0
        // gives a true tie around C# (C and D), which must resolve downward.
        let mut progression = Vec::new();
        for voice in 0..3 {
            progression.push(
                (0..8)
                    .map(|bar| vec![((voice * 4 + bar * 2) % 12) as i32])
                    .collect::<Vec<_>>(),
            );
        }
        progression[1][0] = vec![0, 2];
        plan.fit_to_schillinger(&progression).unwrap();
        assert_eq!(plan.timeline.moment_at(1, 0.5), FugueMoment::Rest);

        let after: Vec<_> = plan
            .timeline
            .entries
            .iter()
            .flat_map(|entry| entry.protected_notes.iter())
            .map(|n| (n.channel, n.start, n.duration, n.velocity))
            .collect();
        assert_eq!(after, before, "scale fitting must not alter articulation");
        assert_eq!(plan.timeline.entries[0].protected_notes[0].pitch, 60);
        assert!(plan.timeline.entries.iter().any(|entry| entry.form.invert));
        assert!(plan
            .timeline
            .entries
            .iter()
            .any(|entry| entry.form.time_scale == 2.0));

        for entry in &plan.timeline.entries {
            for note in &entry.protected_notes {
                let voice = note.channel as usize;
                let bar = (note.start / 4.0).floor() as usize % progression[voice].len();
                assert!(progression[voice][bar]
                    .iter()
                    .any(|pc| pc.rem_euclid(12) == note.pitch.rem_euclid(12)));
                let in_plan = plan.notes.iter().find(|candidate| {
                    candidate.channel == note.channel
                        && tick(candidate.start) == tick(note.start)
                        && plan.fixed.pitch_for(candidate).is_some()
                }).unwrap();
                assert_eq!(in_plan.pitch, note.pitch);
                assert_eq!(plan.fixed.pitch_for(in_plan), Some(note.pitch));
            }
        }

        // The 2x entry's long final note stays one event even though it crosses
        // a bar line, and repeated attacks remain separate.
        let stretched = plan.timeline.entries.iter().find(|entry| entry.form.time_scale == 2.0).unwrap();
        assert_eq!(stretched.protected_notes.len(), source.len());
        assert!(stretched.protected_notes.iter().any(|note| note.duration == 3.0));
        assert_ne!(stretched.protected_notes[0].start, stretched.protected_notes[1].start);

        let first_return = plan.timeline.entries.iter().find(|entry| entry.is_return).unwrap();
        let target = first_return.protected_notes.iter().min_by(|a, b| a.start.partial_cmp(&b.start).unwrap()).unwrap();
        let preparation = plan.notes.iter().find(|note| {
            note.channel == first_return.form.channel && plan.timeline.approach_for(note).is_some()
        }).unwrap();
        assert_eq!(plan.timeline.approach_for(preparation).unwrap().target_pitch, target.pitch);
    }

    #[test]
    fn repeat_entry_gap_scales_and_same_voice_waits_until_available() {
        let mut c = config();
        c.fugue_repeat_subject = true;
        c.fugue_entry_spacing = 8.0;
        c.render_length = 3;
        let contours = Contours::from_config(&c);
        let p = plan_exposition(&subject(), 4.0, &c, &contours).unwrap();
        assert_eq!(p.entries.iter().take(5).map(|e| e.start).collect::<Vec<_>>(), vec![0.0, 8.0, 16.0, 24.0, 32.0]);

        c.fugue_entry_spacing = 1.0;
        c.fugue_transform_cycle = vec![FugueTransformation::new(false, 2.0)];
        let contours = Contours::from_config(&c);
        let p = plan_exposition(&subject(), 4.0, &c, &contours).unwrap();
        assert_eq!(p.entries.iter().skip(3).take(4).map(|e| e.start).collect::<Vec<_>>(), vec![6.0, 8.0, 10.0, 14.0]);
        assert!(p.entries.iter().skip(3).all(|e| e.span == 8.0));
        for channel in 0..3 {
            let blocks: Vec<_> = p.entries.iter().filter(|e| e.channel == channel).collect();
            assert!(blocks.windows(2).all(|w| w[0].start + w[0].span <= w[1].start + EPS));
        }
    }

    #[test]
    fn repeat_blocks_keep_source_rests_silent_and_stop_at_first_incomplete_entry() {
        let mut c = config();
        c.fugue_repeat_subject = true;
        c.pl = 7;
        c.render_length = 1; // 28 beats
        let rested = vec![
            Note::new(60, 1.0, 1.0, 90, 0, 0),
            Note::new(62, 3.0, 0.5, 80, 0, 0),
        ];
        let contours = Contours::from_config(&c);
        let p = plan_exposition(&rested, 4.0, &c, &contours).unwrap();
        // inv2x at beat 24 would end at 32, so it and the following shorter
        // cycle item are both omitted rather than skipping ahead.
        assert_eq!(p.entries.iter().map(|e| e.start).collect::<Vec<_>>(), vec![0.0, 4.0, 8.0, 12.0, 16.0]);

        for entry in &p.entries {
            for note in p.notes.iter().filter(|n| n.channel == entry.channel
                && n.start < entry.start + entry.span - EPS
                && n.start + n.duration > entry.start + EPS)
            {
                assert!(p.fixed.pitch_for(note).is_some(),
                    "generated note {:?} entered protected block {:?}", note, entry);
            }
        }
    }

    #[test]
    fn repeat_validation_is_strict_but_disabled_mode_ignores_the_cycle() {
        let mut c = config();
        c.fugue_transform_cycle.clear();
        let contours = Contours::from_config(&c);
        let legacy = plan_exposition(&subject(), 4.0, &c, &contours).unwrap();
        assert_eq!(legacy.entries.len(), 3);
        assert_eq!(legacy.fixed.len(), 9);

        c.fugue_repeat_subject = true;
        assert!(plan_exposition(&subject(), 4.0, &c, &contours).unwrap_err().0.contains("transformation step"));
        c.fugue_transform_cycle = vec![FugueTransformation::new(false, 0.75)];
        assert!(plan_exposition(&subject(), 4.0, &c, &contours).is_err());

        c.fugue_transform_cycle = vec![FugueTransformation::new(false, 2.0)];
        let long = vec![Note::new(60, 0.0, 16.5, 90, 0, 0)];
        c.pl = 20;
        assert!(plan_exposition(&long, 20.0, &c, &Contours::from_config(&c))
            .unwrap_err().0.contains("32 beats"));

        c.fugue_transform_cycle = vec![FugueTransformation::new(true, 1.0)];
        c.fugue_answer_interval = 0;
        c.fugue_bass_subject_shift = 0;
        c.pl = 4;
        c.render_length = 2;
        let wide = vec![
            Note::new(20, 0.0, 1.0, 90, 0, 0),
            Note::new(127, 1.0, 3.0, 90, 0, 0),
        ];
        assert!(plan_exposition(&wide, 4.0, &c, &Contours::from_config(&c))
            .unwrap_err().0.contains("outside 0..127"));
    }

    #[test]
    fn disabled_repeat_matches_the_original_exposition_planner_note_for_note() {
        fn original_plan(subject: &[Note], source_length: f64, c: &Config, contours: &Contours) -> Vec<Note> {
            let subject_end = subject.iter().map(|n| n.start + n.duration).fold(0.0, f64::max);
            let spacing = if c.fugue_entry_spacing > EPS { c.fugue_entry_spacing } else { source_length };
            let mut notes = Vec::new();
            for (channel, offset, transpose) in [(1, 0.0, 0), (0, spacing, c.fugue_answer_interval), (2, 2.0 * spacing, c.fugue_bass_subject_shift)] {
                for source in subject {
                    let mut note = *source;
                    note.pitch += transpose;
                    note.start += offset;
                    note.channel = channel;
                    note.muted = 0;
                    notes.push(note);
                }
                let base = c.start_notes[channel as usize];
                let entry_end = offset + subject_end;
                notes.extend(gen_voice(base, &c.voice_rhythm, &[0], channel, 1, c, contours)
                    .into_iter().filter_map(|mut n| {
                        let end = n.start + n.duration;
                        if end <= entry_end + EPS { return None; }
                        if n.start < entry_end {
                            n.start = entry_end;
                            n.duration = end - entry_end;
                        }
                        Some(n)
                    }));
            }
            notes.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap()
                .then_with(|| a.channel.cmp(&b.channel)));
            notes
        }

        let mut c = config();
        c.fugue_repeat_subject = false;
        c.voice_rhythm_patterns = vec![vec![0.5], vec![1.0], vec![2.0]];
        let contours = Contours::from_config(&c);
        crate::utils::SeededRng::set_seed(9123.0);
        let expected = original_plan(&subject(), 4.0, &c, &contours);
        crate::utils::SeededRng::set_seed(9123.0);
        let actual = plan_exposition(&subject(), 4.0, &c, &contours).unwrap().notes;
        assert_eq!(
            serde_json::to_value(actual).unwrap(),
            serde_json::to_value(expected).unwrap(),
        );
    }
}
