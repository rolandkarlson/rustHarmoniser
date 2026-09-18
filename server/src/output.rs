//! Deterministic output planning and render persistence.
//!
//! The core engine stays pure. This module converts its immutable notes into
//! validated Ableton clip plans, writes `output.json`, and archives enough
//! information to reproduce both the music and its output transformations.

use rustnote_core::model::{Config, Note};
use rustnote_core::render::{try_render, Leading, RenderResult};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{self, Write};
use std::time::Instant;

const MAX_ROUTES: usize = 64;
const MAX_ROUTE_EVENTS: usize = 100_000;
const MAX_BATCH_EVENTS: usize = 250_000;
const MAX_PATTERN_STEPS: usize = 256;
const MAX_CLIP_LENGTH: f64 = 1_000_000.0;
const EPS: f64 = 0.000_001;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OutputMode {
    Notes,
    TranceArp,
    ContourArp,
    VoiceArp,
}
impl Default for OutputMode {
    fn default() -> Self {
        Self::Notes
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct OutputRoute {
    pub enabled: bool,
    pub track_index: i32,
    pub clip_index: i32,
    pub name: String,
    pub mode: OutputMode,
    pub voices: Vec<i32>,
    pub transpose: i32,
    pub time_scale: f64,
    pub merge_adjacent: bool,
    pub humanize_beats: f64,
    pub fold_pitch: Option<i32>,
    pub arp_durations: Vec<f64>,
    pub voice_pattern: Vec<i32>,
    pub initial_pitch: i32,
    pub lead_voice: i32,
    pub bass_voice: i32,
    pub lead_octave: i32,
}

impl Default for OutputRoute {
    fn default() -> Self {
        Self {
            enabled: true,
            track_index: 4,
            clip_index: 2,
            name: "Harmonizer".into(),
            mode: OutputMode::Notes,
            voices: vec![0, 1, 2, 3, 4],
            transpose: 0,
            time_scale: 1.0,
            merge_adjacent: false,
            humanize_beats: 0.0,
            fold_pitch: None,
            arp_durations: vec![0.75, 0.75, 0.75, 0.75, 0.5, 0.5],
            voice_pattern: vec![4, 3, 4],
            initial_pitch: 60,
            lead_voice: 0,
            bass_voice: 4,
            lead_octave: 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct OutputSettings {
    pub enabled: bool,
    pub routes: Vec<OutputRoute>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClipNote {
    pub pitch: i32,
    pub start_time: f64,
    pub duration: f64,
    pub velocity: i32,
    pub mute: bool,
    pub probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClipPlan {
    pub track_index: i32,
    pub clip_index: i32,
    pub name: String,
    pub length: f64,
    pub notes: Vec<ClipNote>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClipSummary {
    pub track_index: i32,
    pub clip_index: i32,
    pub name: String,
    pub length: f64,
    pub note_count: usize,
}

impl From<&ClipPlan> for ClipSummary {
    fn from(c: &ClipPlan) -> Self {
        Self {
            track_index: c.track_index,
            clip_index: c.clip_index,
            name: c.name.clone(),
            length: c.length,
            note_count: c.notes.len(),
        }
    }
}

pub struct CompletedRender {
    pub message: String,
    pub result: RenderResult,
    pub output: OutputSettings,
    pub clips: Result<Vec<ClipPlan>, String>,
}

#[derive(Clone, Copy)]
struct WorkNote {
    pitch: i32,
    start: f64,
    duration: f64,
    velocity: i32,
    muted: i32,
    channel: i32,
    probability: i32,
}
impl From<&Note> for WorkNote {
    fn from(n: &Note) -> Self {
        Self {
            pitch: n.pitch,
            start: n.start,
            duration: n.duration,
            velocity: n.velocity,
            muted: n.muted,
            channel: n.channel,
            probability: n.probability,
        }
    }
}

struct OutputRng(u64);
impl OutputRng {
    fn new(seed: f64, route: usize) -> Self {
        let mut bits = seed.to_bits() ^ (route as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        if bits == 0 {
            bits = 0xA076_1D64_78BD_642F;
        }
        Self(bits)
    }
    fn next(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / ((1u64 << 53) as f64)
    }
    fn index(&mut self, len: usize) -> usize {
        ((self.next() * len as f64) as usize).min(len - 1)
    }
    fn gaussianish(&mut self) -> f64 {
        (self.next() + self.next() + self.next()) / 1.5 - 1.0
    }
}

pub fn default_output_settings(config: &Config) -> OutputSettings {
    let voice_count = if config.fugue_mode {
        config.fugue_voice_count.clamp(3, 8)
    } else {
        5
    };
    if config.fugue_mode {
        let routes = (0..voice_count)
            .map(|voice| {
                let mut r = OutputRoute::default();
                r.track_index = config
                    .fugue_output_tracks
                    .get(voice as usize)
                    .copied()
                    .unwrap_or(4 + voice);
                r.clip_index = config.fugue_output_clip;
                r.name = format!("Fugue Voice {}", voice + 1);
                r.voices = vec![voice];
                r
            })
            .collect();
        return OutputSettings {
            enabled: false,
            routes,
        };
    }
    let voices: Vec<i32> = (0..voice_count).collect();
    let mut mixed = OutputRoute::default();
    mixed.track_index = 0;
    mixed.name = "Harmonizer Mix".into();
    mixed.voices = voices;
    mixed.humanize_beats = 0.4;
    let mut trance = OutputRoute {
        track_index: 1,
        name: "Trance Arp".into(),
        mode: OutputMode::TranceArp,
        ..mixed.clone()
    };
    trance.humanize_beats = 0.0;
    let contour = OutputRoute {
        track_index: 2,
        name: "Contour Arp".into(),
        mode: OutputMode::ContourArp,
        initial_pitch: 60,
        ..trance.clone()
    };
    let folded = OutputRoute {
        track_index: 3,
        name: "Folded Chords".into(),
        mode: OutputMode::Notes,
        fold_pitch: Some(36),
        ..trance.clone()
    };
    let mut routes = vec![mixed, trance, contour, folded];
    for voice in 0..voice_count {
        routes.push(OutputRoute {
            track_index: 4 + voice,
            name: format!("Voice {}", voice + 1),
            voices: vec![voice],
            merge_adjacent: true,
            ..OutputRoute::default()
        });
    }
    OutputSettings {
        enabled: false,
        routes,
    }
}

/// Render and persist without touching Ableton. Invalid disabled output is
/// retained as a preview error so routing edits never prevent generation.
pub fn run_render(
    config: &Config,
    leading: Option<&Leading>,
    output: Option<OutputSettings>,
) -> io::Result<CompletedRender> {
    let start_time = Instant::now();
    let result = try_render(config, leading, None)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let output = output.unwrap_or_else(|| default_output_settings(config));
    let clips = plan_clips(&result.notes, config, &output).map_err(|e| e.to_string());
    let mut file = File::create("output.json")?;
    file.write_all(serde_json::to_string_pretty(&result.notes)?.as_bytes())?;
    let archived = match archive_render(
        "render",
        config,
        leading,
        &result,
        &output,
        clips.as_ref().ok().map(Vec::as_slice),
        clips.as_ref().err().map(String::as_str),
    ) {
        Ok(path) => format!(" — archived to {path}"),
        Err(e) => {
            eprintln!("render archive failed: {e}");
            String::new()
        }
    };
    let message = format!(
        "Generated {} notes in {:?}{}",
        result.notes.len(),
        start_time.elapsed(),
        archived
    );
    Ok(CompletedRender {
        message,
        result,
        output,
        clips,
    })
}

pub fn archive_render(
    dir: &str,
    config: &Config,
    leading: Option<&Leading>,
    result: &RenderResult,
    output: &OutputSettings,
    clips: Option<&[ClipPlan]>,
    output_error: Option<&str>,
) -> io::Result<String> {
    std::fs::create_dir_all(dir)?;
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let path = format!("{dir}/render_{ts}.json");
    let doc = serde_json::json!({
        "timestamp_unix_ms": ts,
        "input": { "config": config, "leading": leading.map(|l| serde_json::json!({"notes": l.notes, "clip_length": l.clip_length})), "output_settings": output },
        "output": { "note_count": result.notes.len(), "notes": result.notes, "breakdown": result.breakdown, "clips": clips, "clip_plan_error": output_error },
    });
    let mut file = File::create(&path)?;
    file.write_all(serde_json::to_string_pretty(&doc)?.as_bytes())?;
    Ok(path)
}

pub fn validate_output_settings(
    config: &Config,
    output: &OutputSettings,
    require_routes: bool,
) -> io::Result<()> {
    let voice_count = if config.fugue_mode {
        usize::try_from(config.fugue_voice_count).unwrap_or(0)
    } else {
        5
    };
    if config.fugue_mode && !(3..=8).contains(&voice_count) {
        return invalid("fugue voice count must be between 3 and 8");
    }
    if output.routes.len() > MAX_ROUTES {
        return invalid("output supports at most 64 routes");
    }
    let enabled: Vec<&OutputRoute> = output.routes.iter().filter(|r| r.enabled).collect();
    if require_routes && enabled.is_empty() {
        return invalid("enable at least one output route");
    }
    let mut destinations = HashSet::new();
    for (index, route) in enabled.into_iter().enumerate() {
        let label = format!("output route {}", index + 1);
        if route.track_index < 0 || route.clip_index < 0 {
            return invalid(format!("{label} needs non-negative track and clip indices"));
        }
        if !destinations.insert((route.track_index, route.clip_index)) {
            return invalid(format!(
                "duplicate output destination at track {} clip {}",
                route.track_index, route.clip_index
            ));
        }
        if config.use_leading_voice
            && route.track_index == config.leading_voice_track
            && route.clip_index == config.leading_voice_clip
        {
            return invalid("an output destination cannot overwrite the imported Lead Clip source");
        }
        if route.name.chars().count() > 128 {
            return invalid(format!("{label} name is longer than 128 characters"));
        }
        if !route.time_scale.is_finite() || !(0.01..=64.0).contains(&route.time_scale) {
            return invalid(format!(
                "{label} time scale must be finite and between 0.01 and 64"
            ));
        }
        if !route.humanize_beats.is_finite() || !(0.0..=4.0).contains(&route.humanize_beats) {
            return invalid(format!(
                "{label} humanize amount must be finite and between 0 and 4 beats"
            ));
        }
        if !(-127..=127).contains(&route.transpose) {
            return invalid(format!("{label} transpose must be between -127 and 127"));
        }
        if let Some(fold) = route.fold_pitch {
            if !(0..=127).contains(&fold) {
                return invalid(format!("{label} fold pitch must be MIDI 0..127"));
            }
        }
        if !(0..=127).contains(&route.initial_pitch) {
            return invalid(format!("{label} initial pitch must be MIDI 0..127"));
        }
        if !(-8..=8).contains(&route.lead_octave) {
            return invalid(format!("{label} lead octave must be between -8 and 8"));
        }
        if route.voices.is_empty() {
            return invalid(format!("{label} needs at least one source voice"));
        }
        let mut voices = HashSet::new();
        for &voice in &route.voices {
            if voice < 0 || voice as usize >= voice_count || !voices.insert(voice) {
                return invalid(format!(
                    "{label} has an invalid or duplicate source voice {voice}"
                ));
            }
        }
        if matches!(route.mode, OutputMode::TranceArp)
            && (route.lead_voice < 0
                || route.lead_voice as usize >= voice_count
                || route.bass_voice < 0
                || route.bass_voice as usize >= voice_count)
        {
            return invalid(format!("{label} lead and bass voices must exist"));
        }
        let is_arp = !matches!(route.mode, OutputMode::Notes);
        if is_arp && route.arp_durations.len() > MAX_PATTERN_STEPS {
            return invalid(format!("{label} duration pattern is too long"));
        }
        if matches!(route.mode, OutputMode::VoiceArp)
            && route.voice_pattern.len() > MAX_PATTERN_STEPS
        {
            return invalid(format!("{label} voice pattern is too long"));
        }
        if is_arp {
            for &duration in &route.arp_durations {
                if !duration.is_finite() || !(0.01..=64.0).contains(&duration) {
                    return invalid(format!(
                        "{label} arp durations must be finite and between 0.01 and 64 beats"
                    ));
                }
            }
        }
        if matches!(route.mode, OutputMode::VoiceArp) {
            for &voice in &route.voice_pattern {
                if voice < 0 || voice as usize >= voice_count {
                    return invalid(format!(
                        "{label} voice pattern references missing voice {voice}"
                    ));
                }
            }
        }
        if matches!(route.mode, OutputMode::VoiceArp)
            && (route.arp_durations.is_empty() || route.voice_pattern.is_empty())
        {
            return invalid(format!(
                "{label} voice arp needs duration and voice patterns"
            ));
        }
    }
    Ok(())
}

pub fn plan_clips(
    notes: &[Note],
    config: &Config,
    output: &OutputSettings,
) -> io::Result<Vec<ClipPlan>> {
    validate_output_settings(config, output, output.enabled)?;
    for note in notes {
        if !note.start.is_finite()
            || !note.duration.is_finite()
            || note.start < 0.0
            || note.duration <= 0.0
            || !(0..=127).contains(&note.pitch)
            || !(0..=127).contains(&note.velocity)
            || !matches!(note.muted, 0 | 1)
            || !(0..=100).contains(&note.probability)
        {
            return invalid("engine output contains an invalid MIDI note");
        }
    }
    let base_i64 = i64::from(config.pl)
        .checked_mul(4)
        .and_then(|v| v.checked_mul(i64::from(config.render_length)))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "render length overflowed"))?;
    if base_i64 <= 0 {
        return invalid("render length must be positive");
    }
    let base_length = base_i64 as f64;
    let mut plans = Vec::new();
    let mut total_events = 0usize;
    for (route_index, route) in output.routes.iter().enumerate().filter(|(_, r)| r.enabled) {
        let length = base_length * route.time_scale;
        if !length.is_finite() || length > MAX_CLIP_LENGTH {
            return invalid(format!(
                "output route {} clip length is too large",
                route_index + 1
            ));
        }
        let selected: Vec<WorkNote> = notes
            .iter()
            .filter(|n| route.voices.contains(&n.channel))
            .map(WorkNote::from)
            .collect();
        let mut rng = OutputRng::new(config.rng_seed, route_index);
        let mut transformed = match route.mode {
            OutputMode::Notes => selected,
            OutputMode::VoiceArp => voice_arp(&selected, route, base_length)?,
            OutputMode::ContourArp => contour_arp(&selected, route, base_length)?,
            OutputMode::TranceArp => trance_arp(&selected, route, base_length, &mut rng)?,
        };
        apply_pitch(&mut transformed, route)?;
        for n in &mut transformed {
            n.start *= route.time_scale;
            n.duration *= route.time_scale;
        }
        humanize(&mut transformed, route.humanize_beats, &mut rng);
        transformed.retain_mut(|n| {
            if n.start < 0.0 {
                n.start = 0.0;
            }
            if n.start >= length - EPS {
                return false;
            }
            n.duration = n.duration.min(length - n.start);
            n.duration > EPS
        });
        if route.merge_adjacent {
            transformed = merge_adjacent(transformed);
        }
        transformed.sort_by(|a, b| {
            a.start
                .total_cmp(&b.start)
                .then(a.pitch.cmp(&b.pitch))
                .then(a.channel.cmp(&b.channel))
        });
        if transformed.len() > MAX_ROUTE_EVENTS {
            return invalid(format!(
                "output route {} exceeds {MAX_ROUTE_EVENTS} notes",
                route_index + 1
            ));
        }
        total_events = total_events.checked_add(transformed.len()).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "output event count overflowed")
        })?;
        if total_events > MAX_BATCH_EVENTS {
            return invalid(format!("output batch exceeds {MAX_BATCH_EVENTS} notes"));
        }
        plans.push(ClipPlan {
            track_index: route.track_index,
            clip_index: route.clip_index,
            name: route.name.clone(),
            length,
            notes: transformed
                .into_iter()
                .map(to_clip_note)
                .collect::<io::Result<_>>()?,
        });
    }
    Ok(plans)
}

fn invalid<T>(message: impl Into<String>) -> io::Result<T> {
    Err(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}
fn active_at(notes: &[WorkNote], time: f64) -> Vec<WorkNote> {
    notes
        .iter()
        .copied()
        .filter(|n| n.start <= time + 0.001 && n.start + n.duration > time + 0.001)
        .collect()
}
fn duration_pattern(r: &OutputRoute) -> &[f64] {
    if r.arp_durations.is_empty() {
        &[0.75, 0.75, 0.75, 0.75, 0.5, 0.5]
    } else {
        &r.arp_durations
    }
}

fn voice_arp(notes: &[WorkNote], r: &OutputRoute, end: f64) -> io::Result<Vec<WorkNote>> {
    let mut out = Vec::new();
    let mut t = 0.0;
    let mut step = 0;
    while t < end - EPS {
        if step >= MAX_ROUTE_EVENTS {
            return invalid("voice arp exceeds the step limit");
        }
        let d = duration_pattern(r)[step % duration_pattern(r).len()].min(end - t);
        let voice = r.voice_pattern[step % r.voice_pattern.len()];
        if let Some(mut n) = active_at(notes, t).into_iter().find(|n| n.channel == voice) {
            n.start = t;
            n.duration = d;
            out.push(n)
        }
        t += d;
        step += 1;
    }
    Ok(out)
}

fn contour_arp(notes: &[WorkNote], r: &OutputRoute, end: f64) -> io::Result<Vec<WorkNote>> {
    let mut out = Vec::new();
    let mut t = 0.0;
    let mut step = 0;
    let mut previous = r.initial_pitch;
    while t < end - EPS {
        if step >= MAX_ROUTE_EVENTS {
            return invalid("contour arp exceeds the step limit");
        }
        let d = duration_pattern(r)[step % duration_pattern(r).len()].min(end - t);
        let active = active_at(notes, t);
        let active_len = active.len();
        if !active.is_empty() {
            let curve = ((t / 32.0 * std::f64::consts::PI).sin()
                + (t / 8.0 * 2.0 * std::f64::consts::PI).sin())
                / 1.5;
            let target = r.initial_pitch as f64 + curve * 12.0;
            let mut best = None;
            for n in active {
                let pc = n.pitch.rem_euclid(12);
                let base = previous - previous.rem_euclid(12) + pc;
                for oct in -3..=3 {
                    let p = base + oct * 12;
                    if p < r.initial_pitch - 12
                        || p > r.initial_pitch + 12
                        || (p == previous && active_len > 1)
                    {
                        continue;
                    }
                    let dist = (p as f64 - target).abs();
                    if best.map_or(true, |(b, _, _)| dist < b) {
                        best = Some((dist, p, n))
                    }
                }
            }
            if let Some((_, p, mut n)) = best {
                n.pitch = p;
                n.start = t;
                n.duration = d;
                out.push(n);
                previous = p
            }
        }
        t += d;
        step += 1;
    }
    Ok(out)
}

fn trance_arp(
    notes: &[WorkNote],
    r: &OutputRoute,
    end: f64,
    rng: &mut OutputRng,
) -> io::Result<Vec<WorkNote>> {
    let generated;
    let pattern = if r.arp_durations.is_empty() {
        generated = trance_pattern(rng);
        &generated[..]
    } else {
        &r.arp_durations[..]
    };
    let mut out = Vec::new();
    let mut t = 0.0;
    let mut step = 0;
    let mut previous = None;
    while t < end - EPS {
        if step >= MAX_ROUTE_EVENTS || out.len().saturating_add(2) > MAX_ROUTE_EVENTS {
            return invalid("trance arp exceeds the step limit");
        }
        let d = pattern[step % pattern.len()].min(end - t);
        let active = active_at(notes, t);
        let bass = active.iter().copied().find(|n| n.channel == r.bass_voice);
        let bass_pitch = bass.map(|n| n.pitch);
        if let Some(mut n) = bass {
            n.start = t;
            n.duration = d * 0.9;
            n.velocity = (100.0 + rng.next() * 20.0).round() as i32;
            out.push(n)
        }
        let on = t.rem_euclid(1.0) < 0.001;
        let mut pcs: Vec<i32> = active
            .iter()
            .filter(|n| n.channel == r.lead_voice)
            .map(|n| n.pitch.rem_euclid(12))
            .collect();
        pcs.sort_unstable();
        pcs.dedup();
        if pcs.is_empty() {
            pcs = active.iter().map(|n| n.pitch.rem_euclid(12)).collect();
            pcs.sort_unstable();
            pcs.dedup()
        }
        if !pcs.is_empty() {
            let low = bass_pitch.unwrap_or(60) + r.lead_octave * 12;
            let center = previous.unwrap_or(low + 7);
            let mut best = None;
            for pc in pcs {
                let mut base = low - low.rem_euclid(12) + pc;
                if base < low {
                    base += 12
                }
                for oct in 0..2 {
                    let p = base + oct * 12;
                    if p > low + 19 {
                        break;
                    }
                    let dis = bass_pitch.map_or(false, |b| {
                        matches!((p - b).rem_euclid(12), 1 | 2 | 6 | 10 | 11)
                    });
                    let score = (p - center).abs() as f64
                        + if dis {
                            if on {
                                100.0
                            } else {
                                6.0
                            }
                        } else {
                            0.0
                        }
                        + if Some(p) == previous { 1.5 } else { 0.0 };
                    if best.map_or(true, |(s, _)| score < s) {
                        best = Some((score, p))
                    }
                }
            }
            if let Some((_, p)) = best {
                out.push(WorkNote {
                    pitch: p,
                    start: t,
                    duration: d * 0.9,
                    velocity: ((if on { 110.0 } else { 92.0 }) + rng.next() * 15.0).round() as i32,
                    muted: 0,
                    channel: r.lead_voice,
                    probability: 0,
                });
                previous = Some(p)
            }
        }
        t += d;
        step += 1;
    }
    Ok(out)
}

fn trance_pattern(rng: &mut OutputRng) -> Vec<f64> {
    const C: &[&[f64]] = &[
        &[1.0],
        &[0.5, 0.5],
        &[0.75, 0.25],
        &[0.25, 0.75],
        &[0.5, 0.25, 0.25],
        &[0.25, 0.25, 0.5],
        &[0.25, 0.25, 0.25, 0.25],
        &[0.75, 0.75, 0.5],
        &[0.5, 0.75, 0.75],
        &[0.75, 0.5, 0.75],
        &[1.5, 0.5],
        &[0.75, 0.75, 0.25, 0.25],
    ];
    let main = C[rng.index(C.len())];
    let variation = C[rng.index(C.len())];
    let mut out = Vec::new();
    let mut remaining = 4.0;
    while remaining > EPS {
        let mut cell = if remaining <= 2.0 + EPS && rng.next() < 0.5 {
            variation
        } else {
            main
        };
        if cell.iter().sum::<f64>() > remaining + EPS {
            let fits: Vec<_> = C
                .iter()
                .copied()
                .filter(|c| c.iter().sum::<f64>() <= remaining + EPS)
                .collect();
            cell = fits[rng.index(fits.len())]
        }
        for &d in cell {
            let used = d.min(remaining);
            out.push(used);
            remaining -= used;
            if remaining <= EPS {
                break;
            }
        }
    }
    out
}

fn apply_pitch(notes: &mut [WorkNote], r: &OutputRoute) -> io::Result<()> {
    for n in notes {
        let p = if let Some(base) = r.fold_pitch {
            base.checked_add(n.pitch.rem_euclid(12))
        } else {
            Some(n.pitch)
        }
        .and_then(|p| p.checked_add(r.transpose))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "output pitch overflowed"))?;
        if !(0..=127).contains(&p) {
            return invalid(format!("output pitch {p} is outside MIDI 0..127"));
        }
        n.pitch = p
    }
    Ok(())
}

fn humanize(notes: &mut [WorkNote], amount: f64, rng: &mut OutputRng) {
    if amount == 0.0 {
        return;
    }
    let mut groups: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (i, n) in notes.iter().enumerate() {
        groups
            .entry((n.start * 1000.0).round() as i64)
            .or_default()
            .push(i)
    }
    let mut drift = 0.0_f64;
    for indices in groups.values_mut() {
        indices.sort_by_key(|&i| notes[i].pitch);
        let start = notes[indices[0]].start;
        drift = (drift + rng.gaussianish() * 0.3).clamp(-1.0, 1.0);
        let tight = if start.rem_euclid(4.0) < 0.001 {
            0.35
        } else if start.rem_euclid(1.0) < 0.001 {
            0.7
        } else {
            1.0
        };
        let offset = (drift * 0.6 + rng.gaussianish() * 0.4) * amount * tight;
        for (order, &i) in indices.iter().enumerate() {
            notes[i].start = (notes[i].start
                + offset
                + order as f64 * 0.02 * amount
                + rng.gaussianish() * 0.06 * amount)
                .max(0.0)
        }
    }
}

fn merge_adjacent(mut notes: Vec<WorkNote>) -> Vec<WorkNote> {
    notes.sort_by(|a, b| {
        a.channel
            .cmp(&b.channel)
            .then(a.pitch.cmp(&b.pitch))
            .then(a.start.total_cmp(&b.start))
    });
    let mut out: Vec<WorkNote> = Vec::new();
    for n in notes {
        if let Some(last) = out.last_mut() {
            if last.channel == n.channel
                && last.pitch == n.pitch
                && (last.start + last.duration - n.start).abs() <= EPS
                && last.muted == n.muted
            {
                last.duration += n.duration;
                continue;
            }
        }
        out.push(n)
    }
    out
}
fn to_clip_note(n: WorkNote) -> io::Result<ClipNote> {
    if !n.start.is_finite()
        || !n.duration.is_finite()
        || n.start < 0.0
        || n.duration <= 0.0
        || !(0..=127).contains(&n.pitch)
        || !(0..=127).contains(&n.velocity)
    {
        return invalid("output transformation produced an invalid MIDI note");
    }
    let probability = if n.probability <= 0 {
        1.0
    } else {
        (n.probability as f64 / 100.0).clamp(0.0, 1.0)
    };
    Ok(ClipNote {
        pitch: n.pitch,
        start_time: n.start,
        duration: n.duration,
        velocity: n.velocity,
        mute: n.muted != 0,
        probability,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> Config {
        let mut c = Config::default();
        c.pl = 1;
        c.render_length = 1;
        c.rng_seed = 42.0;
        c
    }
    fn notes() -> Vec<Note> {
        vec![
            Note::new(60, 0.0, 1.0, 90, 0, 0),
            Note::new(60, 1.0, 1.0, 80, 0, 0),
            Note::new(48, 0.0, 4.0, 100, 0, 4),
            Note::new(64, 2.0, 2.0, 88, 0, 1),
        ]
    }
    fn settings(r: OutputRoute) -> OutputSettings {
        OutputSettings {
            enabled: true,
            routes: vec![r],
        }
    }
    #[test]
    fn raw_fugue_defaults_preserve_repeated_attacks_and_full_render_length() {
        let mut c = config();
        c.fugue_mode = true;
        c.fugue_voice_count = 3;
        c.fugue_output_tracks = vec![7, 8, 9];
        let o = default_output_settings(&c);
        let p = plan_clips(&notes(), &c, &o).unwrap();
        assert_eq!(p.len(), 3);
        assert_eq!(p[0].length, 4.0);
        assert_eq!(p[0].notes.len(), 2);
        assert_eq!(p[0].notes[0].start_time, 0.0);
        assert_eq!(p[0].notes[1].start_time, 1.0);
        assert!(o
            .routes
            .iter()
            .all(|r| !r.merge_adjacent && r.humanize_beats == 0.0));
    }
    #[test]
    fn transforms_are_deterministic_local_and_do_not_mutate_engine_notes() {
        let c = config();
        let original = notes();
        let before = serde_json::to_string(&original).unwrap();
        let r = OutputRoute {
            track_index: 2,
            voices: vec![0, 1, 4],
            humanize_beats: 0.2,
            time_scale: 2.0,
            ..OutputRoute::default()
        };
        let a = plan_clips(&original, &c, &settings(r.clone())).unwrap();
        let b = plan_clips(&original, &c, &settings(r)).unwrap();
        assert_eq!(a, b);
        assert_eq!(serde_json::to_string(&original).unwrap(), before);
        assert_eq!(a[0].length, 8.0);
    }
    #[test]
    fn merge_fold_and_probability_have_explicit_rust_semantics() {
        let c = config();
        let r = OutputRoute {
            voices: vec![0],
            fold_pitch: Some(36),
            merge_adjacent: true,
            ..OutputRoute::default()
        };
        let p = plan_clips(&notes(), &c, &settings(r)).unwrap();
        assert_eq!(p[0].notes.len(), 1);
        assert_eq!(p[0].notes[0].pitch, 36);
        assert_eq!(p[0].notes[0].duration, 2.0);
        assert_eq!(p[0].notes[0].probability, 1.0);
    }
    #[test]
    fn every_arp_mode_is_bounded() {
        let c = config();
        for mode in [
            OutputMode::VoiceArp,
            OutputMode::ContourArp,
            OutputMode::TranceArp,
        ] {
            let r = OutputRoute {
                mode,
                voices: vec![0, 1, 4],
                voice_pattern: vec![4, 0],
                arp_durations: vec![0.5],
                ..OutputRoute::default()
            };
            let p = plan_clips(&notes(), &c, &settings(r)).unwrap();
            assert!(!p[0].notes.is_empty());
            assert!(p[0].notes.len() <= 16);
            assert!(p[0]
                .notes
                .iter()
                .all(|n| n.start_time + n.duration <= 4.0 + EPS));
        }
    }
    #[test]
    fn validation_rejects_bad_routes_before_planning() {
        let c = config();
        let mut o = settings(OutputRoute::default());
        o.routes.push(o.routes[0].clone());
        assert!(validate_output_settings(&c, &o, true)
            .unwrap_err()
            .to_string()
            .contains("duplicate"));
        o.routes.pop();
        o.routes[0].time_scale = f64::NAN;
        assert!(validate_output_settings(&c, &o, true).is_err());
        o.routes[0].time_scale = 1.0;
        o.routes[0].voices = vec![9];
        assert!(validate_output_settings(&c, &o, true).is_err());
    }

    #[test]
    fn invalid_engine_notes_and_silent_arp_work_are_bounded() {
        let c = config();
        let mut invalid_note = notes();
        invalid_note[0].duration = f64::NAN;
        assert!(plan_clips(&invalid_note, &c, &settings(OutputRoute::default())).is_err());

        let mut long = c;
        long.pl = 300;
        let route = OutputRoute {
            mode: OutputMode::VoiceArp,
            voices: vec![0],
            voice_pattern: vec![0],
            arp_durations: vec![0.01],
            ..OutputRoute::default()
        };
        assert!(plan_clips(&[], &long, &settings(route))
            .unwrap_err()
            .to_string()
            .contains("step limit"));
    }
    #[test]
    fn source_collision_applies_to_every_mode() {
        let mut c = config();
        c.use_leading_voice = true;
        c.leading_voice_track = 4;
        c.leading_voice_clip = 2;
        for mode in [
            OutputMode::Notes,
            OutputMode::TranceArp,
            OutputMode::ContourArp,
            OutputMode::VoiceArp,
        ] {
            let r = OutputRoute {
                mode,
                ..OutputRoute::default()
            };
            assert!(validate_output_settings(&c, &settings(r), true)
                .unwrap_err()
                .to_string()
                .contains("Lead Clip"));
        }
    }
    #[test]
    fn archive_contains_output_settings_and_derived_clips() {
        let dir = std::env::temp_dir().join(format!(
            "rustnote_archive_output_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let c = config();
        let o = settings(OutputRoute::default());
        let clips = plan_clips(&notes(), &c, &o).unwrap();
        let result = RenderResult {
            notes: notes(),
            breakdown: vec![],
            schillinger_notes: vec![],
        };
        let path = archive_render(
            dir.to_str().unwrap(),
            &c,
            None,
            &result,
            &o,
            Some(&clips),
            None,
        )
        .unwrap();
        let doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(doc["input"]["output_settings"]["enabled"], true);
        assert_eq!(
            doc["output"]["clips"][0]["notes"].as_array().unwrap().len(),
            clips[0].notes.len()
        );
        let _ = std::fs::remove_dir_all(dir);
    }
}
