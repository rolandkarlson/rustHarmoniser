//! Named score contributions: what the render archive and API expose so a
//! chosen chord can be inspected instead of reverse-engineered from one f64.
//!
//! The beam search itself runs untraced (the hot path scores millions of
//! candidate chords); after the winning progression is known, each chosen
//! chord is re-scored ONCE with a collecting sink, which yields the exact
//! same (hard, soft) numbers plus this per-term breakdown.

use serde::Serialize;

/// One named, weighted contribution to a soft score. Values are the amounts
/// actually added to the score (weights already applied), so they sum — up to
/// float re-association — to the reported soft score.
#[derive(Debug, Clone, Serialize)]
pub struct Term {
    pub name: &'static str,
    pub value: f64,
}

/// One category of hard-constraint violation and how often it fired.
#[derive(Debug, Clone, Serialize)]
pub struct HardViolation {
    pub name: &'static str,
    pub count: u32,
}

/// The per-voice melodic terms behind one chosen pitch.
#[derive(Debug, Clone, Serialize)]
pub struct VoiceBreakdown {
    pub channel: i32,
    pub pitch: i32,
    pub previous_pitch: Option<i32>,
    /// The voice the leader-selection picked to carry the melodic motion this
    /// chord (its repeat penalties applied instead of the hold bonus).
    pub is_leader: bool,
    pub terms: Vec<Term>,
}

/// The full scoring story of one chosen chord (one rhythmic group).
#[derive(Debug, Clone, Serialize)]
pub struct GroupBreakdown {
    pub start: f64,
    pub bar: i32,
    /// Chord root pitch class the root-aware terms used (None on the simple
    /// candidate path, where no root is defined).
    pub root_pc: Option<i32>,
    /// The score this group contributed to the beam total:
    /// `soft_score - hard_violation_count * VIOLATION_PENALTY`.
    pub score: f64,
    pub soft_score: f64,
    pub hard_violation_count: u32,
    pub hard_violations: Vec<HardViolation>,
    /// Chord-level terms (harmony aggregation, root terms, motion penalties).
    pub chord_terms: Vec<Term>,
    pub voices: Vec<VoiceBreakdown>,
    /// False when the group passed through unscored (no candidates were
    /// generated for it) — terms are empty in that case.
    pub scored: bool,
}

/// Collects term callbacks during one traced `eval`. Duplicate names are
/// summed (e.g. per-pair parallel-motion penalties).
#[derive(Debug, Default)]
pub struct TraceCollector {
    pub chord_terms: Vec<Term>,
    pub voice_terms: Vec<Vec<Term>>,
    pub hard: Vec<HardViolation>,
    pub leader: Option<usize>,
}

impl TraceCollector {
    pub fn with_voices(n: usize) -> Self {
        TraceCollector {
            voice_terms: (0..n).map(|_| Vec::new()).collect(),
            ..Default::default()
        }
    }

    fn merge(list: &mut Vec<Term>, name: &'static str, value: f64) {
        if value == 0.0 {
            return;
        }
        if let Some(t) = list.iter_mut().find(|t| t.name == name) {
            t.value += value;
        } else {
            list.push(Term { name, value });
        }
    }

    pub fn term(&mut self, name: &'static str, value: f64) {
        Self::merge(&mut self.chord_terms, name, value);
    }

    pub fn voice_term(&mut self, voice: usize, name: &'static str, value: f64) {
        if voice < self.voice_terms.len() {
            Self::merge(&mut self.voice_terms[voice], name, value);
        }
    }

    pub fn hard(&mut self, name: &'static str, count: u32) {
        if count == 0 {
            return;
        }
        if let Some(h) = self.hard.iter_mut().find(|h| h.name == name) {
            h.count += count;
        } else {
            self.hard.push(HardViolation { name, count });
        }
    }
}
