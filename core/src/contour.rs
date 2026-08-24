//! Time-varying parameters as a first-class type.
//!
//! The Config wire format keeps its raw `Option<Vec<f64>>` / `Option<Vec<Vec<f64>>>`
//! contour fields (the GUI and archived snapshots depend on it); this module is
//! what the ENGINE consumes. `Contours::from_config` resolves every raw field
//! once, capturing the sampling resolution inside the value, so the rest of the
//! engine asks `contour.at(beats)` instead of repeating `(start / resolution)
//! .floor()` index arithmetic with ad-hoc emptiness checks at every site.
//!
//! Sampling semantics are the legacy ones, kept exactly: the beat position maps
//! to a step index at the contour's resolution, and the index wraps around the
//! contour's length (`get_wrapped`). An EMPTY contour never overrides a scalar —
//! `Contour::new` collapses it to `None` so callers fall back naturally.

use crate::model::Config;
use crate::utils::mod_shim;

/// One sampled curve over beat time at a fixed resolution (beats per step),
/// wrapping past its end. Guaranteed non-empty by construction.
#[derive(Debug, Clone)]
pub struct Contour {
    values: Vec<f64>,
    resolution: f64,
}

impl Contour {
    /// `None` when `values` is empty: an empty contour is "no contour", never
    /// a contour that overrides the scalar fall-back with nothing.
    pub fn new(values: Vec<f64>, resolution: f64) -> Option<Contour> {
        if values.is_empty() || resolution <= 0.0 {
            None
        } else {
            Some(Contour { values, resolution })
        }
    }

    fn from_field(field: Option<&Vec<f64>>, resolution: f64) -> Option<Contour> {
        field.and_then(|v| Contour::new(v.clone(), resolution))
    }

    /// Sample at a beat position (wrapped step index — legacy `get_wrapped`).
    pub fn at(&self, beats: f64) -> f64 {
        let idx = (beats / self.resolution).floor() as usize;
        self.values[mod_shim(idx as i32, self.values.len() as i32) as usize]
    }
}

/// A per-voice family of contours (outer index = voice). Rows may be
/// individually empty; an empty row samples to `None` so the caller's scalar
/// fall-back applies per voice, exactly like the legacy checks.
#[derive(Debug, Clone, Default)]
pub struct VoiceContours {
    rows: Vec<Option<Contour>>,
}

impl VoiceContours {
    pub fn new(rows: Vec<Vec<f64>>, resolution: f64) -> VoiceContours {
        VoiceContours {
            rows: rows.into_iter().map(|r| Contour::new(r, resolution)).collect(),
        }
    }

    fn from_field(field: Option<&Vec<Vec<f64>>>, resolution: f64) -> Option<VoiceContours> {
        field.map(|rows| VoiceContours::new(rows.clone(), resolution))
    }

    /// Sample voice `voice` at `beats`, WRAPPING the voice index around the
    /// number of rows (legacy `mod_shim` row lookup — harmonizer-side families).
    pub fn at(&self, voice: usize, beats: f64) -> Option<f64> {
        if self.rows.is_empty() {
            return None;
        }
        self.rows[mod_shim(voice as i32, self.rows.len() as i32) as usize]
            .as_ref()
            .map(|c| c.at(beats))
    }

    /// Sample voice `voice` at `beats` with a BOUNDS-CHECKED voice index
    /// (legacy `.get(voice)` row lookup — rhythm/Schillinger-side families).
    pub fn at_strict(&self, voice: usize, beats: f64) -> Option<f64> {
        self.rows.get(voice)?.as_ref().map(|c| c.at(beats))
    }
}

/// Every contour the engine can consult, resolved once from a `Config`.
/// `None` fields mean "use the scalar fall-back" at each site.
#[derive(Debug, Clone, Default)]
pub struct Contours {
    /// Per-voice pitch-offset targets for the voice-contour spring.
    /// NOTE: `Some` with no usable row still switches the spring's anchor from
    /// "previous pitch + offset" semantics — see `build_joint_voices`.
    pub voice: Option<VoiceContours>,
    /// Harmony/smoothness balance over time (overrides `harmony_distance_balance`).
    pub harmony_distance: Option<Contour>,
    /// H-Matrix style-row context over time (0..8, fractional LERPs rows).
    pub harmony_matrix: Option<Contour>,
    /// Per-voice melody-force weight over time (overrides scalar `melody_force`).
    pub melody_force: Option<VoiceContours>,
    /// Per-voice note-duration curve (overrides the `voice_rhythm` cycle).
    pub voice_rhythm: Option<VoiceContours>,
    /// Mode (0..6) per bar for the Schillinger scale realisation.
    pub mode: Option<Contour>,
    /// Key root (tonic pc 0..11) per bar — the modulation contour. Overrides
    /// the scalar `config.root` for the scale realisation and every root-keyed
    /// scoring term (see harmonizer::key_root_at).
    pub root: Option<Contour>,
    /// Per-voice chord-structure index into the built-in chord list.
    pub chord_structure: Option<VoiceContours>,
    /// Per-voice Schillinger expansion factor.
    pub schillinger_ex: Option<VoiceContours>,
}

impl Contours {
    /// Resolve every raw Config contour field at the config's resolution.
    pub fn from_config(config: &Config) -> Contours {
        let res = config.voice_contour_resolution;
        // voice_contour is Vec<Vec<i32>> on the wire; widen to f64 for sampling.
        let voice = config.voice_contour.as_ref().map(|rows| {
            VoiceContours::new(
                rows.iter()
                    .map(|r| r.iter().map(|&v| v as f64).collect())
                    .collect(),
                res,
            )
        });
        Contours {
            voice,
            harmony_distance: Contour::from_field(config.harmony_distance_contour.as_ref(), res),
            harmony_matrix: Contour::from_field(config.harmony_matrix_contour.as_ref(), res),
            melody_force: VoiceContours::from_field(config.melody_force_contour.as_ref(), res),
            voice_rhythm: VoiceContours::from_field(config.voice_rhythm_contour.as_ref(), res),
            mode: Contour::from_field(config.mode_contour.as_ref(), res),
            root: Contour::from_field(config.root_contour.as_ref(), res),
            chord_structure: VoiceContours::from_field(config.chord_structure_contour.as_ref(), res),
            schillinger_ex: VoiceContours::from_field(config.schillinger_ex_contour.as_ref(), res),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_contour_is_none() {
        assert!(Contour::new(vec![], 4.0).is_none());
        assert!(Contour::new(vec![1.0], 0.0).is_none());
    }

    #[test]
    fn sampling_floors_to_the_step_and_wraps() {
        let c = Contour::new(vec![10.0, 20.0, 30.0], 4.0).unwrap();
        assert_eq!(c.at(0.0), 10.0);
        assert_eq!(c.at(3.9), 10.0);
        assert_eq!(c.at(4.0), 20.0);
        assert_eq!(c.at(11.9), 30.0);
        assert_eq!(c.at(12.0), 10.0); // wraps
        assert_eq!(c.at(17.0), 20.0);
    }

    #[test]
    fn voice_rows_wrap_or_bounds_check_by_accessor() {
        let vc = VoiceContours::new(vec![vec![1.0], vec![], vec![3.0]], 4.0);
        // Wrapped lookup: voice 3 wraps to row 0; empty row 1 gives None.
        assert_eq!(vc.at(0, 0.0), Some(1.0));
        assert_eq!(vc.at(1, 0.0), None);
        assert_eq!(vc.at(3, 0.0), Some(1.0));
        // Strict lookup: out of range gives None instead of wrapping.
        assert_eq!(vc.at_strict(2, 0.0), Some(3.0));
        assert_eq!(vc.at_strict(3, 0.0), None);
    }

    #[test]
    fn from_config_collapses_empty_fields() {
        let mut cfg = Config::default();
        cfg.harmony_distance_contour = Some(vec![]);
        cfg.mode_contour = Some(vec![1.0]);
        let c = Contours::from_config(&cfg);
        assert!(c.harmony_distance.is_none());
        assert_eq!(c.mode.as_ref().map(|m| m.at(0.0)), Some(1.0));
    }
}
