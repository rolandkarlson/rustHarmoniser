use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct Note {
    pub pitch: i32,
    pub start: f64,
    pub duration: f64,
    pub velocity: i32,
    pub muted: i32, // 0 or 1
    pub channel: i32,
    pub probability: i32,
}

impl Note {
    pub fn new(pitch: i32, start: f64, duration: f64, velocity: i32, muted: i32, channel: i32) -> Self {
        Self {
            pitch,
            start,
            duration,
            velocity,
            muted,
            channel,
            probability: 0,
        }
    }
}

#[derive(Clone)]
pub struct Config {
    pub schillinger_progression: bool,
    pub last_note_exist_in_voice: f64,
    pub same_direction: f64,
    pub consecutive_octav_fift: f64,
    pub no_crossing: f64,
    pub last_note_same: f64,
    pub mode: i32,
    pub interval_exists_in_harmony: f64,
    // New fields
    pub voice_rhythm: Vec<f64>,
    pub harmony_distance_balance: f64,
    pub lookahead_depth: i32,
    pub render_length: i32,
    pub rng_seed: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            schillinger_progression: true,
            last_note_exist_in_voice: 100.0,
            same_direction: 1.0,
            consecutive_octav_fift: 0.0,
            no_crossing: 100.0,
            last_note_same: 10.0,
            mode: 0,
            interval_exists_in_harmony: 1.0,
            voice_rhythm: vec![4.0],
            harmony_distance_balance: -0.1,
            lookahead_depth: 20,
            render_length: 1,
            rng_seed: 5443343433.0,
        }
    }
}
