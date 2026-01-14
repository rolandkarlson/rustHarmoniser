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

#[derive(Clone, Copy)]
pub struct Config {
    pub last_note_exist_in_voice: f64,
    pub same_direction: f64,
    pub consecutive_octav_fift: f64,
    pub no_crossing: f64,
    pub last_note_same: f64,
    pub mode: i32,
    pub interval_exists_in_harmony: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            last_note_exist_in_voice: 100.0,
            same_direction: 0.0,
            consecutive_octav_fift: 0.0,
            no_crossing: 0.0,
            last_note_same: 100.0,
            mode: 0,
            interval_exists_in_harmony: 0.0,
        }
    }
}
