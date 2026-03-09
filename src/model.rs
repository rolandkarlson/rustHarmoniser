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
    pub schillinger_sequence: Vec<i32>,
    pub pl: i32,
    pub harmony_distance_balance: f64,
    pub lookahead_depth: i32,
    pub render_length: i32,
    pub rng_seed: f64,
    pub voice_contour: Option<Vec<Vec<f64>>>,
    pub voice_contour_resolution: f64,
    pub chord_structure: Vec<i32>,
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
            schillinger_sequence: vec![0, 3, 4, 0],
            pl: 4,
            harmony_distance_balance: 0.2,
            lookahead_depth: 2,
            render_length: 40,
            rng_seed: 5443343433.0,
            voice_contour: None,
            voice_contour_resolution: 4.0,
            chord_structure: vec![0, 1, 2, 4, 5],
        }
    }
}

impl Config {
    pub fn randomize_contours(&mut self) {
        use crate::utils::SeededRng;

        let mut contours = Vec::new();
        // 16 voices
        for _ in 0..16 {
            // Generate 3 to 5 points
            let num_points = 3 + SeededRng::random_int(3); // 0..3 -> 0..2.99 -> 0..2. 3 + 0..2 = 3..5
            // utils.rs: (self._seeded_random(max as f64, 0.0)).floor() as i32
            // if max=8, returns 0..7.99 -> floor -> 0..7.
            // 3 + 0..7 = 3..10. Correct.

            let mut points: Vec<(f64, f64)> = Vec::new();
            let total_len = (self.render_length * 32) as f64;

            for _ in 0..num_points {
                let x = SeededRng::seeded_random(total_len, 0.0);
                let y = SeededRng::seeded_random(12.0, -12.0);
                points.push((x, y));
            }

            // Sort by x
            points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Interpolate
            let mut contour_vec = Vec::new();
            // Calculate number of steps.
            // +1 to cover the end?
            let steps = (total_len / self.voice_contour_resolution).ceil() as usize;

            for i in 0..steps {
                let pos = (i as f64) * self.voice_contour_resolution;

                let mut y = 0.0;
                if points.is_empty() {
                    y = 0.0;
                } else if pos <= points[0].0 {
                    y = points[0].1;
                } else if pos >= points.last().unwrap().0 {
                    y = points.last().unwrap().1;
                } else {
                    // Linear interpolation
                    for j in 0..points.len() - 1 {
                        if pos >= points[j].0 && pos <= points[j + 1].0 {
                            let p1 = points[j];
                            let p2 = points[j + 1];
                            // Avoid div by zero if x matches (unlikely with random, but possible if sorted puts them close)
                            if (p2.0 - p1.0).abs() < 0.0001 {
                                y = p1.1;
                            } else {
                                let t = (pos - p1.0) / (p2.0 - p1.0);
                                y = p1.1 + t * (p2.1 - p1.1);
                            }
                            break;
                        }
                    }
                }
                contour_vec.push(y);
            }
            contours.push(contour_vec);
        }
        self.voice_contour = Some(contours);
    }
}
