//! The pure harmonizer engine. No I/O: a render is a function of a `Config`
//! (plus an optional leading clip) and nothing else. The server crate owns
//! every side effect — HTTP, Ableton, and file sinks.

pub mod contour;
pub mod model;
pub mod music_theory;
pub mod rhythm;
pub mod harmonizer;
pub mod schillinger;
pub mod utils;
pub mod render;
pub mod trace;
