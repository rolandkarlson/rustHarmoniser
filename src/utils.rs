use std::cell::RefCell;

thread_local! {
    static THREAD_RNG: RefCell<SeededRng> = RefCell::new(SeededRng::new(5443343433.0));
}

pub struct SeededRng {
    seed: f64,
}

impl SeededRng {
    pub fn new(seed: f64) -> Self {
        Self { seed }
    }

    // Instance methods
    fn _random(&mut self) -> f64 {
        self.seed = (self.seed * 9301.0 + 49297.0) % 233280.0;
        self.seed / 233280.0
    }

    fn _seeded_random(&mut self, max: f64, min: f64) -> f64 {
        let max = if max == 0.0 { 1.0 } else { max };
        let rnd = self._random();
        min + rnd * (max - min)
    }

    fn _random_int(&mut self, max: i32) -> i32 {
        (self._seeded_random(max as f64, 0.0)).floor() as i32
    }

    // Static implementations using thread-local RNG (no mutex contention)
    pub fn random() -> f64 {
        THREAD_RNG.with(|rng| rng.borrow_mut()._random())
    }

    pub fn seeded_random(max: f64, min: f64) -> f64 {
        THREAD_RNG.with(|rng| rng.borrow_mut()._seeded_random(max, min))
    }

    pub fn random_int(max: i32) -> i32 {
        THREAD_RNG.with(|rng| rng.borrow_mut()._random_int(max))
    }

    pub fn set_seed(new_seed: f64) {
        THREAD_RNG.with(|rng| rng.borrow_mut().seed = new_seed);
    }
}// or we can pass it around. JS uses a global.
// Let's try to pass it around for better Rust practice, or use a RefCell thread local if it gets too hairy.

pub fn mod_shim(x: i32, m: i32) -> i32 {
    ((x % m) + m) % m
}

pub fn sin(pos: f64, frequency: f64, amplitude: f64) -> f64 {
    let increase = std::f64::consts::PI / (400.0 / frequency);
    amplitude + amplitude * (pos * increase).sin()
}

pub trait ArrayExt<T> {
    fn get_wrapped(&self, index: usize) -> &T;
    fn exists(&self, item: &T) -> bool where T: PartialEq;
    fn find_closest(&self, target: i32, exclude_list: &[i32]) -> i32 where T: Into<i32> + Copy;
}

impl<T> ArrayExt<T> for [T] {
    fn get_wrapped(&self, index: usize) -> &T {
        &self[mod_shim(index as i32, self.len() as i32) as usize]
    }

    fn exists(&self, item: &T) -> bool where T: PartialEq {
        self.contains(item)
    }

    fn find_closest(&self, target: i32, exclude_list: &[i32]) -> i32 where T: Into<i32> + Copy {
        let mut closest = 0; // Default, though technically should be from array
        let mut min_diff = i32::MAX;
        
        // Handle empty case if needed, or assume non-empty as per JS usage
        if self.is_empty() { return 0; } // Fail safe

        for &item in self {
             let val: i32 = item.into();
             if exclude_list.contains(&val) {
                 continue;
             }
             
             let diff = (val - target).abs();
             if diff < min_diff {
                 min_diff = diff;
                 closest = val;
             }
        }
        closest
    }
}
