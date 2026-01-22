use crate::model::{Note, Config, VoiceState};
use crate::utils::{SeededRng, ArrayExt, mod_shim, sin};
use crate::music_theory::get_harmonic_score_adjusted;
use crate::schillinger;
use rayon::prelude::*;
use std::collections::HashMap;

// --- Data Structures ---
use std::cmp::Ordering;

// ---------- Helpers: small math ----------
fn abs_i32(x: i32) -> i32 { if x < 0 { -x } else { x } }

fn pc(p: i32) -> i32 {
    let m = p % 12;
    if m < 0 { m + 12 } else { m }
}

// ---------- Candidate representation ----------
#[derive(Clone, Debug)]
pub struct Candidate {
    pub voice_state: VoiceState,
    // store per-note chosen pitch for reconstruction if you want
    // but you can also reconstruct from voice_state only (by channel).
}

#[derive(Clone, Debug)]
pub struct ViterbiResult {
    pub path: Vec<VoiceState>, // one VoiceState per group
    pub score: f64,
}

// ---------- Constraint checks ----------
fn violates_voice_crossing(channels_sorted_high_to_low: &[i32], pitches_by_channel: &[i32; 16]) -> bool {
    // Your convention in code: lower channel index = higher voice (0 soprano, 3 bass).
    // Here we assume channels_sorted_high_to_low is already [0,1,2,3,...] for the voices used.
    // Condition: higher voice pitch >= lower voice pitch.
    for w in channels_sorted_high_to_low.windows(2) {
        let ch_hi = w[0] as usize;
        let ch_lo = w[1] as usize;
        let p_hi = pitches_by_channel[ch_hi];
        let p_lo = pitches_by_channel[ch_lo];
        if p_hi > 0 && p_lo > 0 && p_hi < p_lo {
            return true;
        }
    }
    false
}

fn violates_unisons(channels_used: &[i32], pitches_by_channel: &[i32; 16]) -> bool {
    // forbid exact same pitch in two voices (you did score -= 10000)
    for i in 0..channels_used.len() {
        let a = pitches_by_channel[channels_used[i] as usize];
        if a <= 0 { continue; }
        for j in (i + 1)..channels_used.len() {
            let b = pitches_by_channel[channels_used[j] as usize];
            if b <= 0 { continue; }
            if a == b { return true; }
        }
    }
    false
}

fn has_parallel_5_or_8(prev: &VoiceState, next: &VoiceState, channels_used: &[i32]) -> bool {
    // Simple parallel check on pitch-classes: if two voices move and keep interval 7 or 0.
    // This is a simplified version (tighten if you want direction, outer-only, etc.)
    for i in 0..channels_used.len() {
        let chi = channels_used[i] as usize;
        let p1a = prev.pitches[chi] as i32;
        let p2a = next.pitches[chi] as i32;
        if p1a <= 0 || p2a <= 0 { continue; }
        let da = p2a - p1a;

        for j in (i + 1)..channels_used.len() {
            let chj = channels_used[j] as usize;
            let p1b = prev.pitches[chj] as i32;
            let p2b = next.pitches[chj] as i32;
            if p1b <= 0 || p2b <= 0 { continue; }
            let db = p2b - p1b;

            // both moved (optional)
            if da == 0 || db == 0 { continue; }

            let int_prev = abs_i32(p1a - p1b) % 12;
            let int_next = abs_i32(p2a - p2b) % 12;

            if (int_prev == 7 && int_next == 7) || (int_prev == 0 && int_next == 0) {
                // also require same direction for “true” parallels
                if (da > 0 && db > 0) || (da < 0 && db < 0) {
                    return true;
                }
            }
        }
    }
    false
}

// ---------- Scoring ----------
fn transition_score(prev: &VoiceState, next: &VoiceState, channels_used: &[i32], config: &Config) -> f64 {
    // Higher is better (like your code)
    let mut score = 0.0;

    // Smoothness: sum of distance scores
    for &ch in channels_used {
        let i = ch as usize;
        let a = prev.pitches[i] as i32;
        let b = next.pitches[i] as i32;
        if a <= 0 || b <= 0 { continue; }
        let dist = abs_i32(a - b) as f64;
        let max_jump = 7.0;
        if dist == 0.0 {
            score += 1.0;
        } else if dist > max_jump {
            score -= dist * 10.0;
        } else {
            score += (1.0 - (dist / max_jump)).max(0.0);
        }
    }

    // Penalties removed as per request ("no benalties"), EXCEPT for the explicit
    // "last in permutation forces same note penalty" logic requested.
    // This logic implies that we only penalize if *all* voices are stagnant (because if any moved,
    // a valid permutation would place the moving voice last, avoiding the penalty).
    let mut all_same = true;
    for &ch in channels_used {
        let i = ch as usize;
        let a = prev.pitches[i] as i32;
        let b = next.pitches[i] as i32;
        if a > 0 && b > 0 && a != b {
            all_same = false;
            break;
        }
    }

    if all_same {
        score -= config.last_note_same;
    }

    score
}

fn emission_score(cand: &Candidate, _config: &Config) -> f64 {
    // Score within-chord qualities (harmonic consonance)
    let mut score = 0.0;
    let mut count = 0;
    
    let pitches: Vec<i32> = cand.voice_state.pitches.iter()
        .filter(|&&p| p > 0)
        .map(|&p| p as i32)
        .collect();

    if pitches.len() < 2 {
        return 0.0; 
    }

    for i in 0..pitches.len() {
        for j in (i+1)..pitches.len() {
            score += get_harmonic_score_adjusted(pitches[i], pitches[j]);
            count += 1;
        }
    }

    if count > 0 {
        score / count as f64
    } else {
        0.0
    }
}

// ---------- Candidate generation for each chord-group ----------
fn build_candidates_for_group(
    group: &[Note],
    config: &Config,
    per_voice_window: i32,         // e.g. 3 or 5 semitones around the input pitch
    min_pitch: i32,                // e.g. 24
    max_pitch: i32,                // e.g. 96
    max_candidates: usize,
) -> Vec<Candidate> {
    // Determine channels involved (usually 5 notes with channels set)
    let mut channels_used: Vec<i32> = group.iter().map(|n| n.channel).collect();
    channels_used.sort();
    channels_used.dedup();

    // For crossing check we want high->low ordering by channel index
    let mut channels_sorted_high_to_low = channels_used.clone();
    channels_sorted_high_to_low.sort(); // 0,1,2,3,... (0 is highest voice in your convention)

    // Build pitch options per channel based on note.pitch +/- window
    // If multiple notes share a channel in the group, you’ll need a different model.
    // This assumes one note per channel per group.
    let mut options: Vec<(i32, Vec<i32>)> = Vec::new(); // (channel, pitches)
    for &ch in &channels_used {
        let n = group.iter().find(|x| x.channel == ch).unwrap();
        let center = n.pitch;
        let lo = (center - per_voice_window).max(min_pitch);
        let hi = (center + per_voice_window).min(max_pitch);
        let mut ps = Vec::new();
        for p in lo..=hi {
            // hook: if you have chord-tone constraints, enforce here:
            // if !is_allowed_pitch_for_group(p, group, config) { continue; }
            ps.push(p);
        }
        options.push((ch, ps));
    }

    // DFS cartesian product over voices
    let mut out = Vec::new();
    let mut cur = [0i32; 16];
    // start empty (0 means “no pitch”)
    for i in 0..16 { cur[i] = 0; }

    fn dfs(
        idx: usize,
        options: &[(i32, Vec<i32>)],
        cur: &mut [i32; 16],
        channels_used: &[i32],
        channels_sorted_high_to_low: &[i32],
        out: &mut Vec<Candidate>,
        config: &Config,
        max_candidates: usize,
    ) {
        if out.len() >= max_candidates { return; }

        if idx == options.len() {
            // Apply hard constraints
            if violates_unisons(channels_used, cur) { return; }
            if violates_voice_crossing(channels_sorted_high_to_low, cur) { return; }

            // Build VoiceState
            let mut vs = VoiceState::default();
            for &ch in channels_used {
                let p = cur[ch as usize];
                if p > 0 {
                    vs.pitches[ch as usize] = p as u8;
                }
            }
            out.push(Candidate { voice_state: vs });
            return;
        }

        let (ch, ps) = &options[idx];
        let chi = *ch as usize;

        for &p in ps {
            // Example extra hard constraints:
            // - enforce range per voice if you have it
            // - forbid certain pitch classes
            cur[chi] = p;
            dfs(idx + 1, options, cur, channels_used, channels_sorted_high_to_low, out, config, max_candidates);
            cur[chi] = 0;
            if out.len() >= max_candidates { return; }
        }
    }

    dfs(
        0,
        &options,
        &mut cur,
        &channels_used,
        &channels_sorted_high_to_low,
        &mut out,
        config,
        max_candidates,
    );

    out
}

// ---------- True Viterbi DP ----------
pub fn viterbi_harmonise(
    grouped_notes: &[Vec<Note>],
    config: &Config,
    per_voice_window: i32,
    max_candidates_per_group: usize,
) -> ViterbiResult {
    let t_len = grouped_notes.len();
    if t_len == 0 {
        return ViterbiResult { path: Vec::new(), score: 0.0 };
    }

    // Precompute channels_used per group once (for transition scoring)
    let mut group_channels: Vec<Vec<i32>> = Vec::with_capacity(t_len);
    for g in grouped_notes {
        let mut chs: Vec<i32> = g.iter().map(|n| n.channel).collect();
        chs.sort();
        chs.dedup();
        group_channels.push(chs);
    }

    // Generate candidates per timestep (state space)
    // Generate candidates per timestep (state space) - Parallelized
    let cands: Vec<Vec<Candidate>> = grouped_notes.par_iter()
        .map(|g| {
            let cs = build_candidates_for_group(
                g,
                config,
                per_voice_window,
                24,
                96,
                max_candidates_per_group,
            );
            if cs.is_empty() {
                vec![Candidate { voice_state: VoiceState::default() }]
            } else {
                cs
            }
        })
        .collect();

    // DP tables
    // dp[t][j] = best score up to time t ending in candidate j
    // back[t][j] = argmax previous candidate index
    let mut dp: Vec<Vec<f64>> = vec![Vec::new(); t_len];
    let mut back: Vec<Vec<usize>> = vec![Vec::new(); t_len];

    // Init at t=0
    dp[0] = vec![0.0; cands[0].len()];
    back[0] = vec![usize::MAX; cands[0].len()];
    for j in 0..cands[0].len() {
        dp[0][j] = emission_score(&cands[0][j], config);
    }

    // Main DP
    for t in 1..t_len {
        let m = cands[t].len();
        // Prepare parallel writable buffers for current step
        let mut current_dp = vec![-f64::INFINITY; m];
        let mut current_back = vec![usize::MAX; m];

        let channels_used = &group_channels[t]; // you can also union prev+curr if needed

        // Parallelize the inner loop over j
        current_dp.par_iter_mut().zip(current_back.par_iter_mut()).enumerate().for_each(|(j, (dp_val, back_val))| {
            let next_vs = &cands[t][j].voice_state;

            let mut best_val = -f64::INFINITY;
            let mut best_i = 0usize;

            for i in 0..cands[t - 1].len() {
                let prev_vs = &cands[t - 1][i].voice_state;

                let mut stagnation_penalty = 0.0;
                if t >= 3 {
                    let idx_tm1 = i;
                    let idx_tm2 = back[t - 1][idx_tm1];
                    // Ensure backpointer is valid (it should be if path exists)
                    if idx_tm2 != usize::MAX {
                        let idx_tm3 = back[t - 2][idx_tm2];
                        if idx_tm3 != usize::MAX {
                             let c_t = &cands[t][j].voice_state;
                             let c_tm1 = &cands[t-1][idx_tm1].voice_state;
                             let c_tm2 = &cands[t-2][idx_tm2].voice_state;
                             let c_tm3 = &cands[t-3][idx_tm3].voice_state;
                             
                             let mut matched = true;
                             for ch in 0..16 {
                                 let p = c_t.pitches[ch];
                                 if p == 0 || p != c_tm1.pitches[ch] || p != c_tm2.pitches[ch] || p != c_tm3.pitches[ch] {
                                     matched = false;
                                     break;
                                 }
                             }
                             if matched {
                                 stagnation_penalty = config.last_note_exist_in_voice;
                             }
                        }
                    }
                }

                let val = dp[t - 1][i]
                    + transition_score(prev_vs, next_vs, channels_used, config)
                    + emission_score(&cands[t][j], config)
                    - stagnation_penalty;

                if val > best_val {
                    best_val = val;
                    best_i = i;
                }
            }

            *dp_val = best_val;
            *back_val = best_i;
        });

        dp[t] = current_dp;
        back[t] = current_back;
    }

    // Termination: best at final time
    let last = t_len - 1;
    let (mut best_j, mut best_score) = (0usize, -f64::INFINITY);
    for j in 0..dp[last].len() {
        if dp[last][j] > best_score {
            best_score = dp[last][j];
            best_j = j;
        }
    }

    // Reconstruct
    let mut path: Vec<VoiceState> = vec![VoiceState::default(); t_len];
    let mut j = best_j;
    for t in (0..t_len).rev() {
        path[t] = cands[t][j].voice_state.clone();
        if t > 0 {
            j = back[t][j];
        }
    }

    ViterbiResult { path, score: best_score }
}

pub fn harmonise2(income: Vec<Note>, config: &Config, _state: &HarmonizerState) -> Vec<Note> {
    // 1. Group notes by start time
    let mut grouped_notes = group_by_start_array(income);
    let vNotes = viterbi_harmonise(&grouped_notes, config, 4, 3000);
    let mut out = Vec::new();
    for (t, group) in grouped_notes.iter_mut().enumerate() {
        let vs = &vNotes.path[t];
        for n in group.iter_mut() {
            let ch = n.channel as usize;
            let p = vs.pitches[ch];
            if p > 0 {
                n.pitch = p as i32;
                n.muted = 0;
            }
            out.push(n.clone());
        }
    }
    out
}

// --- Rest of Boilerplate (get_permutations, group_by_start_array) ---

pub fn get_permutations(notes: &[Note]) -> Vec<Vec<Note>> {
    let mut results = Vec::new();
    let notes_vec = notes.to_vec();
    permute_recursive(Vec::new(), notes_vec, &mut results);
    results
}

fn permute_recursive(current: Vec<Note>, remaining: Vec<Note>, results: &mut Vec<Vec<Note>>) {
    if remaining.is_empty() {
        results.push(current);
        return;
    }
    for i in 0..remaining.len() {
        let mut next_current = current.clone();
        next_current.push(remaining[i]);
        let mut next_remaining = remaining.clone();
        next_remaining.remove(i);
        permute_recursive(next_current, next_remaining, results);
    }
}

fn group_by_start_array(notes: Vec<Note>) -> Vec<Vec<Note>> {
    let mut map: HashMap<String, Vec<Note>> = HashMap::new();
    let quantize = |f: f64| format!("{:.4}", f);

    for n in notes {
        let key = quantize(n.start);
        map.entry(key).or_insert(Vec::new()).push(n);
    }

    let mut groups: Vec<Vec<Note>> = map.into_values().collect();
    groups.sort_by(|a, b| a[0].start.partial_cmp(&b[0].start).unwrap());

    for g in &mut groups {
        g.sort_by(|a, b| b.pitch.cmp(&a.pitch));
    }

    groups
}

pub struct HarmonizerState {
     pub schillinger_notes: Vec<Vec<i32>>,
}

pub fn gen_voice(base: i32, rhythm_data: &Vec<f64>, pitch_shifts: &[i32], channel: i32, muted: i32) -> Vec<Note> {
     let mut ar = Vec::new();
    let clip_len = schillinger::CLIP_LEN as f64;
    let mut pos = 0.0;
    let mut counter = 0;
    let sf = (SeededRng::random_int(60) + 1) as f64;

    while pos < clip_len {
        let n = base + pitch_shifts[mod_shim(counter, pitch_shifts.len() as i32) as usize];
        let d = rhythm_data[mod_shim(counter, rhythm_data.len() as i32) as usize];
        let v = 1 + SeededRng::random_int(10) + sin(counter as f64, sf, 10.0) as i32;
        ar.push(Note::new(n, pos, d, v, muted, channel));
        pos += d;
        counter += 1;
    }
    ar
}
