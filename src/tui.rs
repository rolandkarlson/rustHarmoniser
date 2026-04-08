use std::io::{self, Stdout};
use std::time::Duration;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, MouseEventKind, MouseButton},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Gauge, Chart, Dataset, Axis, GraphType},
    style::{Color, Modifier, Style},
    symbols,
};
use crate::model::Config;
use crate::run_generation;
use crate::schillinger::generate_progression;
use crate::utils::SeededRng;

#[derive(PartialEq, Eq)]
enum InputMode {
    Normal,
    Editing,
    Contour,
    HarmonyContour,
    ModeContour,
    ChordContour,
}

enum GenerationMessage {
    Progress(usize, usize),
    Finished(Result<String, String>),
}

pub struct App {
    pub config: Config,
    pub state: ListState,
    pub input_mode: InputMode,
    pub input_buffer: String,
    pub keys: Vec<&'static str>,
    pub status_message: String,
    pub progress: f64,
    pub is_generating: bool,
    pub rx: Option<Receiver<GenerationMessage>>,
    
    // Contour editor state
    pub selected_voice: usize,
    pub chart_area: Rect,
    pub last_mouse_pos: Option<(usize, f64)>,
}

impl App {
    pub fn new() -> Self {
        let mut config = Config::default();
        config.randomize_contours();
        
        SeededRng::set_seed(config.rng_seed);
        config.schillinger_sequence = Self::genProgression(&mut config);
        config.init_contours();
        let keys = vec![
            "schillinger_progression",
            "schillinger_sequence",
            "mode",
            "chord_structure",
            "pl",
            "last_note_exist_in_voice",
            "same_direction",
            "consecutive_octav_fift",
            "no_crossing",
            "last_note_same",
            "interval_exists_in_harmony",
            "harmony_distance_balance",
            "lookahead_depth",
            "render_length",
            "voice_rhythm",
            "rng_seed",
            "Edit Voice Contours",
            "Edit Harmony Contour",
            "Edit Mode Contour",
            "Edit Chord Contour",
        ];
        let mut state = ListState::default();
        state.select(Some(0));

        Self {
            config,
            state,
            input_mode: InputMode::Normal,
            input_buffer: String::new(),
            keys,
            status_message: "Press 'r' to run generation. 'q' to quit. Enter to edit.".into(),
            progress: 0.0,
            is_generating: false,
            rx: None,
            selected_voice: 0,
            chart_area: Rect::default(),
            last_mouse_pos: None,
        }
    }

    fn genProgression(mut config: &mut Config) -> Vec<i32> {
        [
            generate_progression(config.pl as usize, config.mode),
            generate_progression(config.pl as usize, config.mode),
            generate_progression(config.pl as usize, config.mode),
            generate_progression(config.pl as usize, config.mode),
            generate_progression(config.pl as usize, config.mode),
            generate_progression(config.pl as usize, config.mode),
            generate_progression(config.pl as usize, config.mode),
            generate_progression(config.pl as usize, config.mode),
            generate_progression(config.pl as usize, config.mode),
            generate_progression(config.pl as usize, config.mode),
            generate_progression(config.pl as usize, config.mode),
            generate_progression(config.pl as usize, config.mode),
            generate_progression(config.pl as usize, config.mode),
            generate_progression(config.pl as usize, config.mode),
            generate_progression(config.pl as usize, config.mode),
            generate_progression(config.pl as usize, config.mode),
            generate_progression(config.pl as usize, config.mode),
            generate_progression(config.pl as usize, config.mode),
        ].concat()
    }

    pub fn next(&mut self) {
        let i = match self.state.selected() {
            Some(i) => {
                if i >= self.keys.len() - 1 {
                    0
                } else {
                    i + 1
                }
            }
            None => 0,
        };
        self.state.select(Some(i));
    }

    pub fn previous(&mut self) {
        let i = match self.state.selected() {
            Some(i) => {
                if i == 0 {
                    self.keys.len() - 1
                } else {
                    i - 1
                }
            }
            None => 0,
        };
        self.state.select(Some(i));
    }

    pub fn get_value(&self, key: &str) -> String {
        match key {
            "schillinger_progression" => self.config.schillinger_progression.to_string(),
            "schillinger_sequence" => self.config.schillinger_sequence.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(", "),
            "chord_structure" => self.config.chord_structure.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(", "),
            "mode" => self.config.mode.to_string(),
            "pl" => self.config.pl.to_string(),
            "last_note_exist_in_voice" => self.config.last_note_exist_in_voice.to_string(),
            "same_direction" => self.config.same_direction.to_string(),
            "consecutive_octav_fift" => self.config.consecutive_octav_fift.to_string(),
            "no_crossing" => self.config.no_crossing.to_string(),
            "last_note_same" => self.config.last_note_same.to_string(),
            "interval_exists_in_harmony" => self.config.interval_exists_in_harmony.to_string(),
            "harmony_distance_balance" => self.config.harmony_distance_balance.to_string(),
            "lookahead_depth" => self.config.lookahead_depth.to_string(),
            "render_length" => self.config.render_length.to_string(),
            "voice_rhythm" => self.config.voice_rhythm.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(", "),
            "rng_seed" => self.config.rng_seed.to_string(),
            "Edit Voice Contours" => "Press Enter".to_string(),
            "Edit Harmony Contour" => "Press Enter".to_string(),
            "Edit Mode Contour" => "Press Enter".to_string(),
            "Edit Chord Contour" => "Press Enter".to_string(),
            _ => "N/A".to_string(),
        }
    }

    pub fn update_value(&mut self) {
        if let Some(i) = self.state.selected() {
            let key = self.keys[i];
            match key {
                "schillinger_progression" => if let Ok(v) = self.input_buffer.parse::<i32>() { self.config.schillinger_progression = if v == 1 { true } else { false }; },
                "schillinger_sequence" => {
                    let parts: Result<Vec<i32>, _> = self.input_buffer.split(',')
                        .map(|s| s.trim().parse::<i32>())
                        .collect();
                    if let Ok(v) = parts {
                        if !v.is_empty() {
                            self.config.schillinger_sequence = v;
                        }
                    }
                },
                "chord_structure" => {
                    let parts: Result<Vec<i32>, _> = self.input_buffer.split(',')
                        .map(|s| s.trim().parse::<i32>())
                        .collect();
                    if let Ok(v) = parts {
                        if !v.is_empty() {
                            self.config.chord_structure = v;
                        }
                    }
                },
                "pl" => if let Ok(v) = self.input_buffer.parse() { 
                    self.config.pl = v; 
                    SeededRng::set_seed(self.config.rng_seed);
                    self.config.schillinger_sequence = generate_progression(self.config.pl as usize, self.config.mode);
                },
                "mode" => if let Ok(v) = self.input_buffer.parse() { self.config.mode = v; },
                "last_note_exist_in_voice" => if let Ok(v) = self.input_buffer.parse() { self.config.last_note_exist_in_voice = v; },
                "same_direction" => if let Ok(v) = self.input_buffer.parse() { self.config.same_direction = v; },
                "consecutive_octav_fift" => if let Ok(v) = self.input_buffer.parse() { self.config.consecutive_octav_fift = v; },
                "no_crossing" => if let Ok(v) = self.input_buffer.parse() { self.config.no_crossing = v; },
                "last_note_same" => if let Ok(v) = self.input_buffer.parse() { self.config.last_note_same = v; },
                "interval_exists_in_harmony" => if let Ok(v) = self.input_buffer.parse() { self.config.interval_exists_in_harmony = v; },
                "harmony_distance_balance" => if let Ok(v) = self.input_buffer.parse() { self.config.harmony_distance_balance = v; },
                "lookahead_depth" => if let Ok(v) = self.input_buffer.parse() { self.config.lookahead_depth = v; },
                "render_length" => if let Ok(v) = self.input_buffer.parse() { self.config.render_length = v; },
                "voice_rhythm" => {
                    let parts: Result<Vec<f64>, _> = self.input_buffer.split(',')
                        .map(|s| s.trim().parse::<f64>())
                        .collect();
                    if let Ok(v) = parts {
                        if !v.is_empty() {
                            self.config.voice_rhythm = v;
                        }
                    }
                },
                "rng_seed" => if let Ok(v) = self.input_buffer.parse() { 
                    self.config.rng_seed = v; 
                    SeededRng::set_seed(self.config.rng_seed);
                    self.config.schillinger_sequence =Self::genProgression(&mut self.config);
                },
                _ => {},
            }
        }
    }
}

pub fn run_tui() -> io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, event::EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new();
    let res = run_app(&mut terminal, &mut app);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, event::DisableMouseCapture)?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{:?}", err)
    }

    Ok(())
}

fn run_app<B: Backend>(terminal: &mut Terminal<B>, app: &mut App) -> io::Result<()> {
    loop {
        terminal.draw(|f| ui(f, app))?;

        // Handle Messages
        if let Some(rx) = &app.rx {
             while let Ok(msg) = rx.try_recv() {
                 match msg {
                     GenerationMessage::Progress(curr, total) => {
                         if total > 0 {
                            app.progress = (curr as f64) / (total as f64);
                         }
                     },
                     GenerationMessage::Finished(res) => {
                         app.is_generating = false;
                         match res {
                             Ok(msg) => app.status_message = format!("Success: {}", msg),
                             Err(e) => app.status_message = format!("Error: {}", e),
                         }
                         app.progress = 1.0;
                         // Don't clear rx immediately if we want to fetch result? 
                         // Ah, we just handled the result.
                     }
                 }
             }
             if !app.is_generating {
                 app.rx = None;
             }
        }

        if event::poll(Duration::from_millis(50))? {
            let event = event::read()?;
            match event {
                Event::Key(key) => {
                    if key.kind == KeyEventKind::Press {
                        if app.input_mode == InputMode::Contour || app.input_mode == InputMode::HarmonyContour || app.input_mode == InputMode::ModeContour || app.input_mode == InputMode::ChordContour {
                            match key.code {
                                KeyCode::Esc => app.input_mode = InputMode::Normal,
                                KeyCode::Char('v') if app.input_mode == InputMode::Contour => {
                                    app.selected_voice = (app.selected_voice + 1) % 16;
                                },
                                KeyCode::Char('c') => {
                                    if app.input_mode == InputMode::Contour {
                                        if let Some(contours) = &mut app.config.voice_contour {
                                            if app.selected_voice < contours.len() {
                                                contours[app.selected_voice].clear();
                                            }
                                        }
                                    } else if app.input_mode == InputMode::HarmonyContour {
                                        if let Some(contour) = &mut app.config.harmony_distance_contour {
                                            contour.clear();
                                        }
                                    } else if app.input_mode == InputMode::ModeContour {
                                        if let Some(contour) = &mut app.config.mode_contour {
                                            contour.clear();
                                        }
                                    } else if app.input_mode == InputMode::ChordContour {
                                        if let Some(contour) = &mut app.config.chord_structure_contour {
                                            contour.clear();
                                        }
                                    }
                                },
                                _ => {}
                            }
                            continue;
                        }

                       if app.input_mode == InputMode::Editing {
                           match key.code {
                                KeyCode::Enter => {
                                    app.update_value();
                                    app.input_mode = InputMode::Normal;
                                },
                                KeyCode::Esc => {
                                    app.input_mode = InputMode::Normal;
                                },
                                KeyCode::Backspace => {
                                    app.input_buffer.pop();
                                },
                                KeyCode::Char(c) => {
                                    app.input_buffer.push(c);
                                },
                                _ => {}
                           }
                           continue;
                       }
                       
                       // Normal mode
                       match key.code {
                           KeyCode::Char('q') => return Ok(()),
                           KeyCode::Char('r') => {
                               if !app.is_generating {
                                   app.is_generating = true;
                                   app.progress = 0.0;
                                   app.status_message = "Generating...".into();
                                   
                                   let (tx, rx) = channel();
                                   let config = app.config.clone();
                                   app.rx = Some(rx);

                                   thread::spawn(move || {
                                        let (internal_tx, internal_rx) = channel();
                                        let tx_clone = tx.clone();
                                        thread::spawn(move || {
                                            while let Ok((c, t)) = internal_rx.recv() {
                                                let _ = tx_clone.send(GenerationMessage::Progress(c, t));
                                            }
                                        });

                                        let res = run_generation(&config, Some(internal_tx));
                                        let res_str = res.map_err(|e| e.to_string());
                                        let _ = tx.send(GenerationMessage::Finished(res_str));
                                   });
                               }
                           },
                           KeyCode::Down => app.next(),
                           KeyCode::Up => app.previous(),
                           KeyCode::Enter => {
                               if let Some(i) = app.state.selected() {
                                     if app.keys[i] == "Edit Voice Contours" {
                                         app.input_mode = InputMode::Contour;
                                         // Initialize contour vector if needed
                                         if app.config.voice_contour.is_none() {
                                             app.config.voice_contour = Some(vec![Vec::new(); 16]);
                                         }
                                     } else if app.keys[i] == "Edit Harmony Contour" {
                                         app.input_mode = InputMode::HarmonyContour;
                                         if app.config.harmony_distance_contour.is_none() {
                                             app.config.init_contours();
                                         }
                                     } else if app.keys[i] == "Edit Mode Contour" {
                                         app.input_mode = InputMode::ModeContour;
                                         if app.config.mode_contour.is_none() {
                                             app.config.init_contours();
                                         }
                                     } else if app.keys[i] == "Edit Chord Contour" {
                                         app.input_mode = InputMode::ChordContour;
                                         if app.config.chord_structure_contour.is_none() {
                                             app.config.init_contours();
                                         }
                                     } else {
                                         app.input_mode = InputMode::Editing;
                                         app.input_buffer = app.get_value(app.keys[i]);
                                     }
                               }
                           }
                           _ => {}
                       }
                    }
                },
                Event::Mouse(mouse) => {
                    if app.input_mode == InputMode::Contour || app.input_mode == InputMode::HarmonyContour || app.input_mode == InputMode::ModeContour || app.input_mode == InputMode::ChordContour {
                        match mouse.kind {
                            MouseEventKind::Down(MouseButton::Left) | MouseEventKind::Drag(MouseButton::Left) => {
                                let x = mouse.column as f64;
                                let y = mouse.row as f64;
                                let area = app.chart_area;

                                if x >= area.x as f64 && x < (area.x + area.width) as f64 &&
                                   y >= area.y as f64 && y < (area.y + area.height) as f64 {

                                    // Map screen coords to chart coords
                                    let is_harmony = app.input_mode == InputMode::HarmonyContour;
                                    let is_mode = app.input_mode == InputMode::ModeContour;
                                    let is_chord = app.input_mode == InputMode::ChordContour;

                                    let chart_x_min = 0.0;
                                    let chart_x_max = (app.config.pl * 4 * app.config.render_length) as f64;
                                    
                                    let (chart_y_min, chart_y_max) = if is_harmony {
                                        (-0.2, 0.5)
                                    } else if is_mode {
                                        (0.0, 7.0)
                                    } else if is_chord {
                                        (0.0, 5.0)
                                    } else {
                                        (-12.0, 12.0)
                                    };

                                    // Approximate inner chart drawing area avoiding borders and axes labels
                                    let inner_x_start = area.x as f64 + 5.0; // Y-axis labels + left border
                                    let inner_width = (area.width as f64 - 6.0).max(1.0); // Right border
                                    
                                    let inner_y_start = area.y as f64 + 1.0; // Top border
                                    let inner_height = (area.height as f64 - 3.0).max(1.0); // X-axis labels + bottom border

                                    let mut rel_x = (x - inner_x_start) / inner_width;
                                    // Invert Y: 1.0 is top (inner_y_start), 0.0 is bottom (inner_y_start + inner_height)
                                    let mut rel_y = 1.0 - (y - inner_y_start) / inner_height; 

                                    rel_x = rel_x.clamp(0.0, 1.0);
                                    rel_y = rel_y.clamp(0.0, 1.0);

                                    let data_x = chart_x_min + rel_x * (chart_x_max - chart_x_min);
                                    let data_y = chart_y_min + rel_y * (chart_y_max - chart_y_min);

                                    let resolution = app.config.voice_contour_resolution;
                                    let idx = (data_x / resolution).round() as usize;

                                    let active_config_vec = if is_harmony {
                                        &mut app.config.harmony_distance_contour
                                    } else if is_mode {
                                        &mut app.config.mode_contour
                                    } else if is_chord {
                                        &mut app.config.chord_structure_contour
                                    } else {
                                        // Custom resolve for VoiceContour mapping specifically avoiding conflict
                                        &mut None
                                    };

                                    if !is_harmony && !is_mode && !is_chord {
                                        if let Some(contours) = &mut app.config.voice_contour {
                                            if app.selected_voice < contours.len() {
                                                let vec = &mut contours[app.selected_voice];
                                                if idx >= vec.len() {
                                                    vec.resize(idx + 1, 0.0);
                                                }
                                                
                                                if let MouseEventKind::Drag(_) = mouse.kind {
                                                    if let Some((prev_idx, prev_val)) = app.last_mouse_pos {
                                                        let start = std::cmp::min(prev_idx, idx);
                                                        let end = std::cmp::max(prev_idx, idx);
                                                        
                                                        if end > start {
                                                            for i in start..=end {
                                                                if i >= vec.len() {
                                                                    vec.resize(i + 1, 0.0);
                                                                }
                                                                let t = (i - start) as f64 / (end - start) as f64;
                                                                let val = if idx > prev_idx {
                                                                    prev_val + t * (data_y - prev_val)
                                                                } else {
                                                                    data_y + t * (prev_val - data_y)
                                                                };
                                                                vec[i] = val;
                                                            }
                                                        }
                                                    }
                                                }
                                                
                                                vec[idx] = data_y;
                                                app.last_mouse_pos = Some((idx, data_y));
                                            }
                                        }
                                    } else {
                                        if let Some(vec) = active_config_vec {
                                            let fill_val = if is_harmony { 0.2 } else if is_mode { 0.0 } else { 1.0 };
                                            if idx >= vec.len() {
                                                vec.resize(idx + 1, fill_val);
                                            }
                                            
                                            if let MouseEventKind::Drag(_) = mouse.kind {
                                                if let Some((prev_idx, prev_val)) = app.last_mouse_pos {
                                                    let start = std::cmp::min(prev_idx, idx);
                                                    let end = std::cmp::max(prev_idx, idx);
                                                    
                                                    if end > start {
                                                        for i in start..=end {
                                                            if i >= vec.len() {
                                                                vec.resize(i + 1, fill_val);
                                                            }
                                                            let t = (i - start) as f64 / (end - start) as f64;
                                                            let val = if idx > prev_idx {
                                                                prev_val + t * (data_y - prev_val)
                                                            } else {
                                                                data_y + t * (prev_val - data_y)
                                                            };
                                                            vec[i] = val;
                                                        }
                                                    }
                                                }
                                            }
                                            vec[idx] = data_y;
                                            app.last_mouse_pos = Some((idx, data_y));
                                        }
                                    }
                                } else {
                                    // Outside area, reset?
                                    app.last_mouse_pos = None;
                                }
                            },
                            MouseEventKind::Up(_) => {
                                app.last_mouse_pos = None;
                            },
                            _ => {}
                        }
                    }
                },
                _ => {}
            }
        }
    }
}

fn ui(f: &mut Frame, app: &mut App) {
    if app.input_mode == InputMode::Contour || app.input_mode == InputMode::HarmonyContour || app.input_mode == InputMode::ModeContour || app.input_mode == InputMode::ChordContour {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(0),
                Constraint::Length(3),
            ])
            .split(f.size());


        // Prepare dataset
        let mut data_points = Vec::new();
        let mut box_points = Vec::new();
        let is_harmony = app.input_mode == InputMode::HarmonyContour;
        let is_mode = app.input_mode == InputMode::ModeContour;
        let is_chord = app.input_mode == InputMode::ChordContour;
        let has_boxes = is_harmony || is_mode || is_chord;

        let active_contour = if is_harmony {
            &app.config.harmony_distance_contour
        } else if is_mode {
            &app.config.mode_contour
        } else if is_chord {
            &app.config.chord_structure_contour
        } else {
            &None
        };
        
        if has_boxes {
            if let Some(contour) = active_contour {
                for (i, &val) in contour.iter().enumerate() {
                    let x = (i as f64) * app.config.voice_contour_resolution;
                    data_points.push((x, val));
                }
                
                // Draw boxes for bars
                let (y_min, y_max) = if is_harmony { (-0.2, 0.5) } else if is_mode { (0.0, 7.0) } else { (0.0, 5.0) };
                for i in 0..app.config.render_length {
                    let x_start = (i as f64) * (app.config.pl as f64 * 4.0);
                    let x_end = ((i + 1) as f64) * (app.config.pl as f64 * 4.0);
                    box_points.push((x_start, y_min));
                    box_points.push((x_start, y_max));
                    box_points.push((x_end, y_max));
                    box_points.push((x_end, y_min));
                    box_points.push((x_start, y_min));
                }
            }
        } else {
            if let Some(contours) = &app.config.voice_contour {
                if app.selected_voice < contours.len() {
                    for (i, &val) in contours[app.selected_voice].iter().enumerate() {
                        let x = (i as f64) * app.config.voice_contour_resolution;
                         if val != 0.0 { // Optimization: only plot non-zeros? Or consistent plot?
                            data_points.push((x, val));
                         }
                    }
                }
            }
        }
        
        let x_max = if has_boxes { (app.config.pl * 4 * app.config.render_length) as f64 } else { (app.config.render_length * 32) as f64 };
        let (y_min_val, y_max_val) = if is_harmony {
            (-0.2, 0.5)
        } else if is_mode {
            (0.0, 7.0)
        } else if is_chord {
            (0.0, 5.0)
        } else {
            (-12.0, 12.0)
        };

        let x_labels = vec![
            Span::styled("0", Style::default().add_modifier(Modifier::BOLD)),
            Span::styled(format!("{}", x_max), Style::default().add_modifier(Modifier::BOLD)),
        ];

        let mut datasets = Vec::new();
        
        if has_boxes {
             datasets.push(
                Dataset::default()
                    .name("Bars")
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(Color::DarkGray))
                    .data(&box_points)
             );
        }

        let contour_name = if is_harmony {
            "Harmony Contour".to_string()
        } else if is_mode {
            "Mode Contour".to_string()
        } else if is_chord {
            "Chord Contour".to_string()
        } else {
            format!("Voice {} Contour", app.selected_voice)
        };

        let active_color = if is_harmony { Color::Magenta } else if is_mode { Color::Yellow } else if is_chord { Color::LightGreen } else { Color::Cyan };

        datasets.push(
            Dataset::default()
                .name(contour_name)
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(active_color))
                .data(&data_points)
        );

        let title = if is_harmony {
            "Harmony Contour Editor (Mouse Draw, C: clear, Esc: back)"
        } else if is_mode {
            "Mode Contour Editor (Mouse Draw, C: clear, Esc: back)"
        } else if is_chord {
            "Chord Contour Editor (Mouse Draw, C: clear, Esc: back)"
        } else {
            "Voice Contour Editor (Mouse Draw, V: switch voice, C: clear, Esc: back)"
        };

        let y_label = if is_harmony { "Balance" } else if is_mode { "Mode" } else if is_chord { "Chord" } else { "Pitch Shift" };

        let chart = Chart::new(datasets)
            .block(Block::default().title(title).borders(Borders::ALL))
            .x_axis(Axis::default()
                .title("Time")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, x_max])
                .labels(x_labels))
            .y_axis(Axis::default()
                .title(y_label)
                .style(Style::default().fg(Color::Gray))
                .bounds([y_min_val, y_max_val])
                .labels(vec![
                    Span::styled(format!("{:.1}", y_min_val), Style::default().add_modifier(Modifier::BOLD)),
                    Span::styled(format!("{:.1}", (y_min_val + y_max_val) / 2.0), Style::default().add_modifier(Modifier::BOLD)),
                    Span::styled(format!("{:.1}", y_max_val), Style::default().add_modifier(Modifier::BOLD)),
                ]));
        
        f.render_widget(chart, chunks[0]);
        app.chart_area = chunks[0]; // Save area for mouse interaction
        
    } else {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(1),
                Constraint::Length(3),
            ])
            .split(f.size());

        let items: Vec<ListItem> = app.keys.iter().map(|key| {
            let val = app.get_value(key);
            ListItem::new(format!("{}: {}", key, val))
        }).collect();

        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title("Configuration"))
            .highlight_style(Style::default().bg(Color::White).fg(Color::Black))
            .highlight_symbol(">> ");

        f.render_stateful_widget(list, chunks[0], &mut app.state);

        if app.is_generating {
            let label = format!("{:.1}%", app.progress * 100.0);
            let gauge = Gauge::default()
                .block(Block::default().borders(Borders::ALL).title("Generating"))
                .gauge_style(Style::default().fg(Color::Green).bg(Color::Black))
                .ratio(app.progress)
                .label(label);
            f.render_widget(gauge, chunks[1]);
        } else {
            let status = match app.input_mode {
                InputMode::Normal => format!("Status: {}", app.status_message),
                InputMode::Editing => format!("Editing: {}_", app.input_buffer),
                _ => String::new(),
            };

            let paragraph = Paragraph::new(status)
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(paragraph, chunks[1]);
        }
    }
}
