use std::io::{self, Stdout};
use std::time::Duration;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    prelude::*,
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Gauge},
};
use crate::model::Config;
use crate::run_generation;

#[derive(PartialEq, Eq)]
enum InputMode {
    Normal,
    Editing,
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
}

impl App {
    pub fn new() -> Self {
        let config = Config::default();
        let keys = vec![
            "schillinger_progression",
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
        }
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
            _ => "N/A".to_string(),
        }
    }

    pub fn update_value(&mut self) {
        if let Some(i) = self.state.selected() {
            let key = self.keys[i];
            match key {
                "schillinger_progression" => if let Ok(v) = self.input_buffer.parse::<i32>() { self.config.schillinger_progression = if v == 1 { true } else { false }; },
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
                "rng_seed" => if let Ok(v) = self.input_buffer.parse() { self.config.rng_seed = v; },
                _ => {},
            }
        }
    }
}

pub fn run_tui() -> io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new();
    let res = run_app(&mut terminal, &mut app);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
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
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
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

                                    // Need a bridge thread or just pass a closure that converts?
                                    // harmonise2 expects Sender<(usize, usize)>
                                    // We can just pass prog_tx.
                                    // And concurrently read prog_rx and forward to tx?
                                    // Or just wrap run_generation to take our specific callback?
                                    // `run_generation` takes Sender<(usize, usize)>.
                                    
                                    // Let's spawn a helper to forward progress if we want to adapt types,
                                    // but here we can just pass the tx directly if we change Message type?
                                    // No, GenerationMessage is the wrapper.
                                    
                                    // Actually, simpler:
                                    // Pass a sender that sends just (usize, usize).
                                    // This thread receives them and forwards GenerationMessage::Progress to main thread.
                                    
                                    let (internal_tx, internal_rx) = channel();
                                    
                                    // We need to run generation in this thread.
                                    // But we also need to forward progress updates.
                                    // But run_generation blocks.
                                    // So we can't receive on internal_rx in THIS thread while run_generation is running.
                                    // Solutions:
                                    // 1. Spawn ANOTHER thread for generation.
                                    // 2. pass `tx.clone()` wrapped in a wrapper struct that sends GenerationMessage directly?
                                    //    But run_generation expects Sender<(usize, usize)>.
                                    
                                    // 3. Just change run_generation signature? No, user accepted plan.
                                    
                                    // Let's go with 1.
                                    let tx_clone = tx.clone();
                                    let config_clone = config.clone();
                                    
                                    thread::spawn(move || {
                                        while let Ok((c, t)) = internal_rx.recv() {
                                            let _ = tx_clone.send(GenerationMessage::Progress(c, t));
                                        }
                                    });

                                    let res = run_generation(&config, Some(internal_tx));
                                    // Convert std::io::Error to String
                                    let res_str = res.map_err(|e| e.to_string());
                                    
                                    let _ = tx.send(GenerationMessage::Finished(res_str));
                               });
                           }
                       },
                       KeyCode::Down => app.next(),
                       KeyCode::Up => app.previous(),
                       KeyCode::Enter => {
                           app.input_mode = InputMode::Editing;
                           if let Some(i) = app.state.selected() {
                               app.input_buffer = app.get_value(app.keys[i]);
                           }
                       }
                       _ => {}
                   }
                }
            }
        }
    }
}

fn ui(f: &mut Frame, app: &mut App) {
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
        };

        let paragraph = Paragraph::new(status)
            .block(Block::default().borders(Borders::ALL));
        f.render_widget(paragraph, chunks[1]);
    }
}
