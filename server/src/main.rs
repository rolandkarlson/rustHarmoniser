mod ableton;
mod api;
mod output;

fn main() -> std::io::Result<()> {
    println!("Starting Web Server mode...");
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            api::start_server().await;
        });
    Ok(())
}
