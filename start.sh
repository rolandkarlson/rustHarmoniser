#!/bin/bash
set -e

echo "Fetching git updates..."
git pull

echo "Starting GUI in background..."
cd gui
npm run dev &  # The '&' runs this in the background
GUI_PID=$!     # Optional: capture the process ID to kill it later

cd .. # Go back to root for cargo

echo "Building and running Rust project..."
cargo run --release
