#!/bin/bash
# Build the Rust backend and start the GUI dev server

set -e

echo "Building Rust project..."
cargo build
