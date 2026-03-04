// SPDX-License-Identifier: AGPL-3.0-or-later

// Test NPU setup

use akida_driver::setup::NpuSetup;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let mut setup = NpuSetup::new();
    setup.run()?;

    Ok(())
}
