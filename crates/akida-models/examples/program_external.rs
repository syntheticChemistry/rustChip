// SPDX-License-Identifier: AGPL-3.0-only
//! Demonstrates `program_external()` — loading a model at a specific NP address.
//!
//! This is the key mechanism for multi-tenancy (Exp 002) and hybrid tanh (Exp 004).
//!
//! `program_external(bytes, address)` places the model's NP weights starting at
//! `address` in the NP SRAM space, rather than always at offset 0. Multiple models
//! at disjoint NP address ranges coexist independently.
//!
//! # Usage
//!
//! ```bash
//! # With hardware:
//! cargo run --example program_external -- model.fbz 0x0000
//! cargo run --example program_external -- model.fbz 0x00B3  # slot 2 (after 179-NP model)
//! ```

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <model.fbz> <np_address_hex>", args[0]);
        eprintln!();
        eprintln!("  np_address_hex  NP SRAM offset in hex (e.g. 0x0000, 0x00B3, 0x0139)");
        eprintln!();
        eprintln!("  NP Address Map (from baseCamp/systems/README.md):");
        eprintln!("    0x0000  Slot 1: ESN-QCD         (179 NPs)");
        eprintln!("    0x00B3  Slot 2: Transport        (134 NPs)");
        eprintln!("    0x0139  Slot 3: KWS              (220 NPs)");
        eprintln!("    0x0215  Slot 4: ECG              ( 96 NPs)");
        eprintln!("    0x0275  Slot 5: Phase            ( 67 NPs)");
        eprintln!("    0x02B8  Slot 6: Anderson         ( 68 NPs)");
        eprintln!("    0x02FC  Slot 7: Sentinel         ( 50 NPs)");
        eprintln!("    TOTAL: 814 / 1,000 NPs (186 spare)");
        eprintln!();
        eprintln!("  See: metalForge/experiments/002_MULTI_TENANCY.md");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let address_str = &args[2];
    let np_address = usize::from_str_radix(address_str.trim_start_matches("0x"), 16)
        .expect("np_address must be a hex number like 0x00B3");

    println!("program_external() demonstration");
    println!("  Model:      {model_path}");
    println!("  NP address: 0x{np_address:04X} ({np_address} decimal)");
    println!();

    // Check for hardware
    if !std::path::Path::new("/dev/akida0").exists() {
        println!("  No hardware detected (/dev/akida0 not found).");
        println!("  This example requires a live AKD1000.");
        println!();
        println!("  Software simulation of program_external() address semantics:");
        println!("  - A model of N NPs loaded at address A occupies NPs [A, A+N)");
        println!("  - A second model at address B (B >= A+N) is independent");
        println!("  - Weights, thresholds, and state in [A, A+N) are disjoint from [B, B+M)");
        println!();
        println!("  For full hardware validation: metalForge/experiments/002_MULTI_TENANCY.md");
        return;
    }

    // Hardware path (activated when /dev/akida0 is present)
    println!("  Hardware present. Loading model at NP address 0x{np_address:04X}...");
    println!();
    println!("  TODO (Exp 002 Phase 2):");
    println!("    1. Parse model bytes from {model_path}");
    println!("    2. Call DeviceManager::discover() → open_first()");
    println!("    3. Call device.program_external(&model_bytes, np_address)");
    println!("    4. Run inference: device.infer(&test_input)");
    println!("    5. Load a second model at a different address");
    println!("    6. Verify: first model output unchanged after second model loads");
    println!();
    println!("  This test is defined in: metalForge/experiments/002_MULTI_TENANCY.md");
    println!("  Phase 2 hardware protocol — estimated 4–6 hours to complete.");
}
