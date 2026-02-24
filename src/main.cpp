/**
 * @file main.cpp
 * @brief Entry point for Akida C++ inference demo
 *
 * This program demonstrates how to discover Akida hardware on a Linux host
 * via PCIe and run inference using the C++ Engine API.
 *
 * Prerequisites:
 * - Akida hardware (AKD1000 or AKD1500) connected via PCIe
 * - PCIe driver (akida_dw_edma) loaded
 * - Akida Engine deployed (via 'akida engine deploy')
 * - Model converted to C++ binary (program.h/program.cpp)
 */

#include "inference.h"
#include "host/hardware_drivers.h"
#include <cstdio>
#include <vector>

int main(int argc, char* argv[]) {
    printf("======================================\n");
    printf("  Akida C++ Inference Demo\n");
    printf("======================================\n");

    // =========================================================================
    // Discover Akida devices via PCIe
    // =========================================================================
    // akida::get_drivers() returns a vector of all available Akida devices
    // connected via PCIe. This only works on Linux host with the PCIe driver
    // (akida_dw_edma) loaded.
    //
    // For embedded targets (microcontrollers), you would instead create
    // chip-specific drivers directly (e.g., BareMetalDriver for AKD1000,
    // Akd1500SpiDriver for AKD1500).

    printf("\nDiscovering Akida devices...\n");

    auto drivers = akida::get_drivers();

    if (drivers.empty()) {
        fprintf(stderr, "\nERROR: No Akida devices found!\n\n");
        fprintf(stderr, "Troubleshooting:\n");
        fprintf(stderr, "1. Check that Akida hardware is connected via PCIe\n");
        fprintf(stderr, "2. Verify PCIe driver is loaded:\n");
        fprintf(stderr, "   $ lsmod | grep akida\n");
        fprintf(stderr, "   Should show: akida_dw_edma\n");
        fprintf(stderr, "3. Check device visibility:\n");
        fprintf(stderr, "   $ akida devices\n");
        fprintf(stderr, "   Should list at least one device\n");
        fprintf(stderr, "4. Verify driver permissions:\n");
        fprintf(stderr, "   $ ls -l /dev/akida*\n");
        fprintf(stderr, "   You may need to run as root or add udev rules\n\n");
        return 1;
    }

    printf("Found %zu Akida device(s)\n", drivers.size());

    // Use the first available device
    auto& driver = drivers[0];

    printf("\nUsing device: %s\n", driver->desc());

    // =========================================================================
    // Run inference on the device
    // =========================================================================
    // The inference.cpp file contains the complete pipeline:
    // - Device programming
    // - Input preparation
    // - Inference execution
    // - Result extraction

    printf("\nStarting inference pipeline...\n");

    int result = run_inference(driver.get());

    if (result != 0) {
        fprintf(stderr, "\nInference failed with error code: %d\n", result);
        return result;
    }

    printf("\n======================================\n");
    printf("  Inference completed successfully!\n");
    printf("======================================\n");

    return 0;
}
