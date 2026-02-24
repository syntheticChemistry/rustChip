/**
 * @file inference.h
 * @brief Header for Akida inference pipeline
 */

#ifndef INFERENCE_H
#define INFERENCE_H

#include "api/akida/hardware_driver.h"

/**
 * Run inference on an Akida device using the hardware driver
 *
 * This function implements the complete inference pipeline:
 * 1. Create hardware device from driver
 * 2. Program the device with the neural network model
 * 3. Prepare input tensor (with dense-to-sparse conversion if needed)
 * 4. Enqueue input for inference
 * 5. Fetch inference results
 * 6. Dequantize output if activation is enabled
 * 7. Extract and display results
 *
 * @param driver Pointer to the hardware driver (from akida::get_drivers())
 * @return 0 on success, non-zero on error
 */
int run_inference(akida::HardwareDriver* driver);

#endif // INFERENCE_H
