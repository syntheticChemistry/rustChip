/**
 * @file system_linux.cpp
 * @brief Linux host implementation of system.h hooks required by the Akida Engine
 *
 * The Akida Engine requires the runtime to implement 4 functions declared in infra/system.h:
 * - msleep: Sleep for milliseconds
 * - time_ms: Get current time in milliseconds
 * - kick_watchdog: Service hardware watchdog (not needed on Linux host)
 * - panic: Fatal error handler
 *
 * This file provides standard Linux implementations using POSIX APIs.
 */

#include "infra/system.h"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <unistd.h>

namespace akida {
namespace system {

/**
 * Sleep for the specified number of milliseconds
 * Uses POSIX usleep() which takes microseconds
 */
void msleep(uint32_t ms) {
    usleep(ms * 1000);
}

/**
 * Get current time in milliseconds since an arbitrary epoch
 * Uses CLOCK_MONOTONIC for reliability (not affected by system time changes)
 */
uint32_t time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint32_t>((ts.tv_sec * 1000) + (ts.tv_nsec / 1000000));
}

/**
 * Service the hardware watchdog timer
 * On Linux host systems there is no hardware watchdog, so this is a no-op
 */
void kick_watchdog() {
    // No hardware watchdog on Linux host
}

/**
 * Fatal error handler - print message to stderr and abort
 * This is called by the engine when it encounters unrecoverable errors
 */
void panic(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "PANIC: ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
    abort();
}

} // namespace system
} // namespace akida
