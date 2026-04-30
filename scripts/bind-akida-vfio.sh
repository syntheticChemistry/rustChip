#!/bin/bash
# Bind AKD1000 to vfio-pci and set permissions for user-level access.
# Run via: pkexec /path/to/bind-akida-vfio.sh
#
# NOTE: For production use, this script is superseded by glowplug's sovereign
# boot (crates/akida-driver/src/glowplug.rs) and the udev rule at
# /etc/udev/rules.d/99-akida-vfio.rules. This script remains useful for
# manual one-off binding on machines without the udev rule installed.
# BDF and IOMMU_GROUP below are machine-specific — adjust for your system.
set -euo pipefail

BDF="0000:e2:00.0"
VENDOR_DEVICE="1e7c bca1"
IOMMU_GROUP=92

echo "[1/4] Loading vfio-pci module..."
modprobe vfio-pci

echo "[2/4] Registering vendor:device with vfio-pci..."
echo "$VENDOR_DEVICE" > /sys/bus/pci/drivers/vfio-pci/new_id 2>/dev/null || true

echo "[3/4] Binding $BDF to vfio-pci..."
if [ ! -L "/sys/bus/pci/devices/$BDF/driver" ]; then
    echo "$BDF" > /sys/bus/pci/drivers/vfio-pci/bind 2>/dev/null || true
fi

echo "[4/4] Setting permissions on /dev/vfio/$IOMMU_GROUP..."
if [ -e "/dev/vfio/$IOMMU_GROUP" ]; then
    chmod 666 "/dev/vfio/$IOMMU_GROUP"
    echo "Done — /dev/vfio/$IOMMU_GROUP is now user-accessible."
else
    echo "Warning: /dev/vfio/$IOMMU_GROUP not found. Check binding."
    ls -la /dev/vfio/
fi

echo ""
echo "Verify:"
ls -la /sys/bus/pci/devices/$BDF/driver 2>/dev/null || echo "  No driver bound (check dmesg)"
ls -la /dev/vfio/ 2>/dev/null
