#!/usr/bin/env bash
# Explicit recovery for this workstation's TP-Link hotspot adapter; never run at sim startup.
set -euo pipefail
if (( EUID != 0 )); then
    echo "Run this helper through pkexec or sudo; driver rebinding requires root." >&2
    exit 1
fi
IFACE=wlxec750c689f4a
UPLINK=wlxec750c68c5d8
MODULE=${1:-/home/ethan/.local/share/gear-sonic/rtl88x2bu-5ghz-recovery/88x2bu.ko}
IW=/home/ethan/.local/share/gear-sonic/wifi-diagnostics/root/usr/sbin/iw
if ! test -r "$MODULE" || ! test -x "$IW"; then
    echo "Fallback module or iw is unavailable. See docs/pico_tracking_gaps.md." >&2
    exit 1
fi
DEVICE=$(readlink -f "/sys/class/net/$IFACE/device")
USB_INTERFACE=${DEVICE##*/}
if ! [[ "$USB_INTERFACE" == 1-4:1.0 ]]; then
    echo "Unexpected target $USB_INTERFACE; refusing to touch another USB adapter." >&2
    exit 1
fi
grep -q '^PRODUCT=2357/138/' "$DEVICE/uevent"
UPLINK_DRIVER=$(readlink -f "/sys/class/net/$UPLINK/device/driver")
echo "Loading temporary 88x2bu driver for $IFACE only ($USB_INTERFACE)."
CURRENT_DRIVER=$(basename "$(readlink -f "$DEVICE/driver")")
if [[ "$CURRENT_DRIVER" != rtl88x2bu ]]; then
    test "$CURRENT_DRIVER" = rtw88_8822bu
    nmcli device set "$IFACE" managed no
    "$IW" reg set CH
    printf '%s' "$USB_INTERFACE" > /sys/bus/usb/drivers/rtw88_8822bu/unbind
    if test -d /sys/module/88x2bu; then
        printf '%s' "$USB_INTERFACE" > /sys/bus/usb/drivers/rtl88x2bu/bind
    elif ! /sbin/insmod "$MODULE" rtw_country_code=CH rtw_switch_usb_mode=0 rtw_power_mgnt=0 rtw_enusbss=0 rtw_ips_mode=0; then
        printf '%s' "$USB_INTERFACE" > /sys/bus/usb/drivers/rtw88_8822bu/bind
        nmcli device set "$IFACE" managed yes
        echo 'Could not load the fallback module; rebuild it for the running kernel.' >&2
        exit 1
    fi
fi
for attempt in $(seq 1 30); do
    if test -d "/sys/class/net/$IFACE"; then break; fi
    sleep 1
done
test -d "/sys/class/net/$IFACE"
ACTUAL=$(readlink -f "/sys/class/net/$IFACE/device/driver")
echo "Hotspot driver: $ACTUAL"
test "$(readlink -f "/sys/class/net/$UPLINK/device/driver")" = "$UPLINK_DRIVER"
"$IW" reg set CH
sleep 2
PHY=$(basename "$(readlink -f "/sys/class/net/$IFACE/phy80211")")
"$IW" phy "$PHY" info | sed -n '/5180.0 MHz/p; /5200.0 MHz/p; /5220.0 MHz/p; /5240.0 MHz/p'
nmcli device set "$IFACE" managed yes
nmcli --wait 25 connection up teleop-sim || {
    journalctl -b -u NetworkManager --since '-2 min' --no-pager -n 25
    journalctl -b -k --since '-2 min' --no-pager -n 25
    exit 1
}
"$IW" dev "$IFACE" info
nmcli -g GENERAL.CONNECTION,IP4.ADDRESS device show "$UPLINK"
