#!/bin/bash

set -euo pipefail

usage() {
        echo "Usage: $0 {start|stop|restart}"
        exit 1
}

[ $# -ge 1 ] || usage

case "$1" in
        start)
                docker run -d \
                        --name washer \
                        --device /dev/snd/ \
                        --group-add audio \
                        -e MONITOR_MQTT_HOST=192.168.116.232 \
                        -e MONITOR_MQTT_TOPIC=washer \
                        -e MONITOR_DEVICE_INDEX=1 \
                        dhogborg/washer:latest
                ;;
        stop)
                docker stop washer
                ;;
        restart)
                "$0" stop || true
                sleep 1
                "$0" start
                ;;
        *)
                usage
                ;;
esac
