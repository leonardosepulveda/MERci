#!/usr/bin/env bash
# Usage: ./slurm_stats.sh 65696135
#    or: ./slurm_stats.sh 65696135_1

if [ -z "$1" ]; then
    echo "Usage: $0 <jobid or jobid_1>"
    exit 1
fi

BASEID="${1%%_*}"

sacct -j "${BASEID}" --parsable2 --units=M \
      -o JobID,State,ElapsedRaw,Elapsed,MaxRSS \
| awk -F'|' '
NR == 1 { next }  # skip header

# Use only per-task .batch lines, e.g. 65696135_1.batch, 65696135_2.batch, ...
$1 ~ /_[0-9]+\.batch$/ && $2 == "COMPLETED" {
    jobid = $1
    state = $2
    eraw  = $3
    e_hms = $4
    rss   = $5

    # ----- TIME -----
    t_sec = -1
    if (eraw ~ /^[0-9]+$/) {
        t_sec = eraw + 0
    } else if (e_hms ~ /^[0-9][0-9]*:[0-9][0-9]:[0-9][0-9]$/) {
        split(e_hms, a, ":")
        t_sec = a[1]*3600 + a[2]*60 + a[3]
    }
    if (t_sec >= 0) {
        n_time++
        sum_time += t_sec
        if (n_time == 1 || t_sec < min_time) min_time = t_sec
        if (t_sec > max_time)               max_time = t_sec
    }

    # ----- MEMORY -----
    if (rss != "") {
        unit = substr(rss, length(rss), 1)
        val  = substr(rss, 1, length(rss)-1) + 0

        if      (unit == "M") m_mib = val
        else if (unit == "G") m_mib = val * 1024
        else if (unit == "K") m_mib = val / 1024
        else                  next   # unknown unit

        n_mem++
        sum_mem += m_mib
        if (n_mem == 1 || m_mib < min_mem) min_mem = m_mib
        if (m_mib > max_mem)               max_mem = m_mib
    }
}

function hms(sec,  h,m,s) {
    h = int(sec / 3600)
    m = int((sec % 3600) / 60)
    s = sec % 60
    return sprintf("%02d:%02d:%02d", h, m, s)
}

END {
    if (n_time == 0 && n_mem == 0) {
        print "No COMPLETED .batch tasks with usable time/memory data."
        exit 1
    }

    if (n_time > 0) {
        mean_time = sum_time / n_time
        print "=== Time (Elapsed, COMPLETED .batch tasks) ==="
        print "Count:", n_time
        print "Min:  " min_time " sec (" hms(min_time) ")"
        print "Mean: " mean_time " sec (" hms(mean_time) ")"
        print "Max:  " max_time " sec (" hms(max_time) ")"
        print ""
    } else {
        print "No usable time info for COMPLETED .batch tasks."
        print ""
    }

    if (n_mem > 0) {
        mean_mem = sum_mem / n_mem
        print "=== Memory (MaxRSS, MiB, COMPLETED .batch tasks) ==="
        print "Count:", n_mem
        printf "Min:  %.2f MiB\n", min_mem
        printf "Mean: %.2f MiB\n", mean_mem
        printf "Max:  %.2f MiB\n", max_mem
    } else {
        print "No usable MaxRSS info for COMPLETED .batch tasks."
    }
}
'
