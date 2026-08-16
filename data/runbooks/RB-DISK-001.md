# Disk Full from Log Backlog
## Category
disk
## Symptoms
- The root or application filesystem reports `No space left on device` or `disk full` errors in the service logs.
- Monitoring shows `/var/log`, `/var/lib/docker`, or the application volume staying above 90% utilization for more than 20 minutes.
- Services begin failing to write telemetry, rotate logs, or create temp files; error rates rise on the app tier immediately afterward.
- Inode usage approaches saturation and the system reports `Too many open files` or inability to create new files.
- A sudden burst of log volume or ephemeral file creation often precedes the incident and coincides with increased container churn.
## Diagnosis Steps
1. Check filesystem usage for the application volume, container runtime directory, and log folder to confirm which mount is exhausted.
2. Inspect recent log rotation settings and verify whether the retention policy is too lenient or disabled for the impacted service.
3. Review application and platform logs to identify any process generating unusually large output, debug logs, or crash dumps.
4. Confirm whether the service is creating transient files or writes are being blocked by a full disk before the failure mode changes to app-level exceptions.
5. Verify the issue is storage exhaustion rather than a database or network problem by checking whether write failures are local file operations, not network or query errors.
## Root Cause
disk_log_rotation_gap
## Fix
- Rotate or prune logs immediately and clear stale files from the affected mount before bringing the service back to a healthy state.
- Enable or tighten log rotation policies with defined max size and retention count, for example 100 MB per file and 5 retained files per service.
- Reduce verbose application logging or disable noisy debug modes in production while the storage issue is active.
- Move high-volume log data to a dedicated volume or external logging path if a single filesystem is being overloaded by telemetry.
- Recheck disk and inode usage after cleanup and confirm the service can write new logs without hitting the device limit again.
## Constraints
- Keep each production filesystem below 80% utilization to leave room for growth and temporary spikes.
- If log rotation is configured, cap any single log file at 100 MB and retain no more than 5 files unless the retention policy is approved.
- Any cleanup or log redirection must not remove data required for audit or incident analysis without approval.
