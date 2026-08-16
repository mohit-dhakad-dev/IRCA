# Database Connection Pool Exhaustion
## Category
db
## Symptoms
- Error rate spikes to 5-15% 500 responses on checkout and API calls.
- Application logs show repeated `ConnectionPoolTimeoutException`, `timeout waiting for connection`, or `too many connections` errors.
- Database metric `db_pool_active_connections` tracks near the configured maximum for 15-45 minutes before the incident peaks.
- Query latency increases sharply while transaction throughput remains flat or slowly declines.
- CPU on the app tier is not the primary signal; the bottleneck is waiting on database connections instead of processing work.
## Diagnosis Steps
1. Check application logs for the affected service and filter for `ConnectionPoolTimeoutException`, `SQLException: timeout waiting for connection`, and connection error counters in the last 2 hours.
2. Compare the peak error window to database pool metrics for the same service: verify active connections are at or near configured max capacity.
3. Confirm whether the database has a single shared pool used by the service, and inspect whether connection acquisition time is rising while query runtime remains stable.
4. Review recent deploys or traffic changes to determine whether the service began creating more concurrent workers, retries, or async jobs that increased pool demand.
5. Verify the issue is not caused by a database outage or query lock contention by checking whether the DB is still healthy and the error pattern is specifically "cannot acquire connection" rather than SQL failures.
## Root Cause
db_connection_pool_exhaustion
## Fix
- Reduce the concurrency of the failing worker pool or batch job so fewer connections are held at once.
- Increase the database pool size in the service configuration to a higher safe limit, then restart the service or reload the connection manager to apply the change.
- Review retry logic and backoff settings to prevent thundering-herd reconnect storms when the pool is near saturation.
- If the pool is being exhausted by a single slow query, tune that query or add indexes to shorten transaction duration and release connections faster.
- After the fix, monitor active connections for 30-60 minutes to ensure they remain below the configured high-water mark under normal traffic.
## Constraints
- max pool size should not exceed the DB server's supported connection ceiling; keep a safety margin of at least 20% below the database's hard limit.
- If the app is configured with a pool of 50, do not increase it above the DB's documented maximum without coordination with the database owner.
- Any pool increase must be validated against the database instance's CPU, memory, and connection limits before rollout.
