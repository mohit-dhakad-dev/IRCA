# Redis Memory Saturation
## Category
memory
## Symptoms
- Redis process logs `Out of memory` or `OOM command not allowed when used memory > 'maxmemory'`.
- Memory usage on the node remains above 90% for 15-30 minutes while cache hit rate drops.
- Application latency spikes and request timeouts appear during cache reads or object lookups.
- Eviction metrics show `evicted_keys` increasing rapidly and `used_memory_rss` tracking near the node limit.
- CPU is not the main issue; the service is slow because memory pressure triggers swap or process restarts.
## Diagnosis Steps
1. Check Redis process metrics for `used_memory`, `used_memory_rss`, `maxmemory`, and `evicted_keys` over the last 2 hours.
2. Confirm whether the node has hit or exceeded the configured memory ceiling and whether swap is active or OOM kill events are present in the kernel log.
3. Review recent deploys or cache configuration changes that may have increased TTLs, object size, or hot-key fanout.
4. Verify whether the failing workload is reading large blobs or repeated keys that are bypassing expiry and driving memory growth.
5. Compare memory growth with cache hit rate: if hit rate falls while eviction climbs, the issue is likely cache oversubscription rather than application logic failure.
## Root Cause
memory_cache_overgrowth
## Fix
- Reduce the maximum cache footprint by lowering `maxmemory` or by shrinking large payloads, TTLs, or hot-key retention windows.
- Reconfigure Redis to use a more aggressive eviction policy such as `allkeys-lru` or `volatile-lru` if the workload tolerates eviction.
- Split large caches onto a dedicated Redis shard or add a second node to avoid a single process exceeding memory limits.
- Remove stale entries and replay or rebuild large in-memory structures after validating that the data is safe to expire.
- After the change, monitor `used_memory_rss`, swap usage, eviction count, and cache hit rate for 30-60 minutes under the peak workload.
## Constraints
- Keep Redis `used_memory_rss` below 75% of the node's total RAM under normal traffic; leave at least 25% headroom for OS and other services.
- If `maxmemory` is set, do not allow it to exceed 80% of total node memory without checking for swap or OOM risk.
- Any cache expansion must be validated against the deployment's memory budget and backup/replication overhead.
