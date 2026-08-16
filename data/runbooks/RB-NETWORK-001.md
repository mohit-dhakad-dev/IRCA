# Ingress Connection Queue Saturation
## Category
network
## Symptoms
- HTTP 502/503 responses rise sharply during peak traffic with latency increasing from under 200 ms to over 1 s.
- Ingress logs show `connection reset by peer`, `upstream connect error`, or repeated `too many open files` style errors.
- Network dashboards show elevated `tcp_established`, `listen_queue`, and `SYN backlog` values near the service limit.
- Packet loss or retransmission rates climb while frontend CPU remains moderate and backend compute is stable.
- Alerts fire for `load_balancer_backend_connection_limit` or `ingress_max_connections_exceeded`.
## Diagnosis Steps
1. Check the ingress or load balancer connection metrics for established connection counts, queue depth, and rejected connection rate over the last hour.
2. Compare those values against the per-node or per-process connection limits for the ingress service and confirm whether the queue is near exhaustion.
3. Inspect backend health checks and upstream connection reuse settings to determine whether connections are not being released quickly enough.
4. Verify whether a traffic burst, failed upstreams, or an increased keep-alive duration caused the connection churn.
5. Confirm the failure is network-layer saturation rather than app-layer logic by checking that errors are concentrated at connection setup and upstream handoff rather than application-specific business logic.
## Root Cause
network_ingress_queue_exhaustion
## Fix
- Reduce request concurrency or enable connection pooling reuse so the ingress does not hold excess sockets open.
- Scale out ingress replicas or add capacity to the load balancer tier before the queue reaches the configured maximum.
- Lower keep-alive duration or tune proxy `max_connections` and backlog settings to match the traffic profile.
- Remove or fix slow upstreams that hold connections open and prevent the ingress from draining properly.
- After mitigation, watch `connection_rejected_total`, queue depth, and upstream latency for the next 30 minutes.
## Constraints
- Keep backend connection count below 70-80% of the ingress process or LB connection ceiling under normal load.
- If a service has a `max_connections` limit of 500, avoid sustained operation above 400 without scaling or tuning.
- Any tuning to backlog values should stay within vendor-recommended ranges and be tested before production rollout.
