# Deployment Health Check Misconfiguration
## Category
deploy
## Symptoms
- The rollout stalls with pods marked `unready` or `CrashLoopBackOff` for several minutes after a deploy begins.
- Readiness or liveness probes fail with `connection refused`, `HTTP 503`, or `probe failed` alerts in the pod logs.
- User traffic is diverted away from the new version but the service never reaches a healthy state, leading to partial rollout or rollback.
- Deployment dashboards show a sudden drop in successful startup count and an increase in restart loops.
- Application logs show the process started normally but never passed the readiness gate after the new revision was launched.
## Diagnosis Steps
1. Review the deployment spec and confirm the readiness and liveness probe paths, ports, and timing values for the new release.
2. Check whether the application takes longer than the configured `initialDelaySeconds` or `timeoutSeconds` to become ready after startup.
3. Verify the probe endpoint is actually returning success on the container port, not a different service or an upstream dependency that is failing.
4. Compare the new version's startup dependencies with the previous version to identify cases where DB migrations, warm-up jobs, or config changes increased boot time.
5. Confirm whether the rollout is blocked by a bad health check rather than an app crash by checking the container logs for a clean process start followed by repeated readiness failures.
## Root Cause
deploy_healthcheck_misconfiguration
## Fix
- Update the readiness probe to match the service's actual startup behavior, including a realistic initial delay and a timeout threshold that reflects cold-start latency.
- Ensure the probe path returns HTTP 200 only after dependencies are initialized, not before the service can safely accept traffic.
- Re-run the deployment with a lower startup risk rollout strategy, such as a canary or a single-instance verification step.
- If the deploy started before migrations completed, pause the rollout and fix the startup order or migration gate.
- After the fix, validate pod readiness and successful traffic acceptance for at least two stable release windows.
## Constraints
- Keep readiness `timeoutSeconds` between 1 and 5 seconds; `failureThreshold` should not exceed 3 for a normally booting service.
- `initialDelaySeconds` should be no lower than 15 seconds for services with database warm-up or cache hydration.
- Do not increase `maxUnavailable` beyond the agreed deployment tolerance without a rollback plan.
