# JWT Signing Key Rotation Mismatch
## Category
auth
## Symptoms
- Users begin seeing intermittent `401 Unauthorized` responses immediately after a key rotation or identity service deploy.
- Authentication logs record `signature verification failed`, `invalid token`, or `unable to verify JWT` errors for a subset of requests.
- Some clients continue working while others fail depending on which node or region served the request.
- Affected sessions often resume after restart or cache flush, indicating stale or mismatched signing metadata rather than a user credential problem.
- Error volume spikes with no corresponding backend service faults, which points toward identity verification drift.
## Diagnosis Steps
1. Check the auth service logs and token validation errors for the window immediately before and after the key rotation event.
2. Confirm whether the active signing key set changed and whether multiple nodes are using different private keys or stale public keys in JWKS.
3. Verify whether the new token issuer or key ID matches the expected `kid` in the distributed authentication layer.
4. Review the identity provider config to determine whether the rotation was applied only on one environment or one cluster while others still expect the previous key.
5. Compare token validation failures against the exact key ID in the request to confirm the mismatch is causing the 401s rather than user identity issues.
## Root Cause
auth_signing_key_mismatch
## Fix
- Restore the prior active signing key and re-enable the previous key in the provider for a controlled overlap window before rotation.
- Update the JWKS or key metadata so all auth nodes publish the same active key set and the correct key ID for new tokens.
- Roll out the identity service change to all nodes and ensure they share the same signing configuration before enabling new tokens.
- If the key rotation is intentional, force a brief overlap period of at least 24 hours and then remove the old key only after all clients refresh tokens.
- Monitor auth failure rates, JWT verification errors, and token issuance logs for 30-60 minutes after the fix.
## Constraints
- Maintain at least a 24-hour key overlap during rotation; do not remove the old key until all active sessions have been reissued or expired.
- Keep the target keyset consistent across all auth nodes and regional replicas; mismatched keys are not a safe tolerated state.
- Any public-key distribution must remain valid for the expected token lifetime and certificate retention window.
