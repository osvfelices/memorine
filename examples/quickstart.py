"""
Memorine quickstart.
Run this to see how it works in under 30 seconds.
"""

from memorine import Mind

# Create a mind for an agent called "demo"
brain = Mind("demo", db_path="/tmp/memorine_demo.db")

# -- Learn some facts --
print("Learning facts...")
brain.learn("Python 3.12 has better error messages than 3.11")
brain.learn("Our API rate limit is 100 requests per minute", category="infra")
brain.learn("The deploy script lives at /opt/deploy/run.sh", category="ops")

# -- Contradiction detection --
print("\nTesting contradiction detection...")
fid, contradictions = brain.learn("Our API rate limit is 200 requests per minute", category="infra")
if contradictions:
    print(f"  Caught contradiction: '{contradictions[0]['fact']}'")
    print(f"  Similarity: {contradictions[0]['similarity']}")

# -- Recall --
print("\nRecalling 'deploy script'...")
results = brain.recall("deploy script")
for r in results:
    print(f"  Found: {r['fact']} (weight: {r['effective_weight']})")

# -- Events with causal chains --
print("\nLogging events...")
e1 = brain.log("DNS timeout on api.example.com", tags=["dns", "error"])
e2 = brain.log("Health check failed", caused_by=e1, tags=["health"])
e3 = brain.log("Auto-scaled to 3 replicas", caused_by=e2, tags=["scaling"])

chain = brain.why(e3)
print("  Causal chain for auto-scaling:")
for step in chain:
    print(f"    -> {step['event']}")

# -- Procedures that learn --
print("\nRunning a procedure...")
with brain.procedure("deploy_production") as run:
    run.step("run tests", success=True)
    run.step("build container", success=True)
    run.step("push to registry", success=False, error="auth token expired")

advice = brain.anticipate("deploy production")
if advice["errors_to_avoid"]:
    print("  Warning from past runs:")
    for err in advice["errors_to_avoid"]:
        print(f"    Step '{err['step']}' failed with: {err['error']}")

# -- Cognitive profile --
print("\nCognitive profile:")
print(brain.profile())

# Clean up
import os
os.unlink("/tmp/memorine_demo.db")
