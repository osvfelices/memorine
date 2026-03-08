"""
Multi-agent memory sharing with Memorine.
Shows how agents share knowledge and learn from each other.
"""

from memorine import Mind

DB = "/tmp/memorine_multi.db"

# Two agents working on the same project
scout = Mind("scout", db_path=DB)
builder = Mind("builder", db_path=DB)

# Scout discovers things and shares them
print("Scout is researching...")
scout.learn("Target site uses Cloudflare CDN", category="recon")
scout.learn("Origin IP is 104.21.45.67", category="recon")
scout.share("Target site uses Cloudflare CDN")
scout.share("Origin IP is 104.21.45.67")

# Builder picks up shared knowledge
print("\nBuilder checks team knowledge...")
shared = builder.shared_with_me()
for fact in shared:
    print(f"  From {fact['from_agent']}: {fact['fact']}")

# Builder can also recall scout's shared facts directly
results = builder.recall("cloudflare")
print(f"\nBuilder recalls 'cloudflare': {results[0]['fact']}")

# Both agents see team knowledge
team = scout.team_knowledge()
print(f"\nTeam knowledge: {len(team)} shared facts")

# Clean up
import os
os.unlink(DB)
