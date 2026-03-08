# OpenClaw Integration

Memorine was built with OpenClaw in mind. This guide walks through the setup.

## Step 1: Install Memorine

```bash
pip install memorine
```

Verify it works:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | \
  python3 -c "import sys,json; m=json.dumps({'jsonrpc':'2.0','id':1,'method':'initialize','params':{}}); sys.stdout.write(f'Content-Length: {len(m)}\r\n\r\n{m}')" | \
  memorine
```

You should see a JSON response with `protocolVersion` and `serverInfo`.

## Step 2: Create the plugin

Create a directory for the plugin:

```bash
mkdir -p ~/.openclaw/extensions/memorine
```

Create three files in that directory.

**`openclaw.plugin.json`**:

```json
{
  "id": "memorine",
  "name": "Memorine",
  "description": "Human-like memory for agents.",
  "configSchema": {
    "type": "object",
    "additionalProperties": false,
    "properties": {}
  }
}
```

**`package.json`**:

```json
{
  "name": "memorine",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "@sinclair/typebox": "0.34.48"
  }
}
```

Then install the dependency:

```bash
cd ~/.openclaw/extensions/memorine
npm install
```

**`index.ts`**: Copy the plugin source from `extras/openclaw-plugin/index.ts`
in this repository.

## Step 3: Enable the plugin

```bash
openclaw plugins enable memorine
openclaw gateway restart
```

## Step 4: Allow tools for your agents

If your agents use explicit `tools.allow` lists, add the memorine tools:

```
memorine_learn, memorine_recall, memorine_log_event, memorine_events,
memorine_share, memorine_team_knowledge, memorine_profile, memorine_anticipate,
memorine_procedure_start, memorine_procedure_step, memorine_procedure_complete,
memorine_correct
```

You can do this through `openclaw config set` or by editing `openclaw.json`
directly.

## Step 5: Verify

Send a message to any agent:

```bash
openclaw agent --agent main -m "Use memorine_learn to store: 'Test fact'. Then use memorine_recall with query 'test' and tell me what you get."
```

The agent should report a successful learn and recall.

## How it works

The plugin spawns a `memorine` Python process per agent and communicates over
stdio using the MCP protocol (JSON-RPC 2.0). Each agent gets its own namespace
in the shared SQLite database, but they can share facts across namespaces using
the share and team_knowledge tools.

## Storage

By default, the database lives at `~/.memorine/memorine.db`. All agents read
and write to the same file. SQLite WAL mode handles concurrent access.
