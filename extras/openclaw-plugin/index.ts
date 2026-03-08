import { spawn } from "node:child_process";
import type { OpenClawPluginApi, AnyAgentTool, OpenClawPluginToolContext } from "openclaw/plugin-sdk";
import { Type } from "@sinclair/typebox";

// MCP JSON-RPC client over stdio
class MemorineClient {
  private proc: ReturnType<typeof spawn> | null = null;
  private buffer = "";
  private pending = new Map<number, { resolve: (v: any) => void; reject: (e: Error) => void }>();
  private nextId = 1;
  private pythonPath: string;

  constructor(pythonPath = "python3") {
    this.pythonPath = pythonPath;
  }

  private ensureRunning() {
    if (this.proc && !this.proc.killed) return;

    this.proc = spawn(this.pythonPath, ["-m", "memorine.mcp_server"], {
      stdio: ["pipe", "pipe", "pipe"],
    });

    this.proc.stdout!.on("data", (chunk: Buffer) => {
      this.buffer += chunk.toString();
      this.processBuffer();
    });

    this.proc.on("exit", () => {
      this.proc = null;
      // Reject all pending
      for (const [, p] of this.pending) {
        p.reject(new Error("Memorine process exited"));
      }
      this.pending.clear();
    });
  }

  private processBuffer() {
    while (true) {
      const headerEnd = this.buffer.indexOf("\r\n\r\n");
      if (headerEnd === -1) break;

      const header = this.buffer.slice(0, headerEnd);
      const match = header.match(/Content-Length:\s*(\d+)/i);
      if (!match) {
        this.buffer = this.buffer.slice(headerEnd + 4);
        continue;
      }

      const length = parseInt(match[1], 10);
      const bodyStart = headerEnd + 4;
      if (this.buffer.length < bodyStart + length) break;

      const body = this.buffer.slice(bodyStart, bodyStart + length);
      this.buffer = this.buffer.slice(bodyStart + length);

      try {
        const msg = JSON.parse(body);
        if (msg.id != null && this.pending.has(msg.id)) {
          const p = this.pending.get(msg.id)!;
          this.pending.delete(msg.id);
          if (msg.error) {
            p.reject(new Error(msg.error.message));
          } else {
            p.resolve(msg.result);
          }
        }
      } catch {
        // skip malformed
      }
    }
  }

  private send(method: string, params: any = {}): Promise<any> {
    this.ensureRunning();
    const id = this.nextId++;
    const body = JSON.stringify({ jsonrpc: "2.0", id, method, params });
    const msg = `Content-Length: ${body.length}\r\n\r\n${body}`;

    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.proc!.stdin!.write(msg);
    });
  }

  async initialize() {
    return this.send("initialize", {});
  }

  async callTool(name: string, args: Record<string, unknown>) {
    return this.send("tools/call", { name, arguments: args });
  }

  destroy() {
    if (this.proc && !this.proc.killed) {
      this.proc.kill();
    }
  }
}

// Shared client instance per agent
const clients = new Map<string, MemorineClient>();

function getClient(ctx: OpenClawPluginToolContext): MemorineClient {
  const key = ctx.agentId ?? "default";
  if (!clients.has(key)) {
    const client = new MemorineClient();
    client.initialize().catch(() => {});
    clients.set(key, client);
  }
  return clients.get(key)!;
}

function makeTool(
  name: string,
  label: string,
  description: string,
  parameters: any,
  ctx: OpenClawPluginToolContext,
): AnyAgentTool {
  const client = getClient(ctx);
  return {
    name,
    label,
    description,
    parameters,
    async execute(_toolCallId: string, params: Record<string, unknown>) {
      // Auto-inject agent_id from context if not provided
      if (!params.agent_id && ctx.agentId) {
        params.agent_id = ctx.agentId;
      }
      const result = await client.callTool(name, params);
      const text = result?.content?.[0]?.text ?? JSON.stringify(result);
      return {
        content: [{ type: "text" as const, text }],
        details: result,
      };
    },
  } as unknown as AnyAgentTool;
}

function createMemorineTools(ctx: OpenClawPluginToolContext): AnyAgentTool[] {
  return [
    makeTool(
      "memorine_learn",
      "Learn Fact",
      "Store a new fact in memory. Automatically detects contradictions with existing knowledge.",
      Type.Object({
        agent_id: Type.Optional(Type.String({ description: "Agent name (auto-detected if omitted)" })),
        fact: Type.String({ description: "The fact to remember" }),
        category: Type.Optional(Type.String({ description: "Category (default: general)" })),
        confidence: Type.Optional(Type.Number({ description: "Confidence 0-1 (default: 1.0)" })),
        relates_to: Type.Optional(Type.String({ description: "Text of a related fact to link to" })),
      }),
      ctx,
    ),
    makeTool(
      "memorine_recall",
      "Recall Facts",
      "Search memory for facts matching a query. Results ranked by importance and recency.",
      Type.Object({
        agent_id: Type.Optional(Type.String()),
        query: Type.String({ description: "Search query" }),
        limit: Type.Optional(Type.Number({ description: "Max results (default: 5)" })),
      }),
      ctx,
    ),
    makeTool(
      "memorine_log_event",
      "Log Event",
      "Record an event that happened. Supports causal chains via caused_by.",
      Type.Object({
        agent_id: Type.Optional(Type.String()),
        event: Type.String({ description: "What happened" }),
        tags: Type.Optional(Type.String({ description: "Comma-separated tags" })),
        caused_by: Type.Optional(Type.Number({ description: "Event ID that caused this" })),
      }),
      ctx,
    ),
    makeTool(
      "memorine_events",
      "Search Events",
      "Search past events by text or tags.",
      Type.Object({
        agent_id: Type.Optional(Type.String()),
        query: Type.Optional(Type.String()),
        tags: Type.Optional(Type.String({ description: "Comma-separated tags" })),
        limit: Type.Optional(Type.Number({ description: "Max results (default: 10)" })),
      }),
      ctx,
    ),
    makeTool(
      "memorine_share",
      "Share Fact",
      "Share a fact with another agent or the whole team.",
      Type.Object({
        agent_id: Type.Optional(Type.String({ description: "Your agent name" })),
        fact: Type.String({ description: "Fact to share" }),
        to_agent: Type.Optional(Type.String({ description: "Target agent (omit = share with everyone)" })),
      }),
      ctx,
    ),
    makeTool(
      "memorine_team_knowledge",
      "Team Knowledge",
      "Get collective knowledge shared across the team.",
      Type.Object({
        agent_id: Type.Optional(Type.String()),
        limit: Type.Optional(Type.Number({ description: "Max results (default: 20)" })),
      }),
      ctx,
    ),
    makeTool(
      "memorine_profile",
      "Cognitive Profile",
      "Get a summary of everything this agent knows — facts, events, procedures, team knowledge.",
      Type.Object({
        agent_id: Type.Optional(Type.String()),
      }),
      ctx,
    ),
    makeTool(
      "memorine_anticipate",
      "Anticipate Task",
      "Predict what you'll need for a task. Returns best procedure, warnings, and past errors.",
      Type.Object({
        agent_id: Type.Optional(Type.String()),
        task: Type.String({ description: "What you're about to do" }),
      }),
      ctx,
    ),
    makeTool(
      "memorine_procedure_start",
      "Start Procedure",
      "Start tracking a procedure execution.",
      Type.Object({
        agent_id: Type.Optional(Type.String()),
        name: Type.String({ description: "Procedure name" }),
        description: Type.Optional(Type.String()),
      }),
      ctx,
    ),
    makeTool(
      "memorine_procedure_step",
      "Log Procedure Step",
      "Log a step result in a running procedure.",
      Type.Object({
        run_id: Type.Number({ description: "Run ID from procedure_start" }),
        step: Type.String({ description: "Step description" }),
        success: Type.Optional(Type.Boolean({ description: "Did the step succeed? (default: true)" })),
        error: Type.Optional(Type.String({ description: "Error message if failed" })),
      }),
      ctx,
    ),
    makeTool(
      "memorine_procedure_complete",
      "Complete Procedure",
      "Mark a procedure run as complete.",
      Type.Object({
        run_id: Type.Number({ description: "Run ID from procedure_start" }),
        success: Type.Optional(Type.Boolean({ description: "Overall success? (default: true)" })),
        error: Type.Optional(Type.String()),
      }),
      ctx,
    ),
    makeTool(
      "memorine_correct",
      "Correct Fact",
      "Correct a fact that turned out to be wrong.",
      Type.Object({
        agent_id: Type.Optional(Type.String()),
        fact_id: Type.Number({ description: "ID of the fact to correct" }),
        new_value: Type.String({ description: "The corrected fact text" }),
      }),
      ctx,
    ),
  ];
}

const TOOL_NAMES = [
  "memorine_learn", "memorine_recall", "memorine_log_event",
  "memorine_events", "memorine_share", "memorine_team_knowledge",
  "memorine_profile", "memorine_anticipate", "memorine_procedure_start",
  "memorine_procedure_step", "memorine_procedure_complete", "memorine_correct",
];

export default function register(api: OpenClawPluginApi) {
  api.registerTool(
    (ctx) => createMemorineTools(ctx),
    { names: TOOL_NAMES },
  );
}
