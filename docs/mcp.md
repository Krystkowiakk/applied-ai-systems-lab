# MCP: labs as servers, agents as hosts

Every lab can be served over the Model Context Protocol, and the agent loop can take its tools
from MCP servers. Code: [`src/aisyslab/mcp/`](../src/aisyslab/mcp/).

## Serve a lab

```bash
pip install -e ".[mcp]"
aisyslab mcp plant-database --seed 4                      # stdio, for a desktop host
aisyslab mcp plant-database --transport http --port 8000  # streamable HTTP at /mcp
```

The server lists the lab's tools with their JSON schemas, runs them against one seeded
environment, publishes the task as the resource `lab://<lab>/task` and puts the agent's
instructions in the server's `instructions`. A refused or invalid call comes back as a result
with `is_error` set and a hint, so the host's model can recover the same way the local agent
does. The environment's policies (read-only SQL, approval tokens, forbidden paths) still apply:
they live in the environment, not in the prompt.

To work a lab from Claude Desktop, add it to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "plant-database": {
      "command": "/path/to/applied-ai-systems-lab/.venv/bin/aisyslab",
      "args": ["mcp", "plant-database", "--seed", "4"]
    }
  }
}
```

**Which transport.** stdio starts one server process per host on the same machine; it suits a
desktop app and never a web service. Streamable HTTP is the default for anything remote or
shared, and it is where sessions and authentication belong.

## Use MCP servers from the agent loop

`McpHost` connects to several servers at once (an in-process server, a stdio command or an
HTTP URL), audits what each offers and merges their tools into one toolbox:

```python
from aisyslab.mcp.host import McpHost

with McpHost({"plant": "http://127.0.0.1:8000/mcp", "gate": stdio_params}) as host:
    for server in host.audit():          # tools and resources, before a model sees them
        print(server.prefix, server.tools, server.resources)
    toolbox = host.toolbox()             # plant__run_query, gate__train_tree, ...
```

- **Prefixes against collisions.** Two servers may both offer `search`; the host exposes
  `plant__search` and `gate__search`, so the model never has to guess.
- **Errors as data across the protocol.** An MCP error result becomes a `ToolError` with the
  original message and hint, so the toolbox reports it to the model instead of crashing.
- **Sync agent, async protocol.** The MCP client is asynchronous; the agent loop is not. The
  host keeps every session inside one long-lived task on a background event loop (what the
  SDK's cancel scopes require) and passes calls through a queue.

Run any lab's agent with its tools routed through an MCP server:

```bash
aisyslab run plant-database --solver agent --provider anthropic --via-mcp
```

## Two tools instead of a list

With tools merged from several servers, every schema in every request costs tokens and
attention. `--discover` gives the agent two tools: `search_tools` (hybrid search over tool
names and descriptions, returning exact schemas) and `call_tool`. The underlying toolbox still
validates arguments and runs the call.

The tools a task cannot finish without stay pinned in the list. In a live run without pinning,
`gpt-5-mini` found the right answer through discovered tools, wrote "Submitting now." and ended its
turn: the submit tool was not in its list, and it never searched for it. With the answer tools
pinned, the same run submitted, read the verifier's feedback and was accepted on the second try.

```bash
aisyslab run plant-database --solver agent --provider openai --via-mcp --discover
```

Tests: [`tests/test_mcp.py`](../tests/test_mcp.py) covers the protocol in-process, a stdio
subprocess end to end, prefixing and auditing across two servers, an agent solving a lab
through MCP, and discovery.
