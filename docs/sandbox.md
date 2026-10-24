# Code mode and the sandbox

Code: [`src/aisyslab/sandbox/`](../src/aisyslab/sandbox/). Lab:
[`production-schedule`](labs.md#production-schedule).

Some work is computation: planning a day on three presses is search over thousands of orderings,
and a model that tries it in its head is guessing. In code mode the model writes a program and the
program does the computing. The tools become a library the program calls (`plant.schedule`,
`plant.check`, `plant.propose`), so a search that would take hundreds of tool calls is one run.

## What the sandbox gives model-written code

`run_python` runs the code in a separate process, in a fresh temporary directory with only the files
it is given:

- an empty environment: no API keys, no `PATH`, nothing from the parent process;
- the standard library only (`python -I -S`);
- limits on CPU time, file size and open files, a memory limit where the platform enforces one, and
  a wall-clock timeout;
- Python's network and process calls (`socket`, `subprocess`, `os.system`, `fork`, `exec`) replaced
  by errors;
- output cut to a bound, with the cut reported.

## What it is not

It is not a security boundary against code written to escape. A process on the same machine can
reach what Python can reach once it imports what it needs past these replacements. For code from
someone you do not trust, run the same contract in a container or a microVM (E2B, Daytona,
Firecracker); here it guards against the ordinary failures of model-written code: loops that never
end, runaway memory or output, and a script that would quietly call the internet or read the keys
in the environment. It does not hide the disk: the child runs as the same user and can open any
file that user can, a `.env` beside the code included. Never serve a tool that runs code over a
network without authentication in front of it (see [MCP](mcp.md#over-the-network)).

## A live run

On 24 October 2026, `gpt-5-mini` at low effort solved seed 0 in eight calls, for about a cent. Its
first two programs failed on its own mistakes; the third found no plan; the fourth found a
feasible plan and submitted it: 100 minutes late against a bar of 58. Given that feedback, its next
program ran a local search, and it submitted a plan 15 minutes late.
