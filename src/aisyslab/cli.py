"""Command line: list labs, run one, benchmark many, read a trace, evaluate, serve."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .core.bench import summarize, to_markdown
from .core.lab import EnvOptions, Lab
from .core.model import ChatModel, ModelError
from .core.runner import RunReport, run_once
from .core.trace import read_trace, render_trace
from .labs import LABS, get_lab
from .providers import PROVIDERS, make_model
from .voice import cli as voice_cli


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aisyslab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List the labs.")
    show = sub.add_parser("show", help="Describe one lab.")
    show.add_argument("lab", choices=list(LABS))

    run = sub.add_parser("run", help="Run one lab once.")
    run.add_argument("lab", choices=list(LABS))
    run.add_argument("--solver", choices=["baseline", "agent"], default="baseline")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--quiet", action="store_true", help="Do not stream trace events.")
    _shared(run)

    bench = sub.add_parser("bench", help="Run labs over several seeds and compare solvers.")
    bench.add_argument("--labs", default="all", help="Comma-separated lab ids, or 'all'.")
    bench.add_argument("--seeds", default="0-4", help="Range '0-4' or list '0,3,7'.")
    bench.add_argument("--solvers", default="baseline", help="'baseline', 'agent' or 'baseline,agent'.")
    bench.add_argument("--out", type=Path, default=Path("runs"), help="Where to write the JSON and Markdown results.")
    bench.add_argument("--price-in", type=float, default=None, help="USD per million input tokens, for a cost column.")
    bench.add_argument("--price-out", type=float, default=None, help="USD per million output tokens.")
    _shared(bench)

    trace = sub.add_parser("trace", help="Print a JSONL trace as a timeline.")
    trace.add_argument("path", type=Path)

    ev = sub.add_parser("eval", help="Evaluate the ticket-triage prompt: score, optimise, calibrate the judge.")
    ev.add_argument("what", choices=["triage", "optimize", "judge"])
    ev.add_argument("--seed", type=int, default=0, help="Dataset seed.")
    ev.add_argument("--models", default=None, help="Comma-separated model ids to compare (with --provider).")
    ev.add_argument("--optimizer-model", default=None, help="Stronger model that rewrites the prompt.")
    ev.add_argument("--judge", action="store_true", help="Add the model judge to the graders.")
    ev.add_argument("--rounds", type=int, default=5)
    ev.add_argument("--out", type=Path, default=Path("runs/evals"))
    ev.add_argument("--provider", choices=PROVIDERS, default=None, help="Without it, offline stand-ins are used.")
    ev.add_argument("--model", default=None)
    ev.add_argument("--base-url", default=None)
    ev.add_argument("--effort", default=None)

    serve = sub.add_parser("mcp", help="Serve a lab's tools as an MCP server (stdio or streamable HTTP).")
    serve.add_argument("lab", choices=list(LABS))
    serve.add_argument("--seed", type=int, default=0)
    serve.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    svc = sub.add_parser("serve", help="Run the maintenance request service: HTTP API, workers and web UI.")
    svc.add_argument("--db", type=Path, default=Path("runs/service.db"))
    svc.add_argument(
        "--provider", choices=PROVIDERS, default=None, help="Use 'local' for a model server on this machine."
    )
    svc.add_argument("--demo", action="store_true", help="No model: an offline stand-in, to try the API and the UI.")
    svc.add_argument("--model", default=None)
    svc.add_argument("--fallback-provider", choices=PROVIDERS, default=None, help="Tried when the first model fails.")
    svc.add_argument("--fallback-model", default=None)
    svc.add_argument("--triage-model", default=None, help="A smaller model for /v1/triage (same provider).")
    svc.add_argument("--base-url", default=None)
    svc.add_argument("--effort", default=None)
    svc.add_argument("--workers", type=int, default=2)
    svc.add_argument("--host", default="127.0.0.1")
    svc.add_argument("--port", type=int, default=8000)

    web = sub.add_parser("search", help="Search the real web through the research SearchAPI (needs OPENAI_API_KEY).")
    web.add_argument("query")
    web.add_argument("--k", type=int, default=5)
    web.add_argument("--model", default="gpt-5-search-api")

    voice_cli.add_parser(sub)

    ten = sub.add_parser("tenant", help="Create a tenant of the service and print its token (shown once).")
    ten.add_argument("name")
    ten.add_argument("--budget", type=int, default=500_000, help="Token budget.")
    ten.add_argument("--db", type=Path, default=Path("runs/service.db"))

    args = parser.parse_args(argv)
    if args.command == "list":
        for lab in LABS.values():
            print(f"{lab.id:<20} {lab.title}")
        return 0
    if args.command == "show":
        _describe(get_lab(args.lab))
        return 0
    if args.command == "trace":
        print(render_trace(read_trace(args.path)))
        return 0
    if args.command == "mcp":
        return _serve_mcp(args)
    if args.command == "eval":
        from .evals.cli import run as run_eval_command

        return run_eval_command(args)
    if args.command == "tenant":
        return _tenant(args)
    if args.command == "search":
        from .research.web import OpenAISearch

        try:
            results = OpenAISearch(args.model).search(args.query, k=args.k)
        except ModelError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        for r in results:
            print(f"{r['title']}\n  {r['url']}\n  {r['snippet']}\n")
        return 0
    if args.command == "serve":
        try:
            return _serve(args)
        except ModelError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    if args.command == "voice":
        try:
            return voice_cli.run(args)
        except ModelError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    try:
        if args.command == "run":
            return _run(args)
        return _bench(args)
    except ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _shared(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", choices=PROVIDERS, default=None, help="Model provider for the agent solver.")
    p.add_argument("--model", default=None, help="Model id; each provider has a default.")
    p.add_argument(
        "--base-url", default=None, help="OpenAI-compatible server, e.g. http://localhost:11434/v1 for Ollama."
    )
    p.add_argument("--effort", default=None, help="Reasoning effort passed to the provider (e.g. low, medium, high).")
    p.add_argument("--flaky", type=float, default=0.0, help="Probability of a transient 503/429 on every API call.")
    p.add_argument(
        "--embedder",
        choices=["trigram", "openai", "local"],
        default="trigram",
        help="Vector ranking for retrieval labs.",
    )
    p.add_argument("--embedding-model", default=None, help="Embedding model id for --embedder openai/local.")
    p.add_argument(
        "--moderation",
        choices=["keywords", "openai"],
        default="keywords",
        help="Moderation for labs that screen messages: an offline keyword list, or OpenAI's moderation model.",
    )
    p.add_argument(
        "--local-model",
        default=None,
        help="A model on this machine (OpenAI-compatible server) for data that must stay local.",
    )
    p.add_argument(
        "--operator",
        choices=["auto", "ask"],
        default="auto",
        help="Who approves irreversible actions: a simulated careful operator, or you.",
    )
    p.add_argument("--trace-dir", type=Path, default=Path("runs/traces"))
    p.add_argument(
        "--via-mcp", action="store_true", help="Serve the lab's tools over MCP in-process; the agent uses the MCP host."
    )
    p.add_argument(
        "--discover", action="store_true", help="Give the agent search_tools and call_tool instead of the tool list."
    )
    p.add_argument(
        "--promptless", action="store_true", help="Run the agent without its system prompt (a tool-description eval)."
    )


def _tenant(args: argparse.Namespace) -> int:
    from .runtime.store import Store

    args.db.parent.mkdir(parents=True, exist_ok=True)
    tenant_id, token = Store(args.db).create_tenant(args.name, args.budget)
    print(f"tenant {tenant_id} ({args.name}), budget {args.budget:,} tokens")
    print(f"token (shown once, only its hash is stored): {token}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from .runtime.api import create_app
    except ImportError:
        print("error: the service needs FastAPI and uvicorn: pip install -e '.[service]'", file=sys.stderr)
        return 2
    from .runtime import ModelGateway, Route, Runtime, Store

    primary: ChatModel
    if args.demo:
        from .runtime.demo import DemoModel

        primary = DemoModel()
    elif args.provider:
        primary = make_model(args.provider, args.model, base_url=args.base_url, effort=args.effort)
    else:
        print("error: serve needs --provider (or --demo to try it without a model)", file=sys.stderr)
        return 2
    models = [primary]
    if args.fallback_provider:
        models.append(make_model(args.fallback_provider, args.fallback_model, effort=args.effort))
    triage = primary
    if args.triage_model and args.provider:
        triage = make_model(args.provider, args.triage_model, base_url=args.base_url)
    args.db.parent.mkdir(parents=True, exist_ok=True)
    store = Store(args.db)
    gateway = ModelGateway({"default": models, "triage": Route((triage, *models[1:]), cache=True)}, store)
    app = create_app(Runtime(store, gateway), workers=args.workers)
    print(
        f"Maintenance request service on http://{args.host}:{args.port}/  (models: {', '.join(m.name for m in models)})"
    )
    print(f"Create a token with: aisyslab tenant <name> --db {args.db}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def _lab(args: argparse.Namespace, lab_id: str) -> Lab:
    from .mcp.route import routed

    return routed(get_lab(lab_id), via_mcp=args.via_mcp, discover=args.discover, promptless=args.promptless)


def _serve_mcp(args: argparse.Namespace) -> int:
    from .mcp.server import http_app, lab_server, serve_stdio

    served = lab_server(get_lab(args.lab), args.seed)
    if args.transport == "stdio":
        import anyio

        anyio.run(serve_stdio, served)
        return 0
    import uvicorn

    print(f"MCP server for '{args.lab}' at http://{args.host}:{args.port}/mcp", file=sys.stderr)
    uvicorn.run(http_app(served, host=args.host), host=args.host, port=args.port)
    return 0


def _model(args: argparse.Namespace, solvers: Sequence[str]) -> ChatModel | None:
    if "agent" not in solvers:
        return None
    if args.provider is None:
        raise ModelError("The agent solver needs --provider (openai, anthropic or local).")
    return make_model(args.provider, args.model, base_url=args.base_url, effort=args.effort)


def _options(args: argparse.Namespace) -> EnvOptions:
    embedder = None
    if args.embedder == "openai":
        from .core.retrieval import OpenAICompatibleEmbedder

        embedder = OpenAICompatibleEmbedder(args.embedding_model or "text-embedding-3-small")
    elif args.embedder == "local":
        from .core.retrieval import OpenAICompatibleEmbedder
        from .providers import DEFAULT_LOCAL_URL

        embedder = OpenAICompatibleEmbedder(
            args.embedding_model or "text-embedding-nomic-embed-text-v1.5", base_url=args.base_url or DEFAULT_LOCAL_URL
        )
    moderator = None
    if args.moderation == "openai":
        from .guardrails.moderation import OpenAIModeration

        moderator = OpenAIModeration()
    local_model = (
        make_model("local", args.local_model, base_url=args.base_url if args.provider == "local" else None)
        if args.local_model
        else None
    )
    return EnvOptions(flaky=args.flaky, embedder=embedder, moderator=moderator, local_model=local_model)


def _configure(args: argparse.Namespace) -> Callable[[Any], None] | None:
    if args.operator != "ask":
        return None

    def ask(request: dict[str, Any]) -> tuple[bool, str]:
        print("\n--- approval requested ---")
        print(json.dumps(request, indent=2, ensure_ascii=False, default=str))
        answer = input("Approve? [y/N] ").strip().lower()
        return (True, "Approved by the operator.") if answer == "y" else (False, "Denied by the operator.")

    def configure(env: Any) -> None:
        if hasattr(env, "operator"):
            env.operator = ask

    return configure


def _run(args: argparse.Namespace) -> int:
    lab = _lab(args, args.lab)
    model = _model(args, [args.solver])
    report = run_once(
        lab,
        args.solver,
        args.seed,
        model=model,
        options=_options(args),
        trace_dir=args.trace_dir,
        echo=None if args.quiet else print,
        configure_env=_configure(args),
    )
    _print_report(report)
    return 0 if report.accepted else 1


def _bench(args: argparse.Namespace) -> int:
    ids = list(LABS) if args.labs == "all" else [i.strip() for i in args.labs.split(",")]
    labs = [_lab(args, i) for i in ids]
    solvers = [s.strip() for s in args.solvers.split(",")]
    model = _model(args, solvers)
    options = _options(args)
    reports: list[RunReport] = []
    for lab in labs:
        for solver in solvers:
            for seed in _seeds(args.seeds):
                report = run_once(
                    lab,
                    solver,
                    seed,
                    model=model,
                    options=options,
                    trace_dir=args.trace_dir,
                    configure_env=_configure(args),
                )
                reports.append(report)
                mark = "ok " if report.accepted else "-- "
                print(
                    f"{mark}{lab.id:<20} {solver:<8} seed {seed:<3} {report.status:<14} {report.tool_calls:>3} tool calls  {report.input_tokens:>8,} tok in"
                )
    table = to_markdown(summarize(reports), price_in=args.price_in, price_out=args.price_out)
    print("\n" + table)
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (args.out / f"bench-{stamp}.json").write_text(json.dumps([asdict(r) for r in reports], indent=2), encoding="utf-8")
    (args.out / f"bench-{stamp}.md").write_text(table + "\n", encoding="utf-8")
    print(f"\nSaved {args.out / f'bench-{stamp}.json'} and .md")
    return 0


def _seeds(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s.strip()]


def _describe(lab: Lab) -> None:
    print(f"{lab.title}\n\n{lab.summary}\n\nTechniques:")
    for t in lab.techniques:
        print(f"  - {t}")
    if lab.origin:
        print(f"\nOrigin: {lab.origin}")


def _print_report(r: RunReport) -> None:
    verdict = "ACCEPTED" if r.accepted else "not accepted"
    score = f", score {r.score:.2f}" if r.score is not None and not r.accepted else ""
    print(f"\n{r.lab} / {r.solver} / seed {r.seed}: {verdict}{score} ({r.status})")
    print(
        f"  submissions {r.submissions}, API calls {r.api_calls}, tool calls {r.tool_calls}, model calls {r.model_calls}, retries {r.retries}"
    )
    if r.model_calls:
        print(f"  tokens in {r.input_tokens:,} (cached {r.cached_input_tokens:,}), out {r.output_tokens:,}")
    print(f"  {r.seconds:.2f}s, trace: {r.trace_path}")
    if r.error:
        print(f"  error: {r.error}")


if __name__ == "__main__":
    raise SystemExit(main())
