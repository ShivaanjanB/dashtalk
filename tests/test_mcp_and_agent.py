import asyncio
import json

import pytest
from mcp import Client

from dashtalk.agent import ask, catalogue
from dashtalk.server import create_server


def run(coro):
    return asyncio.run(coro)


def server(user, warehouse, tmp_path):
    return create_server(user, warehouse=warehouse, audit_path=tmp_path / "audit.jsonl")


def text(result):
    return json.loads("".join(part.text for part in result.content))


def test_server_exposes_three_tools_and_enforces_the_user(warehouse, tmp_path):
    async def go():
        async with Client(server("analyst_emea", warehouse, tmp_path)) as client:
            names = sorted(t.name for t in (await client.list_tools()).tools)
            ok = text(await client.call_tool("query_metric", {"dashboard": "revenue_by_product", "metric": "revenue",
                                                              "filters": {"region": "EMEA", "year": 2025}}))
            denied = text(await client.call_tool("query_metric", {"dashboard": "revenue_by_product", "metric": "revenue",
                                                                  "filters": {"region": "APAC", "year": 2025}}))
            return names, ok, denied
    names, ok, denied = run(go())
    assert names == ["describe_dashboard", "list_dashboards", "query_metric"]
    assert ok["status"] == "ok" and denied["status"] == "denied"


def scripted(*replies):
    replies = iter(replies)

    def chat(messages, tools, model):
        chat.seen.append(messages)
        return next(replies)
    chat.seen = []
    return chat


def tool_call(**arguments):
    return {"message": {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "query_metric", "arguments": arguments}}]}}


def final(content):
    return {"message": {"role": "assistant", "content": content}, "prompt_eval_count": 100, "eval_count": 20}


def test_agent_answers_through_the_tool(warehouse, tmp_path):
    chat = scripted(tool_call(dashboard="sovereign_macro", metric="inflation", filters={"country": "UK", "year": 2023}),
                    final("ANSWER: UK inflation in 2023 was 6.80%."))

    async def go():
        async with Client(server("cfo", warehouse, tmp_path)) as client:
            return await ask(client, "UK inflation in 2023?", chat=chat)
    turn = run(go())
    assert turn.outcome == "answer" and turn.model_calls == 2
    assert turn.tool_calls[0]["status"] == "ok" and turn.tool_calls[0]["rows"][0]["display"] == "6.80%"
    system = chat.seen[0][0]["content"]
    assert "sovereign_macro" in system and "corporate_credit" in system  # the catalogue routes questions
    tool_message = chat.seen[1][-1]
    assert tool_message["role"] == "tool" and json.loads(tool_message["content"])["status"] == "ok"


def test_agent_sees_denials_and_the_catalogue_only_lists_allowed_dashboards(warehouse, tmp_path):
    chat = scripted(tool_call(dashboard="corporate_credit", metric="revenue", filters={"company": "microsoft"}),
                    final("REFUSED: You don't have access to corporate credit metrics."))

    async def go():
        async with Client(server("contractor", warehouse, tmp_path)) as client:
            return await ask(client, "Microsoft revenue?", chat=chat)
    turn = run(go())
    assert turn.outcome == "refused" and turn.tool_calls[0]["status"] == "denied"
    assert "corporate_credit" not in chat.seen[0][0]["content"]


@pytest.mark.parametrize("reply, outcome", [("CLARIFY: Which Apple?", "clarify"), ("The answer is 5", "unformatted"),
                                            ("answer: lower case still counts", "answer")])
def test_outcome_is_read_from_the_reply(warehouse, tmp_path, reply, outcome):
    async def go():
        async with Client(server("cfo", warehouse, tmp_path)) as client:
            return await ask(client, "q", chat=scripted(final(reply)))
    assert run(go()).outcome == outcome


def test_agent_gives_up_politely_after_too_many_steps(warehouse, tmp_path):
    loop = [tool_call(dashboard="sovereign_macro", metric="inflation", filters={"country": "UK"})] * 10

    async def go():
        async with Client(server("cfo", warehouse, tmp_path)) as client:
            return await ask(client, "q", chat=scripted(*loop))
    turn = run(go())
    assert turn.outcome == "clarify" and turn.model_calls == 6


def test_catalogue_text():
    text_ = catalogue({"dashboards": []}, [{"status": "ok", "dashboard": "d", "title": "D", "description": "x.",
                                            "grain": "g", "your_access": "all rows",
                                            "metrics": [{"name": "m", "label": "M", "unit": "usd", "synonyms": ["s"]}],
                                            "dimensions": [{"name": "k", "description": "kd"}]}])
    assert "metric `m`: M, usd (also: s)" in text_ and "dimension `k`: kd" in text_
    assert "can't see any dashboards" in catalogue({"dashboards": []}, [])
