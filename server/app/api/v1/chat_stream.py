from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import run_agent_query_stream
from app.api.v1.chat_common import (
    ChatMessageRequest,
    MAX_MESSAGES_PER_CONVERSATION,
    MAX_STORED_DATA_ROWS,
    WARN_MESSAGES_THRESHOLD,
    bg_title_and_summary,
    resolve_chat_scope,
)
from app.core.database import async_session
from app.dependencies import get_db, get_current_user
from app.models.conversation import Conversation, Message
from app.models.folder import Folder
from app.models.user import User
from app.services.context_service import build_conversation_context, count_tokens, get_recent_files_used
from app.services.audit_log import record_audit_event_safe
from app.services.query_rephraser import rephrase_query
import structlog as _structlog

_pipeline_log = _structlog.get_logger("ai_pipeline")

router = APIRouter()

# ── Concurrency backpressure ─────────────────────────────────────────────────
# Limits simultaneous LLM requests to prevent Azure OpenAI quota exhaustion.
# Requests beyond this limit receive a 503 immediately with a Retry-After header
# instead of silently queuing for 60s then timing out.
# Tune this to ~70% of your Azure deployment's TPM concurrency capacity.
_LLM_SEMAPHORE = asyncio.Semaphore(5)
_RETRY_AFTER_SECONDS = 10


@router.post("/message/stream")
async def chat_message_stream(
    body: ChatMessageRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """True SSE streaming — tokens arrive as the LLM generates them."""
    import json as _json

    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if len(query) > 8000:
        raise HTTPException(status_code=400, detail="Query too long (max 8000 chars).")

    _ = f"chat-{uuid.uuid4().hex[:12]}"

    if body.conversation_id:
        conv = await db.get(Conversation, body.conversation_id)
        if not conv or conv.user_id != user.id:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        if conv.archived_at is not None:
            raise HTTPException(status_code=410, detail="Conversation has been deleted.")
    else:
        conv = Conversation(user_id=user.id, title=query[:100].strip())
        db.add(conv)
        await db.flush()

    msg_count_q = select(func.count(Message.id)).where(Message.conversation_id == conv.id)
    msg_count = (await db.execute(msg_count_q)).scalar() or 0

    if msg_count >= MAX_MESSAGES_PER_CONVERSATION:
        conv.archived_at = datetime.now(timezone.utc)
        new_conv = Conversation(
            user_id=user.id,
            title=f"{conv.title} (continued)",
            summary=conv.summary,
        )
        db.add(new_conv)
        await db.flush()
        if conv.summary:
            db.add(Message(
                conversation_id=new_conv.id,
                role="system",
                content=f"Previous conversation summary: {conv.summary}",
                token_count=count_tokens(conv.summary),
            ))
        conv = new_conv

    user_token_count = count_tokens(query)
    conversation_context = await build_conversation_context(conv, db)
    prior_files = await get_recent_files_used(conv.id, db)
    db.add(Message(
        conversation_id=conv.id,
        role="user",
        content=query,
        token_count=user_token_count,
    ))
    conv.updated_at = datetime.now(timezone.utc)
    conv.token_count = (conv.token_count or 0) + user_token_count
    await db.commit()

    conv_id = conv.id

    # Multi-tenancy: resolve scope BEFORE streaming so 403s return as JSON,
    # not as part of an SSE stream. body.container_id is ignored for non-admins.
    effective_container_id, user_allowed_domains = await resolve_chat_scope(
        user, body.container_id, db
    )
    user_is_admin = bool(getattr(user, "is_admin", False))

    # ── Query rephrasing: ERP-precise rewrite before it enters the agent ────
    # Rewrites the prompt into a precise analytical request; the cleaned text fully
    # replaces the original downstream. The agent still verifies every table/column
    # before use. Falls back to the original on any error/empty/runaway output, and
    # is a no-op when QUERY_REPHRASE_ENABLED is false. The ORIGINAL query is still
    # what we stored as the user message.
    #
    # The domain comes from the selected domain filter (the folder picker's
    # domain_tag) — structured metadata, NOT inferred by the LLM — and is appended
    # to the rewrite deterministically.
    selected_domain: str | None = None
    if body.folder_id:
        _folder = await db.get(Folder, body.folder_id)
        selected_domain = getattr(_folder, "domain_tag", None) if _folder else None

    rephrased = await rephrase_query(
        query,
        conversation_context=conversation_context,
        domain=selected_domain,
        log_context={"user_id": str(user.id), "container_id": effective_container_id},
    )
    agent_query = rephrased.text

    # AI-pipeline log: the rephrased prompt AND the domain it was rewritten under.
    _pipeline_log.info(
        "query_rephrased",
        domain=selected_domain,
        original_query=query[:500],
        rephrased_prompt=agent_query[:1500],
        changed=rephrased.changed,
        reason=rephrased.reason,
        conversation_id=conv_id,
    )

    # ── Backpressure: reject when LLM slots are exhausted ───────────────────
    # Returns 503 immediately instead of silently queuing for 60s then timing out.
    if _LLM_SEMAPHORE.locked() and _LLM_SEMAPHORE._value == 0:
        raise HTTPException(
            status_code=503,
            detail="Server is busy. Please retry in a few seconds.",
            headers={"Retry-After": str(_RETRY_AFTER_SECONDS)},
        )

    try:
        await record_audit_event_safe(
            actor=user,
            action="chat.message_stream",
            event_type="action",
            status_code=200,
            path="/api/chat/message/stream",
            route_template="/api/chat/message/stream",
            container_id=effective_container_id,
            details={
                "conversation_id": conv_id,
                "query_preview": query[:500],
                "allowed_domains": user_allowed_domains,
            },
        )
        await db.commit()
    except Exception:
        await db.rollback()

    async def event_stream():
        yield f"data: {_json.dumps({'event': 'started', 'conversation_id': conv_id})}\n\n"

        try:
            final_payload = None

            async with asyncio.timeout(180), _LLM_SEMAPHORE:
                async for evt in run_agent_query_stream(
                    agent_query, db,
                    conversation_context=conversation_context,
                    user_id=user.id,
                    actor_email=getattr(user, "email", ""),
                    actor_role=getattr(user, "role", ""),
                    is_admin=user_is_admin,
                    allowed_domains=user_allowed_domains,
                    container_id=effective_container_id,
                    prior_files=prior_files,
                    org_id=getattr(user, "organization_id", None),
                    folder_id=body.folder_id,
                ):
                    evt_type = evt["type"]

                    if evt_type == "token":
                        yield f"data: {_json.dumps({'event': 'token', 'content': evt['content']})}\n\n"

                    elif evt_type == "thinking":
                        yield f"data: {_json.dumps({'event': 'thinking', 'tool': evt.get('tool', '')})}\n\n"

                    elif evt_type == "pipeline_step":
                        yield f"data: {_json.dumps({'event': 'pipeline_step', 'step': evt.get('step', ''), 'retrieved_files': evt.get('retrieved_files', 0), 'total_files': evt.get('total_files', 0)})}\n\n"

                    elif evt_type == "tool_result":
                        yield f"data: {_json.dumps({'event': 'tool_result', 'tool': evt.get('tool', '')})}\n\n"

                    elif evt_type == "done":
                        final_payload = evt["payload"]

            if final_payload:
                answer_text = final_payload.get("answer", "")
                full_data = final_payload.get("data", [])
                stored_data = full_data[:MAX_STORED_DATA_ROWS]
                assistant_token_count = count_tokens(answer_text)

                # Use a fresh session — the agent's DB reads (e.g. vector search)
                # may have left the request session in an aborted-transaction state.
                # A separate session is always clean and avoids that corruption.
                async with async_session() as save_db:
                    save_db.add(Message(
                        conversation_id=conv_id,
                        role="assistant",
                        content=answer_text,
                        token_count=assistant_token_count,
                        payload={
                            "data": stored_data,
                            "data_truncated": len(full_data) > MAX_STORED_DATA_ROWS,
                            "chart": final_payload.get("chart"),
                            "row_count": final_payload.get("row_count", 0),
                            "files_used": final_payload.get("files_used", []),
                            "tool_calls": final_payload.get("tool_calls", 0),
                            # SME governance payload — persisted so the panel
                            # survives a conversation reload (omitted/None when off).
                            **({"governance": final_payload["governance"]}
                               if final_payload.get("governance") else {}),
                        },
                    ))
                    upd_conv = await save_db.get(Conversation, conv_id)
                    if upd_conv:
                        upd_conv.updated_at = datetime.now(timezone.utc)
                        upd_conv.token_count = (upd_conv.token_count or 0) + assistant_token_count
                    await save_db.commit()

                final_payload["conversation_id"] = conv_id

                new_count = msg_count + 2
                if new_count >= WARN_MESSAGES_THRESHOLD:
                    final_payload["warning"] = (
                        f"This conversation has {new_count}/{MAX_MESSAGES_PER_CONVERSATION} messages. "
                        "It will auto-continue in a new thread when full."
                    )

                yield f"data: {_json.dumps({'event': 'done', 'result': final_payload})}\n\n"

                background_tasks.add_task(bg_title_and_summary, conv_id)

        except Exception:
            import traceback as _tb
            _pipeline_log.error(
                "chat_stream_unhandled_exception",
                user_id=str(user.id),
                conversation_id=str(conv_id),
                query_preview=query[:200],
                exc_info=True,
                traceback=_tb.format_exc(),
            )
            try:
                async with async_session() as err_db:
                    err_db.add(Message(
                        conversation_id=conv_id,
                        role="assistant",
                        content="Failed to process query. Please try again.",
                        token_count=count_tokens("Failed to process query. Please try again."),
                        payload={"error": True},
                    ))
                    await err_db.commit()
            except Exception:
                pass  # best-effort; yield the error event regardless

            yield (
                f"data: {_json.dumps({'event': 'error', 'detail': 'Failed to process query. Please try again.'})}"
                "\n\n"
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")
