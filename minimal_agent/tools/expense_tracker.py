from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from .base import ToolSpec


class ExpenseTrackerArgs(BaseModel):
    action: Literal["add", "list", "summary", "remove"]
    entry_type: Literal["income", "expense"] | None = None
    amount: float | None = Field(default=None, gt=0)
    category: str | None = Field(default=None, max_length=30)
    note: str | None = Field(default=None, max_length=200)
    expense_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    expense_id: str | None = Field(default=None, max_length=50)
    operation_id: str | None = Field(default=None, max_length=100)
    confirmed: bool = False


def _needs_confirmation(args: ExpenseTrackerArgs) -> bool:
    return args.action in {"add", "remove"}


def _handle(args: ExpenseTrackerArgs, *, session_state: dict) -> dict:
    entries = session_state.setdefault("expenses", [])
    target_date = args.expense_date

    def matches_date(item: dict) -> bool:
        return target_date is None or item["date"] == target_date

    selected_entries = [item for item in entries if matches_date(item)]
    if args.action == "list":
        return {"action": "list", "entries": selected_entries, "count": len(selected_entries), "date": target_date}
    if args.action == "summary":
        income_total = round(sum(item["amount"] for item in selected_entries if item.get("entry_type", "expense") == "income"), 2)
        expense_total = round(sum(item["amount"] for item in selected_entries if item.get("entry_type", "expense") == "expense"), 2)
        by_category: dict[str, dict[str, float]] = {}
        for item in selected_entries:
            category = item["category"]
            kind = item.get("entry_type", "expense")
            category_totals = by_category.setdefault(category, {"income": 0.0, "expense": 0.0, "net_change": 0.0})
            category_totals[kind] = round(category_totals[kind] + item["amount"], 2)
            category_totals["net_change"] = round(category_totals["income"] - category_totals["expense"], 2)
        return {
            "action": "summary",
            "date": target_date,
            "income_total": income_total,
            "expense_total": expense_total,
            "net_change": round(income_total - expense_total, 2),
            "by_category": by_category,
            "count": len(selected_entries),
        }
    if args.action == "add":
        if args.amount is None or not args.category or args.entry_type is None:
            raise ValueError("add 需要 entry_type、amount 和 category")
        expense_date = args.expense_date or date.today().isoformat()
        try:
            date.fromisoformat(expense_date)
        except ValueError as exc:
            raise ValueError("expense_date 不是合法日期") from exc
        item = {
            "id": args.expense_id or f"entry-{len(entries) + 1}",
            "entry_type": args.entry_type,
            "amount": round(args.amount, 2),
            "category": args.category,
            "note": args.note or "",
            "date": expense_date,
        }
        entries.append(item)
        return {"action": "add", "entry": item, "count": len(entries)}
    if not args.expense_id:
        raise ValueError("remove 需要 expense_id")
    removed = [item for item in entries if item["id"] == args.expense_id]
    if not removed:
        raise ValueError(f"账目不存在: {args.expense_id}")
    entries[:] = [item for item in entries if item["id"] != args.expense_id]
    return {"action": "remove", "removed": removed, "count": len(entries)}


def expense_tracker_tool() -> ToolSpec:
    return ToolSpec(
        "expense_tracker",
        "记录、查询和汇总当前会话的收入与支出；summary 返回收入、支出和净变化，可按 expense_date 过滤",
        ExpenseTrackerArgs,
        _handle,
        side_effect="write",
        requires_confirmation=True,
        confirmation_check=_needs_confirmation,
    )
