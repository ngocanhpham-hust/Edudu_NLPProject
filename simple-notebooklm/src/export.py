from pathlib import Path
from typing import Literal

from pydantic import BaseModel

ExportFormat = Literal["text", "md", "json"]


def _sources(model: BaseModel) -> str:
    citations = getattr(model, "citations", [])
    if not citations:
        return ""
    lines = ["", "## Nguồn"]
    for citation in citations:
        lines.append(f"- [{citation.source_marker}] {citation.filename}, trang {citation.page}")
    return "\n".join(lines)


def _to_markdown(model: BaseModel) -> str:
    name = model.__class__.__name__
    if name == "RagAnswer":
        return f"# Trả lời\n\n{model.answer}\n{_sources(model)}\n"
    if name == "Summary":
        points = "\n".join(f"- {p}" for p in model.key_points)
        return f"# Tóm tắt\n\n{model.summary}\n\n## Ý chính\n{points}\n{_sources(model)}\n"
    if name == "QuizSet":
        blocks = ["# Quiz"]
        for idx, item in enumerate(model.items, 1):
            blocks.append(f"\n## Câu {idx}. {item.question}")
            for opt_idx, option in enumerate(item.options):
                marker = "✓" if opt_idx == item.correct_index else "-"
                blocks.append(f"{marker} {option}")
            blocks.append(f"\nGiải thích: {item.explanation}")
        blocks.append(_sources(model))
        return "\n".join(blocks) + "\n"
    if name == "FlashcardSet":
        blocks = ["# Flashcards"]
        for idx, card in enumerate(model.cards, 1):
            blocks.append(f"\n## Thẻ {idx}: {card.front}\n\n{card.back}")
            if card.hint:
                blocks.append(f"\nGợi ý: {card.hint}")
        blocks.append(_sources(model))
        return "\n".join(blocks) + "\n"
    return model.model_dump_json(indent=2)


def export(model: BaseModel, *, fmt: ExportFormat = "text", output: Path | None = None):
    if fmt == "json":
        text = model.model_dump_json(indent=2) + "\n"
    elif fmt in {"text", "md"}:
        text = _to_markdown(model)
    else:
        raise ValueError(f"Unknown fmt '{fmt}'. Expected 'text'|'md'|'json'.")

    if output is None:
        return text
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output
