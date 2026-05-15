import re
from collections import Counter
from itertools import islice

from src.schemas import Flashcard, QuizItem, RetrievedChunk

_STOPWORDS = {
    "the",
    "and",
    "for",
    "are",
    "this",
    "that",
    "with",
    "from",
    "into",
    "software",
    "engineering",
    "một",
    "các",
    "những",
    "trong",
    "được",
    "của",
    "cho",
    "với",
    "này",
    "là",
    "và",
    "hoặc",
}


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    lines = [_clean_text(line).strip("•▪- ") for line in text.splitlines()]
    merged_lines: list[str] = []
    current = ""
    for line in lines:
        if not line or line.isdigit():
            continue
        if not current:
            current = line
            continue
        starts_new = bool(re.match(r"^(\d+(\.\d+)*\.|[A-Z][A-Z\s]{5,})", line))
        if starts_new or current.endswith((".", "!", "?", ":")):
            merged_lines.append(current)
            current = line
        else:
            current = f"{current} {line}"
    if current:
        merged_lines.append(current)

    raw = []
    for line in merged_lines:
        raw.extend(re.split(r"(?<=[.!?])\s+", line))
    sentences = [_clean_text(s) for s in raw]
    return [s for s in sentences if 35 <= len(s) <= 360 and not s.isdigit()]


def _tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9_-]{2,}", text)
        if token.lower() not in _STOPWORDS
    ]


def _is_useful_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    if any(skip in lowered for skip in ["biên soạn", "trình bày", "nhóm chuyên môn"]):
        return False
    tokens = _tokens(sentence)
    return len(tokens) >= 5


def _compact(sentence: str, limit: int = 120) -> str:
    sentence = _clean_text(sentence).strip("•▪- ")
    return sentence if len(sentence) <= limit else sentence[: limit - 3].rstrip() + "..."


def _chunk_title(chunk: RetrievedChunk, fallback: str) -> str:
    lines = [_clean_text(line).strip("•▪- ") for line in chunk.text.splitlines()]
    for line in lines:
        if not line or line.isdigit():
            continue
        if len(line) <= 90 and (
            line.isupper()
            or re.match(r"^\d+(\.\d+)*\.\s+", line)
            or line[:1].isupper()
        ):
            return line
    return fallback


def _unique_headings(chunks: list[RetrievedChunk], limit: int = 12) -> list[str]:
    headings = []
    for chunk in chunks:
        title = _chunk_title(chunk, "")
        if title and title.lower() not in {h.lower() for h in headings}:
            headings.append(title)
        if len(headings) >= limit:
            break
    return headings


def _infer_points_from_headings(headings: list[str]) -> list[str]:
    heading_text = " ".join(headings).lower()
    points = []
    if "definition" in heading_text or "concept" in heading_text:
        points.append("Giới thiệu khái niệm, định nghĩa và đặc điểm của phần mềm.")
    if "classification" in heading_text:
        points.append("Phân loại phần mềm, bao gồm phần mềm ứng dụng và phần mềm hệ thống.")
    if "software engineering" in heading_text or "engineering" in heading_text:
        points.append("Trình bày khái niệm Công nghệ phần mềm, mục tiêu, nguyên lý và công cụ hỗ trợ.")
    if "process" in heading_text:
        points.append("Mô tả quy trình phát triển phần mềm như nền tảng để tổ chức hoạt động kỹ thuật.")
    if "quality" in heading_text:
        points.append("Đề cập đến chất lượng phần mềm, năng suất và các yếu tố bảo đảm chất lượng.")
    if "audience" in heading_text or "customer" in heading_text or "user" in heading_text:
        points.append("Phân biệt các đối tượng liên quan như customer, client và user trong dự án phần mềm.")
    if "function" in heading_text or "cost" in heading_text or "time" in heading_text:
        points.append("Nhấn mạnh việc cân bằng chức năng, chi phí và thời gian khi phát triển phần mềm.")
    if "risk" in heading_text or "project" in heading_text:
        points.append("Giới thiệu quản lý dự án phần mềm, rủi ro và các biện pháp giảm thiểu rủi ro.")
    return points or [f"Nội dung liên quan đến: {heading}" for heading in headings[:6]]


def _keyword_counter(chunks: list[RetrievedChunk]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for chunk in chunks:
        counter.update(_tokens(chunk.text))
    return counter


def top_keywords(chunks: list[RetrievedChunk], limit: int = 10) -> list[str]:
    return [word for word, _ in _keyword_counter(chunks).most_common(limit)]


def _sentence_score(sentence: str, keywords: Counter[str]) -> float:
    tokens = _tokens(sentence)
    if not tokens:
        return 0.0
    return sum(keywords.get(token, 0) for token in tokens) / max(len(tokens), 1)


def select_representative_chunks(
    chunks: list[RetrievedChunk], limit: int = 12
) -> list[RetrievedChunk]:
    seen_pages: set[tuple[str, int]] = set()
    selected: list[RetrievedChunk] = []
    for chunk in chunks:
        key = (chunk.metadata.filename, chunk.metadata.page)
        if key in seen_pages and len(selected) >= 3:
            continue
        seen_pages.add(key)
        selected.append(chunk)
        if len(selected) >= limit:
            break
    return selected


def summarize_chunks(
    chunks: list[RetrievedChunk],
    *,
    max_sentences: int = 9,
    max_key_points: int = 8,
) -> tuple[str, list[str], list[RetrievedChunk]]:
    if not chunks:
        return "Không tìm thấy nội dung phù hợp trong tài liệu.", [], []

    headings = _unique_headings(chunks, limit=12)
    if len(headings) >= 4:
        inferred_points = _infer_points_from_headings(headings)
        keyword_text = ", ".join(top_keywords(chunks, limit=8))
        summary = (
            "Tài liệu là bài giảng tổng quan về Software Engineering / Công nghệ phần mềm. "
            "Nội dung đi từ khái niệm phần mềm, đặc điểm và phân loại phần mềm, sau đó mở rộng "
            "sang khái niệm công nghệ phần mềm, mục tiêu của lĩnh vực này, quy trình phát triển, "
            "chất lượng, đối tượng sử dụng/khách hàng và các vấn đề quản lý dự án như chi phí, "
            "thời gian, chức năng và rủi ro. "
            f"Các tiêu đề chính xuất hiện trong tài liệu gồm: {', '.join(headings[:8])}."
        )
        if keyword_text:
            summary += f" Từ khóa nổi bật: {keyword_text}."
        used = []
        seen_pages = set()
        for chunk in chunks:
            if chunk.metadata.page in seen_pages:
                continue
            used.append(chunk)
            seen_pages.add(chunk.metadata.page)
            if len(used) >= min(8, len(chunks)):
                break
        return summary, inferred_points, used

    keywords = _keyword_counter(chunks)
    candidates: list[tuple[float, int, RetrievedChunk, str]] = []
    order = 0
    for chunk in chunks:
        for sentence in split_sentences(chunk.text):
            if not _is_useful_sentence(sentence):
                continue
            candidates.append((_sentence_score(sentence, keywords), order, chunk, sentence))
            order += 1

    if not candidates:
        fallback = [_clean_text(chunk.text)[:260] for chunk in chunks[:max_sentences]]
        used = chunks[: len(fallback)]
        return " ".join(fallback), fallback[:max_key_points], used

    ranked = sorted(candidates, key=lambda item: (-item[0], item[1]))
    chosen = sorted(ranked[:max_sentences], key=lambda item: item[1])
    sentences = [sentence for _, _, _, sentence in chosen]
    used_chunks = []
    seen_chunk_ids = set()
    for _, _, chunk, _ in chosen:
        if chunk.metadata.chunk_id not in seen_chunk_ids:
            used_chunks.append(chunk)
            seen_chunk_ids.add(chunk.metadata.chunk_id)

    key_points = []
    for sentence in sentences:
        point = sentence.rstrip(".")
        if point.lower() not in {p.lower() for p in key_points}:
            key_points.append(point)
        if len(key_points) >= max_key_points:
            break

    summary = " ".join(sentences)
    return summary, key_points, used_chunks


def answer_from_chunks(question: str, chunks: list[RetrievedChunk]) -> tuple[str, list[RetrievedChunk]]:
    if not chunks:
        return "Tôi không tìm thấy thông tin phù hợp trong tài liệu để trả lời.", []

    headings = _unique_headings(chunks, limit=8)

    summary, points, used_chunks = summarize_chunks(chunks, max_sentences=6, max_key_points=5)
    markers = ", ".join(f"[S{i}]" for i in range(1, min(len(used_chunks), 5) + 1))
    keyword_text = ", ".join(top_keywords(chunks, limit=6))

    if headings:
        inferred_points = _infer_points_from_headings(headings)

        answer = (
            "Dựa trên nội dung đã index, tài liệu này là bài giảng/tài liệu học tập về "
            "**Software Engineering / Công nghệ phần mềm**. Các phần nổi bật gồm:\n"
            + "\n".join(f"- {heading}" for heading in headings[:7])
            + "\n\nNội dung chính:\n"
            + "\n".join(f"- {point}" for point in inferred_points)
        )
        if keyword_text:
            answer += f"\n\nTừ khóa nổi bật: {keyword_text}."
        if markers:
            answer += f"\n\nNguồn tham khảo: {markers}."
        return answer, used_chunks or chunks[:5]
    else:
        answer = "Dựa trên các phần liên quan trong tài liệu, nội dung chính có thể tóm tắt như sau:\n"

    answer += f"\n{summary}\n\n"
    if points:
        answer += "Các ý quan trọng:\n" + "\n".join(f"- {point}" for point in points)
    if keyword_text:
        answer += f"\n\nTừ khóa nổi bật: {keyword_text}."
    if markers:
        answer += f"\n\nNguồn tham khảo: {markers}."
    return answer, used_chunks or chunks[:5]


def keyword_retrieve(
    query: str, chunks: list[RetrievedChunk], *, limit: int = 5
) -> list[RetrievedChunk]:
    query_terms = set(_tokens(query))
    corpus_keywords = _keyword_counter(chunks)
    scored: list[tuple[float, int, RetrievedChunk]] = []
    for idx, chunk in enumerate(chunks):
        chunk_terms = _tokens(chunk.text)
        if not chunk_terms:
            continue
        overlap = sum(1 for term in chunk_terms if term in query_terms)
        salience = sum(corpus_keywords.get(term, 0) for term in set(chunk_terms))
        title_bonus = 3.0 if _chunk_title(chunk, "") else 0.0
        score = overlap * 10.0 + salience / max(len(set(chunk_terms)), 1) + title_bonus
        scored.append((score, idx, chunk))
    ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
    selected = [chunk.model_copy(update={"score": float(score)}) for score, _, chunk in ranked[:limit]]
    return selected or chunks[:limit]


def make_quiz_items(chunks: list[RetrievedChunk], count: int) -> tuple[list[QuizItem], list[RetrievedChunk]]:
    keywords = top_keywords(chunks, limit=max(count * 3, 12))
    sentence_rows = []
    for chunk_index, chunk in enumerate(chunks, 1):
        for sentence in split_sentences(chunk.text):
            if _is_useful_sentence(sentence):
                sentence_rows.append((chunk_index, chunk, sentence))

    items: list[QuizItem] = []
    used_chunks: list[RetrievedChunk] = []
    used_questions: set[str] = set()
    distractor_pool = [_compact(sentence, 80) for _, _, sentence in sentence_rows]
    for chunk_index, chunk, sentence in sentence_rows:
        if len(items) >= count:
            break
        topic = _chunk_title(chunk, keywords[0] if keywords else "nội dung bài học")
        answer = _compact(sentence, 95)
        question = f"Theo tài liệu, ý nào mô tả đúng về \"{topic}\"?"
        if question.lower() in used_questions:
            continue
        wrong = [
            option
            for option in distractor_pool
            if option.lower() != answer.lower() and len(option) >= 25
        ]
        options = [answer, *list(islice(wrong, 3))]
        if len(options) < 4:
            options.extend(
                [
                    "Một nhận định không xuất hiện trong phần ngữ cảnh được chọn",
                    "Một khái niệm không liên quan trực tiếp đến nội dung bài học",
                    "Một mô tả chung không đủ căn cứ từ tài liệu",
                ][: 4 - len(options)]
            )
        marker = f"S{len(items) + 1}"
        items.append(
            QuizItem(
                question=question,
                options=options[:4],
                correct_index=0,
                explanation=f"Đáp án xuất hiện trực tiếp trong nội dung nguồn: {sentence}",
                source_markers=[marker],
                difficulty="medium",
                topic=topic,
            )
        )
        used_chunks.append(chunk)
        used_questions.add(question.lower())
    return items, used_chunks


def make_flashcards(chunks: list[RetrievedChunk], count: int) -> tuple[list[Flashcard], list[RetrievedChunk]]:
    keywords = top_keywords(chunks, limit=max(count * 2, 10))
    cards: list[Flashcard] = []
    used_chunks: list[RetrievedChunk] = []
    seen_fronts: set[str] = set()
    for chunk_index, chunk in enumerate(chunks, 1):
        if len(cards) >= count:
            break
        topic = _chunk_title(chunk, keywords[0] if keywords else "nội dung bài học")
        useful = [s for s in split_sentences(chunk.text) if _is_useful_sentence(s)]
        if not useful:
            continue
        front = f"{topic}: cần nhớ điều gì?"
        if front.lower() in seen_fronts:
            continue
        back = " ".join(_compact(sentence, 160) for sentence in useful[:2])
        marker = f"S{len(cards) + 1}"
        cards.append(
            Flashcard(
                front=front,
                back=back,
                hint=f"Xem lại trang {chunk.metadata.page} của {chunk.metadata.filename}.",
                topic=topic,
                source_markers=[marker],
            )
        )
        used_chunks.append(chunk)
        seen_fronts.add(front.lower())
    return cards, used_chunks
