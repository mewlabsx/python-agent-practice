from pathlib import Path

from embedding_provider import embed_texts


def load_documents(directory):
    directory_path = Path(directory)
    documents = []

    for file_path in directory_path.glob("*.md"):
        text = file_path.read_text(encoding="utf-8")
        documents.append({
            "source": file_path.name,
            "text": text
        })

    return documents


# 按空行切段落，再把段落攒成块。
# 不是按字数硬切 —— 段落不被切断是前提，所以单块可能略超 max_chars。
def split_text_by_paragraphs(text, max_chars=500):
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]

    parts = []
    current_part = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)

        # 当前块已经有内容，再加这一段就超了 -> 先把当前块封箱
        if current_part and current_length + paragraph_length > max_chars:
            parts.append("\n\n".join(current_part))
            current_part = []
            current_length = 0

        current_part.append(paragraph)
        current_length += paragraph_length

    # 循环结束时最后一块还没封箱
    if current_part:
        parts.append("\n\n".join(current_part))

    return parts


# 把 md 文本解析成结构化 chunk：一台逐行读的状态机。
# 每读一行做一次四选一 —— 章节标题 / 角色行 / 指标标题 / 正文。
def chunk_documents(documents, indicator_catalog, max_chars=500):
    chunks = []

    for document in documents:
        source = document["source"]
        lines = document["text"].splitlines()

        # 这 5 个变量描述的是同一个东西：「当前正在处理的这个指标」。
        # 它们成套出现、成套清空。
        current_chapter_title = None
        current_chapter_role = None
        current_indicator_id = None
        current_indicator_name = None
        current_text_lines = []

        # 定义在 for 里面，所以看得见上面那 5 个变量 —— 这是闭包，不是传参。
        # 读外层变量不用声明；这里只有 current_text_lines 被重新赋值，所以只声明它。
        def flush_indicator():
            nonlocal current_text_lines

            # 此刻没有指标是常态（还没遇到，或刚清场），安静退出
            if current_indicator_id is None:
                return

            text = "\n".join(current_text_lines).strip()
            if not text:
                current_text_lines = []
                return

            # 走到这里说明一定有指标，所以查不到只能是文档写错了
            indicator = indicator_catalog.get(current_indicator_id)
            if indicator is None:
                raise ValueError(f"Unknown indicator: {current_indicator_id}")

            if current_chapter_role is None:
                raise ValueError(f"Missing chapter role for {current_indicator_id}")

            parts = split_text_by_paragraphs(text, max_chars=max_chars)

            for part_index, part_text in enumerate(parts, start=1):
                chunk_id = f"{current_indicator_id}.{current_chapter_role}.{part_index:03d}"

                chunks.append({
                    "chunk_id": chunk_id,
                    "source": source,
                    "chapter_title": current_chapter_title,
                    "chapter_role": current_chapter_role,
                    "indicator_id": current_indicator_id,
                    "indicator_name": current_indicator_name,
                    "indicator_aliases": indicator["aliases"],
                    "part_index": part_index,
                    "text": part_text
                })

            current_text_lines = []

        for line in lines:
            stripped = line.strip()

            # 章节标题 `# xxx`
            if stripped.startswith("# ") and not stripped.startswith("## "):
                flush_indicator()
                current_chapter_title = stripped[2:].strip()
                current_chapter_role = None
                current_indicator_id = None
                current_indicator_name = None
                continue

            # 章节角色 `role: xxx` —— 唯一不 flush 的结构性行，它只记录、不换段
            if stripped.startswith("role:"):
                current_chapter_role = stripped.split(":", 1)[1].strip()
                continue

            # 指标标题 `## ID 名字`
            if stripped.startswith("## "):
                flush_indicator()

                heading = stripped[3:].strip()
                heading_parts = heading.split(" ", 1)

                if len(heading_parts) != 2:
                    raise ValueError(f"Invalid indicator heading: {heading}")

                current_indicator_id = heading_parts[0]
                current_indicator_name = heading_parts[1]
                current_text_lines = []
                continue

            # 其余所有行都是正文；还没进入任何指标之前，正文直接丢掉
            if current_indicator_id is not None:
                current_text_lines.append(line)

        # 文件读完了，最后一个指标还没结算
        flush_indicator()

    return chunks


# 把正文和 metadata 拼成一段文本。
# 正文往往太短、语义不够，所以要带上指标名和别名一起送去向量化。
def build_embedding_text(chunk):
    aliases = "；".join(chunk.get("indicator_aliases", []))

    return (
        f"指标：{chunk.get('indicator_name', '')}\n"
        f"指标别名：{aliases}\n"
        f"章节类型：{chunk.get('chapter_role', '')}\n"
        f"章节名称：{chunk.get('chapter_title', '')}\n\n"
        f"正文：\n"
        f"{chunk.get('text', '')}"
    )


def build_index(chunks):
    embedding_texts = [build_embedding_text(chunk) for chunk in chunks]
    embeddings = embed_texts(embedding_texts)

    index = {}
    for chunk, embedding in zip(chunks, embeddings):
        index[chunk["chunk_id"]] = embedding

    return index


if __name__ == "__main__":
    from rag import indicator_catalog

    documents = load_documents("knowledge")
    chunks = chunk_documents(documents, indicator_catalog, max_chars=500)

    for chunk in chunks:
        print("\n=========================")
        print(chunk["chunk_id"])
        print(chunk["chapter_title"])
        print(chunk["chapter_role"])
        print(chunk["indicator_name"])
        print(chunk["text"])
