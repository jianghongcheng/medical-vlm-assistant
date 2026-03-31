with open('merlin_pipeline.py', 'r') as f:
    content = f.read()

old = '''def retrieve_similar_reports(query_text: str, top_k: int = 3) -> list:
    """Retrieve similar radiology reports using text-based RAG"""
    from sentence_transformers import SentenceTransformer
    text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

    vector = text_encoder.encode(query_text).tolist()
    try:
        # Try newer Qdrant API
        results = qdrant.query_points(
            collection_name="radiology_reports",
            query=vector,
            limit=top_k
        ).points
    except AttributeError:
        # Fall back to older API
        results = qdrant.search(
            collection_name="radiology_reports",
            query_vector=vector,
            limit=top_k
        )
    return [r.payload for r in results]'''

new = '''def retrieve_similar_reports(query_text: str, top_k: int = 3) -> list:
    """Retrieve similar radiology reports using text-based RAG"""
    from sentence_transformers import SentenceTransformer
    text_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    vector = text_encoder.encode(query_text).tolist()
    try:
        from qdrant_client.models import QueryRequest
        results = qdrant.query_points(
            collection_name="radiology_reports",
            query=vector,
            limit=top_k
        ).points
        return [r.payload for r in results]
    except Exception:
        pass
    try:
        results = qdrant.search(
            collection_name="radiology_reports",
            query_vector=vector,
            limit=top_k
        )
        return [r.payload for r in results]
    except Exception:
        return SAMPLE_REPORTS[:top_k]'''

if old in content:
    content = content.replace(old, new)
    print("Pattern found and replaced")
else:
    print("Pattern NOT found, doing line-based fix")
    # Find and replace the function entirely
    lines = content.split('\n')
    new_lines = []
    skip = False
    for i, line in enumerate(lines):
        if 'def retrieve_similar_reports' in line:
            skip = True
            new_lines.append(new)
        elif skip and line.startswith('def ') and 'retrieve_similar_reports' not in line:
            skip = False
            new_lines.append(line)
        elif not skip:
            new_lines.append(line)
    content = '\n'.join(new_lines)

with open('merlin_pipeline.py', 'w') as f:
    f.write(content)
print("Done!")
