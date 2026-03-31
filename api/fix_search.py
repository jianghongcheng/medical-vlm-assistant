# Read the file
with open('merlin_pipeline.py', 'r') as f:
    content = f.read()

# Fix qdrant search API for newer versions
old = """    results = qdrant.search(
        collection_name="radiology_reports",
        query_vector=vector,
        limit=top_k
    )
    return [r.payload for r in results]"""

new = """    try:
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
    return [r.payload for r in results]"""

content = content.replace(old, new)

with open('merlin_pipeline.py', 'w') as f:
    f.write(content)

print("Fixed!")
