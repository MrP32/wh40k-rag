import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import json, os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH     = r"C:\Projects\wh40k-app\chroma_db"
COLLECTION_NAME = "warhammer40k"
TEST_QUERY      = "Librarius Conclave detachment rule enhancements"

embedding_fn = OllamaEmbeddingFunction(
    url="http://127.0.0.1:11434/api/embeddings",
    model_name="nomic-embed-text"
)
client     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

# What does Claude extract as filters?
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
resp = anthropic_client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=100,
    messages=[{"role": "user", "content": f"Extract ChromaDB filters for: {TEST_QUERY}\nReturn JSON with subject and category fields only."}]
)
print(f"Filter extraction: {resp.content[0].text.strip()}\n")

# What chunks are retrieved with subject=space marines, category=faction_rules?
results = collection.query(
    query_texts=[TEST_QUERY],
    n_results=40,
    where={"$and": [
        {"subject": {"$eq": "space marines"}},
        {"content_category": {"$eq": "faction_rules"}}
    ]}
)

chunks    = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]

print(f"Total retrieved: {len(chunks)}\n")
print("=" * 60)
for i, (chunk, meta, dist) in enumerate(zip(chunks[:10], metadatas[:10], distances[:10])):
    words = len(chunk.split())
    print(f"── {i+1} | dist={dist:.4f} | words={words} | section={meta.get('section_type')}")
    print(f"   {chunk[:200].replace(chr(10), ' ')}")
    print()