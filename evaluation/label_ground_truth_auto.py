import json
import time
from pathlib import Path

# Imports your existing Qwen model
from rag.generation import get_client, MODEL
# Change this line at the top of label_ground_truth_auto.py

INPUT_FILE = Path("evaluation/results/ground_truth.json")
OUTPUT_FILE = Path("evaluation/results/ground_truth_labeled.json")

def evaluate_relevance_with_llm(query: str, text: str, client) -> bool:
    """
    Uses the LLM to deterministically judge if a chunk contains the 
    factual answer to the query. Returns True if relevant, False otherwise.
    """
    prompt = f"""You are a strict evaluation judge. 
Given the user's QUERY and a document CHUNK, determine if the CHUNK contains the exact, factual answer to the QUERY.
Respond with exactly one word: "YES" if it contains the answer, or "NO" if it does not.

QUERY: {query}

CHUNK: 
{text}

ANSWER:"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL, # Continues using Qwen for metric consistency
                messages=[
                    {"role": "system", "content": "You are a binary relevance evaluator."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=5,
                temperature=0.0,
            )
            decision = response.choices[0].message.content.strip().upper()
            return "YES" in decision
        except Exception as e:
            # Exponential backoff if the API token limit is hit again
            if "429" in str(e) and attempt < max_retries - 1:
                wait_time = 15 * (attempt + 1)
                print(f"\n[!] Rate limited. Backing off for {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"\n[!] LLM Evaluation Error: {e}")
                return False
    return False

def main():
    print("Loading LLM client...")
    client = get_client()

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Checkpoint Recovery: Map existing queries so we don't repeat API calls
    completed_queries = {}
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                completed_queries = {
                    item["query"]: item
                    for item in existing_data
                    if "relevant_doc_keys" in item
                }
            print(f"Resuming pipeline. Found {len(completed_queries)} previously completed queries.")
        except Exception:
            completed_queries = {}

    labeled_data = []

    print(f"\nStarting automated LLM-as-a-Judge labeling for {len(data)} queries...\n")

    for i, item in enumerate(data, 1):
        query = item["query"]
        
        # Fast-forward past already evaluated chunks
        if query in completed_queries:
            print(f"[{i:02d}/{len(data)}] Skipping (Already Labeled): {query[:45]}...")
            labeled_data.append(completed_queries[query])
            continue

        print(f"[{i:02d}/{len(data)}] Grading: {query[:45]}...")
        relevant_keys = []
        
        for candidate in item["candidates"]:
            if candidate["source"] != item["source"]:
                candidate["relevant"] = False
                continue
                
            is_relevant = evaluate_relevance_with_llm(query, candidate["text"], client)
            candidate["relevant"] = is_relevant
            
            if is_relevant:
                relevant_keys.append(candidate["doc_key"])
                
            time.sleep(0.3) 

        item["relevant_doc_keys"] = relevant_keys
        labeled_data.append(item)
        print(f"   -> Found {len(relevant_keys)} relevant chunks.")
        
        # Stateful Save: Write to disk after every single query
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(labeled_data, f, indent=2, ensure_ascii=False)

    print(f"\nAutomated ground truth saved to:\n{OUTPUT_FILE}")

if __name__ == "__main__":
    main()