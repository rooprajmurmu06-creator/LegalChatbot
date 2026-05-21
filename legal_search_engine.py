#!/usr/bin/env python3
"""
Legal Knowledge Graph Search Engine
Master Index + Lazy Loading + LRU Cache + BM25 Search
"""

import json
import os
import re
from collections import defaultdict, OrderedDict
from typing import List, Dict, Set, Tuple
import time
from rank_bm25 import BM25Okapi
# ========== Configuration ==========
import os
from dotenv import load_dotenv

# Try to load .env from the current working directory, then from the script's directory
load_dotenv()

# Get the directory where this file (legal_search_engine.py) is located
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Read KG_ROOT from environment variable, with fallback
KG_ROOT_ENV = os.getenv("KG_DATA_PATH", "")
if KG_ROOT_ENV:
    if os.path.isabs(KG_ROOT_ENV):
        KG_ROOT = KG_ROOT_ENV
    else:
        KG_ROOT = os.path.join(_BASE_DIR, KG_ROOT_ENV)
else:
    KG_ROOT = os.path.join(_BASE_DIR, "knowledge_graphs")
    
MASTER_INDEX_FILE = os.path.join(KG_ROOT, "metadata.json")
CACHE_SIZE = 3
BUILD_INDEX = True

# Ensure KG_ROOT exists (create directory if needed)
os.makedirs(KG_ROOT, exist_ok=True)

print(f"📁 Knowledge Graph root: {KG_ROOT}")

class MasterIndex:
    def __init__(self, root_folder=KG_ROOT, index_file=MASTER_INDEX_FILE):
        self.root_folder = root_folder
        self.index_file = index_file
        self.keyword_to_files = defaultdict(list)
        self.node_to_files = defaultdict(list)
        self.file_metadata = {}
        if os.path.exists(index_file) and not BUILD_INDEX:
            self._load_index()
        else:
            self._build_index()

    def _load_index(self):
        print(f"📀 Loading master index from {self.index_file}...")
        with open(self.index_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.keyword_to_files = defaultdict(list, data.get("keyword_to_files", {}))
        self.node_to_files = defaultdict(list, data.get("node_to_files", {}))
        self.file_metadata = data.get("file_metadata", {})
        print(f"✅ Master index loaded: {len(self.file_metadata)} KG files")

    def _build_index(self):
        print("🔨 Building master index (scanning all KG files)...")
        for root, dirs, files in os.walk(self.root_folder):
            for file in files:
                if not file.endswith('.json') or file == "metadata.json":
                    continue
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, self.root_folder)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        triples = data.get("triples", data.get("edges", []))
                    if not triples:
                        continue
                    keywords = set()
                    nodes = set()
                    for src, rel, tgt in triples[:200]:
                        src_lower = src.lower()
                        tgt_lower = tgt.lower()
                        nodes.add(src_lower)
                        nodes.add(tgt_lower)
                        for word in src_lower.split() + tgt_lower.split():
                            if len(word) >= 3 and word not in self._stopwords():
                                keywords.add(word)
                    self.file_metadata[relative_path] = {
                        "path": file_path,
                        "triple_count": len(triples),
                        "keywords": list(keywords),
                        "nodes": list(nodes),
                        "folder": root
                    }
                    for kw in keywords:
                        self.keyword_to_files[kw].append(relative_path)
                    for node in nodes:
                        self.node_to_files[node].append(relative_path)
                    print(f"   📄 {relative_path}: {len(triples)} triples, {len(keywords)} keywords")
                except Exception as e:
                    print(f"   ⚠️ Error reading {file_path}: {e}")
        print(f"\n💾 Saving master index to {self.index_file}...")
        index_data = {
            "keyword_to_files": dict(self.keyword_to_files),
            "node_to_files": dict(self.node_to_files),
            "file_metadata": self.file_metadata,
            "build_time": time.time()
        }
        os.makedirs(self.root_folder, exist_ok=True)
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Master index built: {len(self.file_metadata)} KG files")

    def _stopwords(self):
        return {"the", "and", "for", "with", "from", "this", "that", "were", "was",
                "what", "who", "where", "when", "why", "how", "is", "are", "be"}

    def find_relevant_kgs(self, question: str, max_kgs: int = 3) -> List[Tuple[str, int, dict]]:
        question_lower = question.lower()
        keywords = re.findall(r'\b[a-z]{3,}\b', question_lower)
        keywords = [k for k in keywords if k not in self._stopwords()]
        scores = defaultdict(int)
        for kw in keywords:
            for file_path in self.keyword_to_files.get(kw, []):
                scores[file_path] += 1
        for kw in keywords:
            for file_path in self.node_to_files.get(kw, []):
                scores[file_path] += 2
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for file_path, score in ranked[:max_kgs]:
            if file_path in self.file_metadata:
                results.append((file_path, score, self.file_metadata[file_path]))
        return results

    def get_stats(self):
        total_triples = sum(m.get("triple_count", 0) for m in self.file_metadata.values())
        return {
            "total_files": len(self.file_metadata),
            "total_triples": total_triples,
            "unique_keywords": len(self.keyword_to_files),
            "unique_nodes": len(self.node_to_files)
        }

class LRUCache(OrderedDict):
    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity
    def get(self, key):
        if key not in self:
            return None
        self.move_to_end(key)
        return self[key]
    def put(self, key, value):
        if key in self:
            self.move_to_end(key)
        self[key] = value
        if len(self) > self.capacity:
            oldest = next(iter(self))
            print(f"   🗑️ Cache evicted: {oldest}")
            del self[oldest]

class KGCache:
    def __init__(self, cache_size=CACHE_SIZE):
        self.cache = LRUCache(cache_size)

    def load_kg(self, file_path: str) -> Dict:
        cached = self.cache.get(file_path)
        if cached:
            print(f"   📦 Cache hit: {os.path.basename(file_path)}")
            return cached
        print(f"   📂 Loading: {os.path.basename(file_path)}...")
        start_time = time.time()
        full_path = os.path.join(KG_ROOT, file_path)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                triples = data.get("triples", data.get("edges", []))
            kg_index = self._build_kg_index(triples)
            kg_index["triples"] = triples
            kg_index["file_path"] = file_path
            self.cache.put(file_path, kg_index)
            load_time = time.time() - start_time
            print(f"   ✅ Loaded {len(triples)} triples in {load_time:.2f}s")
            return kg_index
        except Exception as e:
            print(f"   ❌ Error loading {file_path}: {e}")
            return {"triples": [], "node_out": {}, "node_in": {}, "bm25": None}

    def _build_kg_index(self, triples: List) -> Dict:
        node_out = defaultdict(list)
        node_in = defaultdict(list)
        node_names = set()
        for src, rel, tgt in triples:
            src_lower = src.lower()
            tgt_lower = tgt.lower()
            node_names.add(src_lower)
            node_names.add(tgt_lower)
            node_out[src_lower].append((tgt_lower, rel.lower()))
            node_in[tgt_lower].append((src_lower, rel.lower()))
        node_names = list(node_names)
        tokenized_nodes = [name.split() for name in node_names]
        bm25 = BM25Okapi(tokenized_nodes)
        return {
            "node_names": node_names,
            "bm25": bm25,
            "node_out": dict(node_out),
            "node_in": dict(node_in)
        }

    def search_in_kg(self, kg_index: Dict, question: str, max_facts: int = 30, top_k_nodes: int = 5) -> List[str]:
        node_names = kg_index.get("node_names", [])
        bm25 = kg_index.get("bm25")
        node_out = kg_index.get("node_out", {})
        node_in = kg_index.get("node_in", {})
        if not bm25 or not node_names:
            return []
        query_tokens = question.lower().split()
        if not query_tokens:
            return []
        scores = bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k_nodes]
        matched_nodes = [node_names[i] for i in top_indices if scores[i] > 0]
        if not matched_nodes:
            for node in node_names:
                if any(kw in node for kw in query_tokens):
                    matched_nodes.append(node)
                    if len(matched_nodes) >= top_k_nodes:
                        break
        facts = []
        for node in matched_nodes:
            for neighbor, rel in node_out.get(node, []):
                facts.append(f"{node} --[{rel}]--> {neighbor}")
            for pred, rel in node_in.get(node, []):
                facts.append(f"{pred} --[{rel}]--> {node}")
        seen = set()
        unique_facts = []
        for f in facts:
            if f not in seen:
                seen.add(f)
                unique_facts.append(f)
        return unique_facts[:max_facts]

    def get_stats(self):
        return {
            "cache_size": len(self.cache),
            "cache_capacity": self.cache.capacity,
            "cached_files": list(self.cache.keys())
        }

class LegalKGSearchEngine:
    def __init__(self):
        print("=" * 60)
        print("📚 Legal Knowledge Graph Search Engine")
        print("   Master Index + Lazy Loading + LRU Cache + BM25 Search")
        print("=" * 60)
        print("\n🔍 Initializing Master Index...")
        self.master_index = MasterIndex()
        stats = self.master_index.get_stats()
        print(f"   📊 Master Index Stats:")
        print(f"      • {stats['total_files']} KG files")
        print(f"      • {stats['total_triples']:,} total triples")
        print(f"      • {stats['unique_keywords']:,} unique keywords")
        print(f"      • {stats['unique_nodes']:,} unique nodes")
        self.cache = KGCache()
        self.query_count = 0
        self.total_search_time = 0

    def search(self, question: str, max_kgs: int = 2) -> List[str]:
        self.query_count += 1
        start_time = time.time()
        print(f"\n🔍 Query #{self.query_count}: {question}")
        print("-" * 40)
        relevant_kgs = self.master_index.find_relevant_kgs(question, max_kgs)
        if not relevant_kgs:
            print("   ❌ No relevant KG files found")
            return []
        print(f"   📁 Found {len(relevant_kgs)} relevant KG(s):")
        for kg_path, score, metadata in relevant_kgs:
            print(f"      • {kg_path} (score: {score}, triples: {metadata['triple_count']})")
        all_facts = []
        for kg_path, score, metadata in relevant_kgs:
            kg_index = self.cache.load_kg(kg_path)
            facts = self.cache.search_in_kg(kg_index, question, max_facts=20)
            all_facts.extend(facts)
            if len(all_facts) >= 30:
                break
        search_time = time.time() - start_time
        self.total_search_time += search_time
        print(f"\n   📊 Results: {len(all_facts)} facts found")
        print(f"   ⏱️ Search time: {search_time:.3f}s")
        return all_facts[:30]

    def get_stats(self):
        avg_time = self.total_search_time / self.query_count if self.query_count > 0 else 0
        return {
            "total_queries": self.query_count,
            "avg_search_time": avg_time,
            "cache_stats": self.cache.get_stats(),
            "master_index_stats": self.master_index.get_stats()
        }