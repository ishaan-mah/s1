from datasets import load_dataset, Dataset, concatenate_datasets
import random
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from networkx import fast_gnp_random_graph, is_directed_acyclic_graph
import networkx as nx
import json
import os

SAVE_PATH = "/shared/share_mala/Ishaan/s1k_mixed_Data/"

def generate_random_dag(num_nodes=5, edge_prob=0.3, seed=None, max_tries=100):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    nodes = list(range(num_nodes))
    random.shuffle(nodes)

    edges = [
        (nodes[i], nodes[j])
        for i in range(num_nodes)
        for j in range(i + 1, num_nodes)
        if random.random() < edge_prob
    ]

    return DiscreteBayesianNetwork(edges)

def get_ancestors(model, node):
    ancestors = set()
    def dfs(n):
        for parent in model.get_parents(n):
            if parent not in ancestors:
                ancestors.add(parent)
                dfs(parent)
    dfs(node)
    return ancestors

def is_confounder(model, Z, X, Y):
    ancestors_X = get_ancestors(model, X)
    ancestors_Y = get_ancestors(model, Y)
    if Z not in ancestors_X or Z not in ancestors_Y:
        return False
    if not model.is_dconnected(X, Y):
        return False
    if model.is_dconnected(X, Y, observed=[Z]):
        return False
    return True

def generate_query(model, seed, query_type):
    nodes = list(model.nodes())
    if len(nodes) < 3:
        return None

    random.seed(seed)
    X, Y, Z = random.sample(nodes, 3)
    edges = list(model.edges())

    prompt_prefix = (
        "Answer with exactly one word. Do not explain or add any text. Just one word.\n\n"
        f"Input: Given the DAG with edges {edges}, "
    )

    if query_type == "confounder":
        answer = int(is_confounder(model, Z, X, Y))
        question = f"is {Z} a confounder for {X} and {Y}?"

    elif query_type == "parent":
        answer = int((Z, X) in model.edges())
        question = f"is {Z} a parent of {X}?"

    elif query_type == "child":
        answer = int((X, Z) in model.edges())
        question = f"is {Z} a child of {X}?"

    elif query_type == "conditional independence":
        answer = int(not model.is_dconnected(X, Y, observed=[Z]))
        question = f"are {X} and {Y} conditionally independent given {Z}?"

    elif query_type == "ancestor":
        answer = int(nx.has_path(model, Z, X) and Z != X)
        question = f"is {Z} an ancestor of {X}?"

    elif query_type == "descendant":
        answer = int(nx.has_path(model, X, Z) and X != Z)
        question = f"is {Z} a descendant of {X}?"

    elif query_type == "dependent":
        answer = int(model.is_dconnected(X, Y, observed=[]))
        question = f"are {X} and {Y} dependent (d-connected without conditioning)?"

    elif query_type == "independent":
        answer = int(not model.is_dconnected(X, Y, observed=[]))
        question = f"are {X} and {Y} marginally independent (not d-connected)?"

    elif query_type == "collider":
        answer = int((X, Z) in model.edges() and (Y, Z) in model.edges())
        question = f"is {Z} a collider on the path between {X} and {Y}?"

    elif query_type == "mediator":
        answer = int((X, Z) in model.edges() and (Z, Y) in model.edges())
        question = f"is {Z} a mediator from {X} to {Y}?"

    else:
        return None

    answer_text = "Yes" if answer else "No"
    prompt = f"{prompt_prefix}{question}\nAnswer: {answer_text}"
    return {"text": prompt, "source_type": "dag", "cot_type": "none"}


def generate_causal_query_dataset(n_samples=1000, edge_prob=0.3, seed=42):
    question_types = [
        "confounder", "parent", "child", "conditional independence",
        "ancestor", "descendant", "dependent", "independent",
        "collider", "mediator"
    ]
    dataset = []
    for i in range(n_samples):
        num_nodes = random.randint(5, 10)
        dag = generate_random_dag(num_nodes, edge_prob=edge_prob, seed=seed + i)
        query_type = random.choice(question_types)
        query = generate_query(dag, seed + i, query_type)
        if query:
            dataset.append(query)
    return dataset

def create_and_save_dataset():
    print("🔹 Loading 1000 S1K examples...")
    raw_s1k = load_dataset("simplescaling/s1K")["train"].select(range(1000))

    # Safely extract text and construct simplified format
    s1k_list = []
    for ex in raw_s1k:
        text = ex.get("text")
        if not text:
            question = ex.get("question", "")
            solution = ex.get("solution", "")
            text = f"{question}\nAnswer: {solution}"
        s1k_list.append({
            "text": text,
            "source_type": "s1k",
            "cot_type": ex.get("cot_type", "none")
        })

    s1k = Dataset.from_list(s1k_list)

    print("🔹 Generating 1000 synthetic DAG examples...")
    # from train.synthetic_generator import generate_causal_query_dataset  # ← adjust path if needed
    dag_data = generate_causal_query_dataset(n_samples=1000)
    dag_ds = Dataset.from_list(dag_data)

    print("🔹 Concatenating datasets...")
    mixed_ds = concatenate_datasets([s1k, dag_ds])
    print(f"✅ Final dataset size: {len(mixed_ds)}")
    
    mixed_ds = mixed_ds.shuffle(seed=42)
    print(f"💾 Saving Hugging Face dataset to {SAVE_PATH} ...")
    mixed_ds.save_to_disk(SAVE_PATH)

    # Optional: save JSONL for inspection
    json_path = os.path.join(SAVE_PATH, "s1k_mixed.json")
    with open(json_path, "w") as f:
        for ex in mixed_ds:
            json.dump(ex, f)
            f.write("\n")
    print(f"📄 JSON copy saved at {json_path}")


    # Save to JSONL format for inspection
    json_path = os.path.join(SAVE_PATH, "s1k_mixed.json")
    with open(json_path, "w") as f:
        for ex in mixed_ds:
            json.dump(ex, f)
            f.write("\n")
    print(f"✅ JSON copy saved at {json_path}")

if __name__ == "__main__":
    create_and_save_dataset()
