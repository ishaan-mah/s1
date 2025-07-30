from datasets import load_dataset, Dataset, concatenate_datasets
import random
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from networkx import fast_gnp_random_graph
import networkx as nx
import json
import os

SAVE_PATH = "/shared/share_mala/Ishaan/s1k_mixed_Data/"

def generate_random_dag(num_nodes=5, edge_prob=0.3, seed=None):
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

def format_trace(query_type, X, Y, Z, edges, answer):
    if query_type == "confounder":
        steps = [
            f"1. {Z} must be an ancestor of both {X} and {Y}.",
            f"2. {X} and {Y} must be dependent without conditioning.",
            f"3. Conditioning on {Z} should make them independent.",
            f"4. If all conditions are met, then {Z} is a confounder."
        ]
    elif query_type == "collider":
        steps = [
            f"1. Check if edges ({X}, {Z}) and ({Y}, {Z}) exist in DAG.",
            f"2. If both exist, {Z} is a collider."
        ]
    elif query_type == "conditional independence":
        steps = [
            f"1. Examine if any unblocked path connects {X} and {Y}.",
            f"2. Then condition on {Z} and see if the path is blocked."
        ]
    elif query_type == "mediator":
        steps = [
            f"1. Confirm if ({X}, {Z}) and ({Z}, {Y}) are in DAG.",
            f"2. If both exist, {Z} is a mediator from {X} to {Y}."
        ]
    elif query_type == "parent":
        steps = [
            f"If edge {Z} -> {X} exists, {Z} is a parent of {X}."
        ]
    elif query_type == "child":
        steps = [
            f"If edge {X} -> {Z} exists, {Z} is a child of {X}."
        ]
    elif query_type == "ancestor":
        steps = [
            f"If a directed path {Z} -> ... -> {X} exists and {Z} is not {X}, then {Z} is an ancestor of {X}."
        ]
    elif query_type == "descendant":
        steps = [
            f"If a directed path {X} -> ... -> {Z} exists and {X} is not {Z}, then {Z} is an descendant of {X}."
        ]
    elif query_type == "dependent":
        steps = [
            f"{X} and {Y} are d-connected (no conditioning) then they are dependent."
        ]
    elif query_type == "independent":
        steps = [
            f"{X} and {Y} are d-separated (no conditioning) then they are marginally independent."
        ]
    else:
        steps = [
            f"1. Apply rules based on d-separation or path structure for '{query_type}'."
        ]
    return "\n".join(steps) + f"\nAnswer: {answer}"

def generate_query(model, seed, query_type):
    nodes = list(model.nodes())
    if len(nodes) < 3:
        return None
    random.seed(seed)
    X, Y, Z = random.sample(nodes, 3)
    edges = list(model.edges())
    prompt_prefix = f"Given the DAG with edges {edges}, "
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
        question = f"are {X} and {Y} dependent without conditioning?"
    elif query_type == "independent":
        answer = int(not model.is_dconnected(X, Y, observed=[]))
        question = f"are {X} and {Y} marginally independent?"
    elif query_type == "collider":
        answer = int((X, Z) in model.edges() and (Y, Z) in model.edges())
        question = f"is {Z} a collider on the path between {X} and {Y}?"
    elif query_type == "mediator":
        answer = int((X, Z) in model.edges() and (Z, Y) in model.edges())
        question = f"is {Z} a mediator from {X} to {Y}?"
    else:
        return None
    answer_text = "Yes" if answer else "No"
    reasoning = format_trace(query_type, X, Y, Z, edges, answer_text)
    text = f"<|im_start|>user\n{prompt_prefix}{question}\n\nPlease explain your reasoning step-by-step.\n<|im_start|>assistant\n{reasoning}"
    return {
        "text": text,
        "source_type": "dag",
        "cot_type": "cot"
    }

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
    s1k_raw = load_dataset("simplescaling/s1K", split="train[:1000]")
    s1k_list = s1k_raw.to_list()
    for ex in s1k_list:
        ex["source_type"] = "s1k"
        ex["cot_type"] = ex.get("cot_type", "cot")
        # Construct the 'text' field like the causal format
        question = ex.get("question", "")
        cot = ex.get("cot") or ex.get("thinking_trajectories", [""])[0]
        ex["text"] = f"<|im_start|>user\n{question}\n\nPlease explain your reasoning step-by-step.\n<|im_start|>assistant\n{cot}"
    s1k_dataset = Dataset.from_list(s1k_list)

    print("🔹 Generating 1000 causal DAG reasoning examples...")
    dag_dataset = Dataset.from_list(generate_causal_query_dataset(n_samples=1000))

    print("🔹 Concatenating datasets...")
    mixed_ds = concatenate_datasets([s1k_dataset, dag_dataset]).shuffle(seed=42)
    print(f"✅ Final dataset size: {len(mixed_ds)}")

    print(f"💾 Saving to {SAVE_PATH}")
    mixed_ds.save_to_disk(SAVE_PATH)

    json_path = os.path.join(SAVE_PATH, "s1k_mixed_reasoning.json")
    with open(json_path, "w") as f:
        for ex in mixed_ds:
            json.dump(ex, f)
            f.write("\n")
    print(f"📄 JSON copy saved to {json_path}")

if __name__ == "__main__":
    create_and_save_dataset()
