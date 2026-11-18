"""Retrieval-augmented AfriMMLU prompts (Algorithm 2).

The five templates are IrokoBench's AfriMMLU direct prompts, copied verbatim from
lm_eval/tasks/afrimmlu (afrimmlu_direct_kin_prompt_1..5). The only change is {context}, placed
right before "Question:". Nothing else in the instructions changes, so any difference from the
closed-book baseline comes from the passages.

Environment variables:
  AFRIMMLU_RAG_FILE  retrieval file (one JSON line per question, with "question" and "passages"),
                     e.g. results/retrieval/bm25/afrimmlu_kin.jsonl
  RAG_K              passages used per question (default 3)
  RAG_SKIP_SUBJECTS  comma-separated subjects answered closed-book (default: none), for the
                     "no retrieval for math" ablation, e.g. elementary_mathematics
"""
import ast
import json
import os

_PASSAGES = None


def _load():
    global _PASSAGES
    if _PASSAGES is None:
        path = os.environ.get("AFRIMMLU_RAG_FILE")
        if not path:
            raise RuntimeError("set AFRIMMLU_RAG_FILE to a retrieval file (see tasks/afrimmlu_rag/utils.py)")
        _PASSAGES = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    _PASSAGES[r["question"].strip()] = r["passages"]
    return _PASSAGES


def _context(doc) -> str:
    k = int(os.environ.get("RAG_K", "3"))
    skip = {s.strip() for s in os.environ.get("RAG_SKIP_SUBJECTS", "").split(",") if s.strip()}
    if k <= 0 or doc["subject"] in skip:
        return ""
    passages = _load().get(doc["question"].strip())
    if passages is None:
        raise KeyError(f"no retrieved passages for question: {doc['question'][:80]!r}")
    lines = [f"[{i}] {p['title']}: {p['text']}" for i, p in enumerate(passages[:k], start=1)]
    return "Context:\n" + "\n".join(lines) + "\n\n"


def process_docs(dataset):
    """Attach the context to every document once, and fail loudly if a question has no passages."""
    # keep_in_memory + no cache: a cached copy could hold passages from another retriever or k
    return dataset.map(lambda doc: {"context": _context(doc)}, keep_in_memory=True, load_from_cache_file=False)


def doc_to_choice(doc):
    return ast.literal_eval(doc["choices"])


def _fill(template: str, doc) -> str:
    c = ast.literal_eval(doc["choices"])
    return template.format(subject=doc["subject"], question=doc["question"], context=doc["context"],
                           choice1=c[0], choice2=c[1], choice3=c[2], choice4=c[3])


def prompt_1(doc):
    return _fill("""You are a highly knowledgeable and intelligent artificial intelligence
                model answers multiple-choice questions about {subject}

                {context}Question: {question}

                Choices:
                        A: {choice1}
                        B: {choice2}
                        C: {choice3}
                        D: {choice4}

                Answer:  """, doc)


def prompt_2(doc):
    return _fill("""As an expert in {subject}, choose the most accurate answer to the question below.
Your goal is to select the correct option 'A', 'B', 'C', or 'D' by understanding the nuances of the topic.

{context}Question: {question}
Choices:
        A: {choice1}
        B: {choice2}
        C: {choice3}
        D: {choice4}
Answer: """, doc)


def prompt_3(doc):
    return _fill("""You are a subject matter expert in {subject}.

  Utilizing your expertise in {subject}, answer the following multiple-choice question
  by picking 'A', 'B', 'C', or 'D'.

{context}Question: {question}
Choices:
        A: {choice1}
        B: {choice2}
        C: {choice3}
        D: {choice4}
Answer: """, doc)


def prompt_4(doc):
    return _fill("""Analyze each question critically and determine the most correct option based on your understanding of the subject matter

{context}Question: {question}
Choices:
        A: {choice1}
        B: {choice2}
        C: {choice3}
        D: {choice4}
Answer: """, doc)


def prompt_5(doc):
    return _fill("""Given your proficiency in {subject}, please answer the subsequent multiple-choice question with 'A', 'B', 'C', or 'D'.

{context}Question: {question}
Choices:
        A: {choice1}
        B: {choice2}
        C: {choice3}
        D: {choice4}
Answer: """, doc)