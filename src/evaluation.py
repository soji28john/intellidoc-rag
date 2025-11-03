"""
Evaluation Module for RAG system
Assess RAG system performance using retrieval and generation matrics
"""

import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

rouge = Rouge()

def evaluate_retrieval(rag_pipeline, test_questions, ground_truth_keywords, k=5):
    precision_scores, recall_scores, mrr_scores = [], [], []
    for question, ground_truth in zip(test_questions, ground_truth_keywords):
        results = rag_pipeline.query(question, return_chunks=True)
        retrieved_texts = results["retrieved_texts"][:k]
        
        relevant = sum(any(kw.lower() in txt.lower() for kw in ground_truth)
                       for txt in retrieved_texts)
        precision = relevant / k
        recall = relevant / len(ground_truth)
        
        # MRR calculation
        rank = None
        for i, txt in enumerate(retrieved_texts):
            if any(kw.lower() in txt.lower() for kw in ground_truth):
                rank = i + 1
                break
        mrr = 1 / rank if rank else 0
        precision_scores.append(precision)
        recall_scores.append(recall)
        mrr_scores.append(mrr)
    return {
        'Precision@k': round(np.mean(precision_scores),3),
        'Recall@k': round(np.mean(recall_scores),3),
        'MRR': round(np.mean(mrr_scores),3)
    }
    """
    Evaluate retrieval quality
    
    Metrics:
    - Precision@k: Relevant docs in top-k results
    - Recall@k: % of relevant docs retrieved
    - MRR (Mean Reciprocal Rank)
    """
    

def evaluate_generation(generated_answers, reference_answers):
    bleu_scores, rouge_scores = [], []
    for gen, ref in zip(generated_answers, reference_answers):
        if isinstance(gen, dict):
            gen = gen.get("answer", "")
        bleu_scores.append(sentence_bleu([ref.split()], gen.split()))
        rouge_scores.append(rouge.get_scores(gen, ref)[0]['rouge-l']['f'])
    return {
        'BLEU': round(np.mean(bleu_scores),3),
        'ROUGE-L': round(np.mean(rouge_scores),3)
    }
    """
    Evaluate answer quality
    
    Metrics:
    - BLEU score
    - ROUGE score
    - BERTScore (semantic similarity)
    - Human evaluation (qualitative)
    """
   