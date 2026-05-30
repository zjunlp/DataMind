"""
Semantic similarity evaluation metrics using embeddings.
"""

import numpy as np
from typing import Optional, List
from .base import BaseMetric, MetricResult


class SemanticSimilarityMetric(BaseMetric):
    """
    Semantic similarity metric using sentence embeddings.
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.8,
        **kwargs
    ):
        """
        Initialize semantic similarity metric.
        
        Args:
            model_name: Name of the sentence transformer model
            similarity_threshold: Threshold for considering answers similar
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self._model = None
    
    @property
    def name(self) -> str:
        return "semantic_similarity"
    
    @property
    def supports_batch_evaluation(self) -> bool:
        return True  # Sentence transformers support batch encoding
    
    def _get_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for semantic similarity. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    def _compute_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def evaluate(
        self, 
        prediction: str, 
        ground_truth: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs
    ) -> MetricResult:
        """
        Evaluate semantic similarity between prediction and ground truth.
        
        Args:
            prediction: Model prediction
            ground_truth: Ground truth answer
            query: Original query (unused)
            **kwargs: Additional context
            
        Returns:
            MetricResult with similarity score
        """
        if ground_truth is None:
            return MetricResult(
                metric_name=self.name,
                score=None,
                details={"reason": "No ground truth available"}
            )
        
        if not prediction.strip() or not ground_truth.strip():
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                details={"reason": "Empty prediction or ground truth"}
            )
        
        try:
            model = self._get_model()
            
            # Encode texts
            embeddings = model.encode([prediction, ground_truth])
            pred_embedding = embeddings[0]
            truth_embedding = embeddings[1]
            
            # Compute similarity
            similarity = self._compute_cosine_similarity(pred_embedding, truth_embedding)
            
            # Determine if it's a match based on threshold
            is_match = similarity >= self.similarity_threshold
            match_score = 1.0 if is_match else 0.0
            
            details = {
                "cosine_similarity": similarity,
                "similarity_threshold": self.similarity_threshold,
                "is_semantic_match": is_match,
                "model_name": self.model_name,
                "embedding_dim": len(pred_embedding),
            }
            
            # For continuous evaluation, use similarity as score
            # For binary evaluation, use match_score
            score = similarity  # Can be changed to match_score for binary evaluation
            
            return MetricResult(
                metric_name=self.name,
                score=score,
                details=details
            )
            
        except Exception as e:
            return MetricResult(
                metric_name=self.name,
                score=None,
                error=str(e),
                details={"model_name": self.model_name}
            )
    
    def evaluate_batch(
        self,
        predictions: List[str],
        ground_truths: List[str],
        queries: Optional[List[str]] = None,
        **kwargs
    ) -> List[MetricResult]:
        """
        Evaluate multiple predictions in batch for efficiency.
        
        Args:
            predictions: List of model predictions
            ground_truths: List of ground truth answers
            queries: List of original queries (unused)
            **kwargs: Additional context
            
        Returns:
            List of MetricResult objects
        """
        try:
            model = self._get_model()
            
            # Filter out None ground truths and track indices
            valid_indices = []
            valid_predictions = []
            valid_ground_truths = []
            
            for i, (pred, truth) in enumerate(zip(predictions, ground_truths)):
                if truth is not None and pred.strip() and truth.strip():
                    valid_indices.append(i)
                    valid_predictions.append(pred)
                    valid_ground_truths.append(truth)
            
            results = []
            
            if valid_predictions:
                # Batch encode all texts
                all_texts = valid_predictions + valid_ground_truths
                embeddings = model.encode(all_texts)
                
                # Split embeddings
                pred_embeddings = embeddings[:len(valid_predictions)]
                truth_embeddings = embeddings[len(valid_predictions):]
                
                # Compute similarities
                valid_results = []
                for pred_emb, truth_emb, pred, truth in zip(
                    pred_embeddings, truth_embeddings, valid_predictions, valid_ground_truths
                ):
                    similarity = self._compute_cosine_similarity(pred_emb, truth_emb)
                    is_match = similarity >= self.similarity_threshold
                    
                    details = {
                        "cosine_similarity": similarity,
                        "similarity_threshold": self.similarity_threshold,
                        "is_semantic_match": is_match,
                        "model_name": self.model_name,
                        "embedding_dim": len(pred_emb),
                    }
                    
                    valid_results.append(MetricResult(
                        metric_name=self.name,
                        score=similarity,
                        details=details
                    ))
            
            # Reconstruct full results list with None placeholders
            valid_iter = iter(valid_results) if valid_predictions else iter([])
            for i in range(len(predictions)):
                if i in valid_indices:
                    results.append(next(valid_iter))
                else:
                    # Handle invalid cases
                    if ground_truths[i] is None:
                        reason = "No ground truth available"
                    else:
                        reason = "Empty prediction or ground truth"
                    
                    results.append(MetricResult(
                        metric_name=self.name,
                        score=None if ground_truths[i] is None else 0.0,
                        details={"reason": reason}
                    ))
            
            return results
            
        except Exception as e:
            # Fallback to individual evaluation on error
            return [
                MetricResult(
                    metric_name=self.name,
                    score=None,
                    error=str(e),
                    details={"model_name": self.model_name}
                )
                for _ in predictions
            ]


class BinarySemanticSimilarityMetric(SemanticSimilarityMetric):
    """
    Binary version of semantic similarity metric (returns 1.0 or 0.0).
    """
    
    @property
    def name(self) -> str:
        return "binary_semantic_similarity"
    
    def evaluate(
        self, 
        prediction: str, 
        ground_truth: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs
    ) -> MetricResult:
        """
        Evaluate binary semantic similarity.
        """
        result = super().evaluate(prediction, ground_truth, query, **kwargs)
        
        # Convert continuous similarity to binary score
        if result.score is not None:
            binary_score = 1.0 if result.score >= self.similarity_threshold else 0.0
            result.score = binary_score
            result.details["binary_score"] = binary_score
            result.details["continuous_similarity"] = result.details.get("cosine_similarity")
        
        return result