"""
Evaluation and Color-Coding Engine
Calculates dataset-wide similarity percentage and character-level black/red color statuses.
Follows official formula: Similarity (%) = (1 - Total Red Chars / Total Chars) * 100 >= 95%
"""
from typing import List, Dict, Any, Tuple
from core.dictionary import HanNomDictionary
from core.config import MIN_SIMILARITY_THRESHOLD

class AlignmentEvaluator:
    def __init__(self, aligner=None):
        self.dict_engine = HanNomDictionary.get_instance()

    def evaluate_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single aligned pair:
        Returns verification details, character-level match statuses.
        """
        nom_val = record.get("SinoNom_Char", "")
        qn_val = record.get("ChuQN_txt", "")
        nom_text = "" if nom_val is None or (isinstance(nom_val, float) and str(nom_val) == "nan") else str(nom_val)
        qn_text = "" if qn_val is None or (isinstance(qn_val, float) and str(qn_val) == "nan") else str(qn_val)
        
        v_res = self.dict_engine.verify_sentence_pair(nom_text, qn_text)
        return {
            "total_chars": v_res["total_chars"],
            "matched_chars": v_res["matched_chars"],
            "mismatched_chars": v_res["mismatched_chars"],
            "similarity_pct": v_res["similarity_pct"],
            "is_valid": v_res["is_valid"],
            "char_statuses": v_res["char_statuses"]
        }

    def evaluate_dataset(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate entire dataset across all rows.
        """
        total_chars = 0
        total_red = 0
        total_black = 0
        per_row_eval = []

        for r in records:
            res = self.evaluate_record(r)
            per_row_eval.append(res)
            total_chars += res["total_chars"]
            total_red += res["mismatched_chars"]
            total_black += res["matched_chars"]

        if total_chars > 0:
            dataset_similarity = (1.0 - (total_red / total_chars)) * 100.0
        else:
            dataset_similarity = 100.0

        is_passed = dataset_similarity >= (MIN_SIMILARITY_THRESHOLD * 100.0)

        return {
            "total_chars": total_chars,
            "total_black_chars": total_black,
            "total_red_chars": total_red,
            "similarity_percentage": dataset_similarity,
            "overall_similarity": dataset_similarity / 100.0,
            "is_passed": is_passed,
            "row_evaluations": per_row_eval
        }

    evaluate_corpus = evaluate_dataset
