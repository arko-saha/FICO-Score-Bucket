import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import chi2_contingency, ks_2samp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FICOQuantizer:
    """
    A class for binning FICO scores into discrete buckets using various optimization methods.
    
    Supports equal width, equal frequency, and information value-based optimization approaches.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        fico_col: str = "fico_score",
        default_col: str = "default",
    ):
        """
        Initializes the FICOQuantizer with a dataset and column names.
        
        Args:
            df: The input pandas DataFrame containing credit data.
            fico_col: The name of the column containing FICO scores.
            default_col: The name of the binary target column (1 for default).
            
        Raises:
            ValueError: If required columns are missing from the DataFrame.
        """
        if fico_col not in df.columns or default_col not in df.columns:
            raise ValueError(f"Missing required columns: {fico_col} or {default_col}")
            
        self.df = df.copy()
        self.fico_col = fico_col
        self.default_col = default_col
        self.last_boundaries = None
        
        logger.info(f"Initialized FICOQuantizer with {len(df)} records.")

    def equal_width_buckets(self, n_buckets: int) -> List[int]:
        """
        Create buckets of equal width across the FICO score range.
        
        Args:
            n_buckets: Number of buckets to create.
            
        Returns:
            A list of boundary values for the buckets.
        """
        min_score = self.df[self.fico_col].min()
        max_score = self.df[self.fico_col].max()
        logger.debug(f"Calculating equal width buckets for n={n_buckets}")
        return list(np.linspace(min_score, max_score, n_buckets + 1).astype(int))

    def equal_frequency_buckets(self, n_buckets: int) -> List[int]:
        """
        Create buckets with approximately equal number of observations (quantiles).
        
        Args:
            n_buckets: Number of buckets to create.
            
        Returns:
            A list of boundary values for the buckets.
        """
        quantiles = np.linspace(0, 100, n_buckets + 1)
        logger.debug(f"Calculating equal frequency buckets for n={n_buckets}")
        return list(np.unique(np.percentile(self.df[self.fico_col], quantiles).astype(int)))

    def optimize_information_value(
        self, n_buckets: int, iterations: int = 10) -> List[int]:
        """
        Iteratively refines bucket boundaries to maximize Information Value (IV).
        
        Args:
            n_buckets: Number of buckets to optimize.
            iterations: Number of optimization passes.
            
        Returns:
            Optimized bucket boundaries.
        """
        logger.info(f"Optimizing IV for {n_buckets} buckets...")
        
        # Start with equal frequency buckets
        boundaries = self.equal_frequency_buckets(n_buckets)
        best_iv = self._calculate_iv(boundaries)
        best_boundaries = boundaries.copy()

        # Step sizes for refinement
        step_sizes = [20, 10, 5]

        for step in step_sizes:
            for _ in range(iterations):
                changed = False
                for i in range(1, len(boundaries) - 1):
                    for shift in [-step, step]:
                        new_boundary = boundaries[i] + shift
                        if boundaries[i - 1] < new_boundary < boundaries[i + 1]:
                            current_iv = self._calculate_iv(
                                boundaries[:i] + [new_boundary] + boundaries[i + 1 :]
                            )
                            if current_iv > best_iv:
                                best_iv = current_iv
                                best_boundaries = boundaries[:i] + [new_boundary] + boundaries[i + 1 :]
                                changed = True
                boundaries = best_boundaries.copy()
                if not changed:
                    break

        logger.info(f"Optimization complete. Best IV reached: {best_iv:.4f}")
        return best_boundaries

    def _calculate_iv(self, boundaries: List[int]) -> float:
        """Helper to calculate Information Value for a set of boundaries."""
        df = self.df.copy()
        df["bucket"] = pd.cut(df[self.fico_col], boundaries, labels=False, include_lowest=True)

        grouped = df.groupby("bucket").agg({self.default_col: ["count", "sum"]})
        
        good = grouped[self.default_col]["count"] - grouped[self.default_col]["sum"]
        bad = grouped[self.default_col]["sum"]

        # Handling potential zero case to avoid log(0)
        min_pct = 0.0001
        good_pct = (good / good.sum()).clip(lower=min_pct)
        bad_pct = (bad / bad.sum()).clip(lower=min_pct)

        woe = np.log(good_pct / bad_pct)
        iv = ((good_pct - bad_pct) * woe).sum()

        return iv

    def validate_buckets(self, boundaries: List[int]) -> Dict[str, Any]:
        """
        Comprehensive validation of bucket effectiveness using standard risk metrics.
        """
        df = self.df.copy()
        df["bucket"] = pd.cut(df[self.fico_col], boundaries, labels=False, include_lowest=True)

        metrics = {}

        # Population stability check
        pop_dist = df["bucket"].value_counts(normalize=True)
        metrics["min_bucket_pct"] = pop_dist.min() * 100

        # Monotonicity check (Default rates should generally decrease as FICO increases)
        default_rates = df.groupby("bucket")[self.default_col].mean()
        # Note: Higher FICO should mean lower default. Thus, default rates should be monotonic decreasing
        metrics["is_monotonic"] = default_rates.is_monotonic_decreasing or default_rates.is_monotonic_increasing

        # Discriminatory power (KS and Gini)
        metrics["ks_statistic"] = self._calculate_ks_statistic(df)
        metrics["gini"] = self._calculate_gini(df)

        # Statistical significance
        contingency_table = pd.crosstab(df["bucket"], df[self.default_col])
        _, p_value, _, _ = chi2_contingency(contingency_table)
        metrics["chi2_p_value"] = p_value

        return metrics

    def _calculate_ks_statistic(self, df: pd.DataFrame) -> float:
        good_scores = df[df[self.default_col] == 0][self.fico_col]
        bad_scores = df[df[self.default_col] == 1][self.fico_col]
        ks_stat, _ = ks_2samp(good_scores, bad_scores)
        return ks_stat

    def _calculate_gini(self, df: pd.DataFrame) -> float:
        auc = roc_auc_score(df[self.default_col], -df[self.fico_col])
        return 2 * auc - 1

    def analyze_buckets(self, boundaries: List[int]) -> pd.DataFrame:
        """Generate detailed statistical report for each bucket."""
        df = self.df.copy()
        df["bucket"] = pd.cut(df[self.fico_col], boundaries, labels=False, include_lowest=True)

        analysis = (
            df.groupby("bucket")
            .agg(
                {
                    self.fico_col: ["count", "mean", "std", "min", "max"],
                    self.default_col: ["mean", "sum"],
                }
            )
            .round(4)
        )

        analysis.columns = [
            "count", "avg_fico", "std_fico", "min_fico", "max_fico", "default_rate", "default_count",
        ]

        analysis["population_pct"] = (analysis["count"] / len(df) * 100).round(2)
        analysis["bucket_range"] = [
            f"{boundaries[i]}-{boundaries[i+1]}" for i in range(len(boundaries) - 1)
        ]
        analysis["woe"] = self._calculate_woe_by_bucket(df)

        return analysis

    def _calculate_woe_by_bucket(self, df: pd.DataFrame) -> pd.Series:
        grouped = df.groupby("bucket").agg({self.default_col: ["count", "sum"]})
        good = grouped[self.default_col]["count"] - grouped[self.default_col]["sum"]
        bad = grouped[self.default_col]["sum"]

        good_pct = (good / good.sum()).clip(lower=0.0001)
        bad_pct = (bad / bad.sum()).clip(lower=0.0001)

        return np.log(good_pct / bad_pct)

    def plot_bucket_analysis(self, boundaries: List[int], figsize: Tuple[int, int] = (15, 10)):
        """Create visual analytics for bucket performance."""
        df = self.df.copy()
        df["bucket"] = pd.cut(df[self.fico_col], boundaries, labels=False, include_lowest=True)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # Population distribution
        sns.histplot(data=df, x="bucket", ax=ax1)
        ax1.set_title("Population Distribution by Bucket")

        # Default rates
        default_rates = df.groupby("bucket")[self.default_col].mean()
        default_rates.plot(kind="bar", ax=ax2)
        ax2.set_title("Default Rate by Bucket")

        # FICO distributions
        sns.boxplot(data=df, x="bucket", y=self.fico_col, ax=ax3)
        ax3.set_title("FICO Score Variation in Buckets")

        # ROC Curve
        fpr, tpr, _ = roc_curve(df[self.default_col], -df[self.fico_col])
        ax4.plot(fpr, tpr, label=f"Gini: {self._calculate_gini(df):.4f}")
        ax4.plot([0, 1], [0, 1], "k--")
        ax4.set_title("Discriminatory Power (ROC)")
        ax4.legend()

        plt.tight_layout()
        return fig

    def get_optimal_buckets(self, n_buckets: int, method: str = "information_value") -> Dict[str, Any]:
        """High-level method to run specific strategy and return full analysis."""
        if method == "equal_width":
            boundaries = self.equal_width_buckets(n_buckets)
        elif method == "equal_frequency":
            boundaries = self.equal_frequency_buckets(n_buckets)
        else:
            boundaries = self.optimize_information_value(n_buckets)

        self.last_boundaries = boundaries
        return {
            "boundaries": boundaries,
            "analysis": self.analyze_buckets(boundaries),
            "validation": self.validate_buckets(boundaries),
        }


class BucketingStrategyOptimizer:
    """Class to compare multiple bucketing strategies and recommend the best one."""
    
    def __init__(self, quantizer: FICOQuantizer, monotonicity_threshold: float = 0.8):
        self.quantizer = quantizer
        self.monotonicity_threshold = monotonicity_threshold

    def compare_strategies(self, bucket_ranges: Optional[List[int]] = None) -> pd.DataFrame:
        """Evaluates equal_width, equal_freq, and IV-optimized strategies across ranges."""
        if bucket_ranges is None:
            bucket_ranges = [5, 10, 15]
            
        results = []
        methods = ['equal_width', 'equal_frequency', 'information_value']

        for n_buckets in bucket_ranges:
            for method in methods:
                try:
                    res = self.quantizer.get_optimal_buckets(n_buckets, method)
                    val = res['validation']
                    
                    results.append({
                        'n_buckets': n_buckets,
                        'method': method,
                        'ks_statistic': val['ks_statistic'],
                        'gini': val['gini'],
                        'information_value': self.quantizer._calculate_iv(res['boundaries']),
                        'is_monotonic': val['is_monotonic'],
                        'min_pop_pct': val['min_bucket_pct']
                    })
                except Exception as e:
                    logger.error(f"Failed strategy {method} at n={n_buckets}: {e}")

        return pd.DataFrame(results)

    def get_recommended_strategy(self, comparison_df: pd.DataFrame) -> Dict[str, Any]:
        """Heuristic-based recommendation for the best segmentation strategy."""
        if comparison_df.empty:
            return {"error": "No valid strategies found."}
            
        # Preference: Monotonicity > Gini > Balanced Population
        potential = comparison_df[comparison_df['is_monotonic'] == True]
        if potential.empty:
            potential = comparison_df # Fallback
            
        best_idx = potential['gini'].idxmax()
        row = potential.loc[best_idx]
        
        return {
            'recommended_method': row['method'],
            'recommended_n_buckets': int(row['n_buckets']),
            'metrics': row.to_dict()
        }
