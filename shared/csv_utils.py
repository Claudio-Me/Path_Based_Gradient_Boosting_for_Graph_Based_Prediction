"""CSV output utilities with timestamp-based backup mechanism."""
import csv
import os
from datetime import datetime
from typing import Dict, List, Union


def get_timestamped_path(output_dir: str, prefix: str) -> str:
    """
    Generate timestamped CSV path.

    Format: {output_dir}/{prefix}_Performance_{YYYYMMDD_HHMMSS}.csv

    This ensures each run creates a unique file, preventing overwrites
    and preserving all previous results as backups.

    Args:
        output_dir: Directory for output (e.g., 'PathBoost_results')
        prefix: Filename prefix (e.g., 'Sequential_PathBoost')

    Returns:
        Full path like 'PathBoost_results/Sequential_PathBoost_Performance_20241227_143022.csv'
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_Performance_{timestamp}.csv"
    return os.path.join(output_dir, filename)


class ResultsCSVWriter:
    """
    CSV writer that handles the required output format.

    Columns: Dataset, Metric, Mean, Std_Top10, Std_All100

    One row per metric per dataset. This long-format structure makes it
    easy to analyze and compare metrics across datasets.
    """

    FIELDNAMES = ['Dataset', 'Metric', 'Mean', 'Std_Top10', 'Std_All100']

    def __init__(self, csv_path: str):
        """
        Initialize the CSV writer.

        Args:
            csv_path: Path to the CSV file (created immediately with header)
        """
        self.csv_path = csv_path
        self._write_header()

    def _write_header(self):
        """Write CSV header."""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()

    def write_results(
        self,
        dataset: str,
        results: Dict[str, tuple],
    ):
        """
        Write results for a dataset.

        Args:
            dataset: Dataset name
            results: Dict mapping metric name to (mean, std_top10, std_all100) tuple

        Example:
            writer.write_results("MUTAG", {
                "accuracy": (0.857, 0.012, 0.034),
                "f1_macro": (0.849, 0.013, 0.035),
            })
        """
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            for metric, values in results.items():
                mean, std_top10, std_all100 = values
                writer.writerow({
                    'Dataset': dataset,
                    'Metric': metric,
                    'Mean': f'{mean:.6f}' if isinstance(mean, (int, float)) else mean,
                    'Std_Top10': f'{std_top10:.6f}' if isinstance(std_top10, (int, float)) else std_top10,
                    'Std_All100': f'{std_all100:.6f}' if isinstance(std_all100, (int, float)) else std_all100,
                })

    def write_failure(self, dataset: str, metrics: List[str], reason: str = "FAILED"):
        """
        Write failure row(s) for a dataset.

        Args:
            dataset: Dataset name
            metrics: List of metric names to write failure rows for
            reason: Failure reason ('FAILED' or 'TIMEOUT')

        Example:
            writer.write_failure("PTC_MR", ["accuracy", "f1"], "TIMEOUT")
        """
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            for metric in metrics:
                writer.writerow({
                    'Dataset': dataset,
                    'Metric': metric,
                    'Mean': reason,
                    'Std_Top10': reason,
                    'Std_All100': reason,
                })
