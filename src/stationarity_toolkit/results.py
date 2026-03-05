from dataclasses import dataclass
import pandas as pd


@dataclass
class TestResult:
    test_name: str
    statistic: float
    p_value: float
    is_stationary: bool
    interpretation: str
    educational_note: str


@dataclass
class DetectionResult:
    trend_stationary: bool
    variance_stationary: bool
    seasonal_stationary: bool
    tests: dict
    
    @property
    def summary(self) -> str:
        """Return 3-line summary of stationarity status."""
        lines = [
            f"Trend Stationary: {'✅ Yes' if self.trend_stationary else '❌ No'}",
            f"Variance Stationary: {'✅ Yes' if self.variance_stationary else '❌ No'}",
            f"Seasonal Stationary: {'✅ Yes' if self.seasonal_stationary else '❌ No'}"
        ]
        return "\n".join(lines)
    
    def report(self, filepath: str = None) -> pd.DataFrame:
        """Generate report as DataFrame and optionally save markdown to file."""
        # Build DataFrame
        rows = []
        for test_type in ['trend', 'variance', 'seasonal']:
            if test_type not in self.tests or not self.tests[test_type]:
                continue
            for t in self.tests[test_type]:
                rows.append({
                    'Test Type': test_type.capitalize(),
                    'Test': t.test_name,
                    'Result': '✅ Stationary' if t.is_stationary else '❌ Non-stationary',
                    'Statistic': f"{t.statistic:.4f}",
                    'P-value': f"{t.p_value:.4f}",
                    'Note': t.educational_note,
                    'Interpretation': t.interpretation
                })
        
        df = pd.DataFrame(rows)
        
        # If filepath provided, write markdown
        if filepath:
            lines = ["# Stationarity Detection Report\n"]
            
            # Summary section
            lines.append("## Summary\n")
            lines.append(f"- Trend Stationary: {'✅ Yes' if self.trend_stationary else '❌ No'}")
            lines.append(f"- Variance Stationary: {'✅ Yes' if self.variance_stationary else '❌ No'}")
            lines.append(f"- Seasonal Stationary: {'✅ Yes' if self.seasonal_stationary else '❌ No'}\n")
            
            # Test results by type
            for test_type, label in [('trend', 'Trend'), ('variance', 'Variance'), ('seasonal', 'Seasonal')]:
                lines.append(f"## {label} Tests\n")
                
                if test_type not in self.tests or not self.tests[test_type]:
                    lines.append("No tests run\n")
                    continue
                
                test_results = self.tests[test_type]
                all_passed = all(t.is_stationary for t in test_results)
                
                if all_passed:
                    lines.append("All tests passed ✅\n")
                else:
                    for t in test_results:
                        result = "✅ Stationary" if t.is_stationary else "❌ Non-stationary"
                        lines.append(f"### {t.test_name}\n")
                        lines.append(f"- Result: {result}")
                        lines.append(f"- Note: {t.educational_note}")
                        lines.append(f"- Interpretation: {t.interpretation}")
                        lines.append(f"- Statistic: {t.statistic:.4f}")
                        lines.append(f"- P-value: {t.p_value:.4f}\n")
            
            content = "\n".join(lines)
            with open(filepath, 'w') as f:
                f.write(content)
        
        return df
