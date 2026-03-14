import pytest
import math
from stationarity_toolkit.results import TestResult, DetectionResult


def test_test_result_structure():
    """Test TestResult has correct attributes."""
    result = TestResult(
        test_name="ADF Test",
        statistic=-3.5,
        p_value=0.01,
        is_stationary=True,
        interpretation="Series is stationary",
        educational_note="ADF tests for unit root"
    )
    
    assert isinstance(result.test_name, str)
    assert isinstance(result.statistic, float)
    assert isinstance(result.p_value, float)
    assert isinstance(result.is_stationary, bool)
    assert isinstance(result.interpretation, str)
    assert isinstance(result.educational_note, str)
    assert 0 <= result.p_value <= 1


def test_test_result_with_nan_p_value():
    """Test TestResult accepts nan p_value."""
    result = TestResult(
        test_name="Test",
        statistic=1.0,
        p_value=float('nan'),
        is_stationary=False,
        interpretation="",
        educational_note=""
    )
    
    assert math.isnan(result.p_value)


def test_detection_result_structure():
    """Test DetectionResult has correct attributes."""
    test1 = TestResult("ADF", -3.5, 0.01, True, "Stationary", "Note")
    test2 = TestResult("KPSS", 0.2, 0.1, True, "Stationary", "Note")
    
    result = DetectionResult(
        trend_stationary=True,
        variance_stationary=True,
        seasonal_stationary=True,
        tests={'trend': [test1], 'variance': [test2], 'seasonal': []}
    )
    
    assert isinstance(result.trend_stationary, bool)
    assert isinstance(result.variance_stationary, bool)
    assert isinstance(result.seasonal_stationary, bool)
    assert isinstance(result.tests, dict)


def test_detection_result_tests_verbose():
    """Test DetectionResult.tests with verbose mode (list of TestResult)."""
    test1 = TestResult("ADF", -3.5, 0.01, True, "Stationary", "Note")
    test2 = TestResult("KPSS", 0.2, 0.1, True, "Stationary", "Note")
    
    result = DetectionResult(
        trend_stationary=True,
        variance_stationary=False,
        seasonal_stationary=True,
        tests={'trend': [test1, test2], 'variance': [], 'seasonal': []}
    )
    
    assert len(result.tests['trend']) == 2
    assert all(isinstance(t, TestResult) for t in result.tests['trend'])
    assert result.tests['trend'][0].test_name == "ADF"
    assert result.tests['trend'][1].test_name == "KPSS"


def test_detection_result_tests_non_verbose():
    """Test DetectionResult.tests with non-verbose mode (empty lists)."""
    result = DetectionResult(
        trend_stationary=True,
        variance_stationary=True,
        seasonal_stationary=False,
        tests={'trend': [], 'variance': [], 'seasonal': []}
    )
    
    assert result.tests['trend'] == []
    assert result.tests['variance'] == []
    assert result.tests['seasonal'] == []


def test_summary_property():
    result = DetectionResult(True, False, True, tests={})
    s = result.summary
    assert "Trend Stationary: ✅ Yes" in s
    assert "Variance Stationary: ❌ No" in s
    assert "Seasonal Stationary: ✅ Yes" in s


def test_report_returns_dataframe():
    t1 = TestResult("ADF", -3.5, 0.01, True, "Stationary", "Note")
    t2 = TestResult("Levene", 2.0, 0.04, False, "Non-stationary", "Variance note")
    result = DetectionResult(True, False, True, tests={
        'trend': [t1], 'variance': [t2], 'seasonal': []
    })
    df = result.report()
    assert len(df) == 2
    assert df.iloc[0]['Test'] == "ADF"
    assert df.iloc[1]['Test'] == "Levene"


def test_report_writes_markdown(tmp_path):
    t1 = TestResult("ADF", -3.5, 0.01, True, "Stationary", "Note")
    t2 = TestResult("Levene", 2.0, 0.04, False, "Non-stationary", "Variance note")
    result = DetectionResult(True, False, True, tests={
        'trend': [t1], 'variance': [t2], 'seasonal': []
    })
    filepath = str(tmp_path / "report.md")
    result.report(filepath=filepath)
    with open(filepath) as f:
        content = f.read()
    assert "# Stationarity Detection Report" in content
    assert "All tests passed ✅" in content
    assert "Levene" in content
    assert "❌ Non-stationary" in content
