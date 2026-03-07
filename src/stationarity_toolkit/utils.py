import pandas as pd


def get_contextual_periods(ts: pd.Series) -> list[int]:
    """
    Get contextual seasonal periods based on time series frequency.
    
    Args:
        ts: Time series with datetime index
        
    Returns:
        List of expected seasonal periods for the given frequency
    """
    freq = getattr(ts.index, 'freq', None)
    if freq is None:
        freq = pd.infer_freq(ts.index)
    
    if freq:
        freq_str = str(freq)
        # Check for weekly frequency (can be 'W', 'W-MON', '<Week: weekday=0>', etc.)
        if 'W' in freq_str or 'Week' in freq_str:
            return [52]
        elif freq_str.startswith('D') or freq_str.startswith('B'):
            return [7, 30, 365]
        elif freq_str.startswith('H'):
            return [24, 168]
        elif freq_str.startswith('M') and 'Month' not in freq_str:
            return [3, 6, 12]
        elif freq_str.startswith('Q'):
            return [4]
    
    return [7, 12, 30, 52, 365]
