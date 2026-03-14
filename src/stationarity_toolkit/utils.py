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
        if 'W' in freq_str or 'Week' in freq_str:
            return [52]
        elif 'Day' in freq_str or freq_str.startswith('D') or 'Business' in freq_str or freq_str.startswith('B'):
            return [7, 30, 365]
        elif 'Hour' in freq_str or freq_str.startswith('H') or freq_str.startswith('h'):
            return [24, 168]
        elif 'Quarter' in freq_str or freq_str.startswith('Q'):
            return [4]
        elif 'Month' in freq_str or freq_str.startswith('M'):
            return [3, 6, 12]
    
    return [7, 12, 30, 52, 365]
