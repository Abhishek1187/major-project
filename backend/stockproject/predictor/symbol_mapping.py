# Mapping from stock symbols to company names for news queries
symbol_to_company_name = {
    "RELIANCE.NS": "RELIANCE.NS",
    "AXISBANK.NS": "AXISBANK.NS",
    "HDFCBANK.NS": "HDFCBANK.NS",
    "ONGC.NS": "ONGC.NS",
    "SBIN.NS": "SBIN.NS",
    "INFY.NS": "INFY.NS",
    "ADANIPORTS.NS": "ADANIPORTS.NS",
    "ADANIENT.NS": "ADANIENT.NS",
    "BAJFINANCE.NS": "BAJFINANCE.NS",
    "BHARTIARTL.NS": "BHARTIARTL.NS",
    # Add more mappings as needed
}

def get_company_name_from_symbol(symbol: str) -> str:
    """
    Convert stock symbol to company name for news query.
    If symbol not found, return symbol itself.
    """
    return symbol_to_company_name.get(symbol.upper(), symbol.upper())
