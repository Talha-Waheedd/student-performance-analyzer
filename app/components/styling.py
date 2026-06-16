def get_custom_css():
    return """
    <style>
    /* Premium Dark Theme Adjustments */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        color: #00b4d8;
        font-weight: 700;
    }
    
    .stButton>button {
        background-color: #0077b6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #00b4d8;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 180, 216, 0.4);
    }
    
    /* Metrics Cards */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
        color: #90e0ef;
    }
    
    /* Risk Badges */
    .risk-low {
        background-color: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .risk-medium {
        background-color: rgba(241, 196, 15, 0.2);
        color: #f1c40f;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .risk-high {
        background-color: rgba(231, 76, 60, 0.2);
        color: #e74c3c;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    </style>
    """
