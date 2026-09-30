"""
dataset_utils.py - Dataset sanitization, wire-service debiasing, and benchmark data loader.
"""

import re
import pandas as pd

# Regex patterns for common wire service and dataset leakage datelines
WIRE_PATTERNS = [
    r"^[A-Z\s]+(?:\([A-Za-z\s]+\))?\s*[-–—]\s*(?:Reuters\s*[-–—])?",
    r"\([A-Za-z\s]*Reuters[A-Za-z\s]*\)\s*[-–—]?",
    r"\bReuters\b\s*[-–—]?",
    r"\bAssociated Press\b\s*[-–—]?",
    r"\b\(AP\)\b\s*[-–—]?",
    r"\bAP\s*[-–—]",
    r"\b\(AFP\)\b\s*[-–—]?",
    r"^(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s*[-–—]",
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+[A-Za-z]+\s+\d+\s+[-–—]",
]

def scrub_wire_bias(text: str) -> str:
    """
    Strips wire service datelines, publisher watermarks, and weekday shortcuts
    that cause artificial dataset leakage and shortcut learning in ML classifiers.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    cleaned = text
    for pattern in WIRE_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
        
    # Remove redundant whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# Curated benchmark dataset representing balanced, debiased articles
BENCHMARK_SAMPLES = [
    # REAL NEWS SAMPLES
    {
        "title": "NASA James Webb Space Telescope Detects Water Vapor in Rocky Planet Zone",
        "text": "Astronomers using NASA's James Webb Space Telescope have detected water vapor in the inner disk of the planetary system PDS 70. This system is located 370 light-years away and hosts two gas giant planets. The detection indicates that water is present in the region where terrestrial planets may be forming, providing critical evidence regarding the origin of water on habitable worlds.",
        "label": 1  # Real
    },
    {
        "title": "Federal Reserve Holds Interest Rates Steady, Citing Inflation Progress",
        "text": "The Federal Reserve concluded its two-day policy meeting by leaving the benchmark federal funds rate unchanged. Central bank officials stated that while economic indicators remain resilient, continued evidence of moderating inflation is needed before implementing rate cuts. Chairman Jerome Powell emphasized that policy decisions will remain data-dependent on upcoming employment and consumer price index figures.",
        "label": 1  # Real
    },
    {
        "title": "European Union Reaches Historic Agreement on Comprehensive Artificial Intelligence Act",
        "text": "European lawmakers and member states concluded negotiations on the Artificial Intelligence Act, establishing the world's first comprehensive legal framework for AI governance. The legislation classifies AI applications by risk categories, imposing strict transparency obligations on general-purpose models and outright bans on biometric surveillance systems deemed high-risk to fundamental human rights.",
        "label": 1  # Real
    },
    {
        "title": "Renewable Energy Capacity Surpasses Coal in Global Generation Mix Report",
        "text": "Global renewable electricity generation surpassed 30 percent for the first time, driven by record installations of solar photovoltaic panels and wind turbines, according to the International Energy Agency's annual review. The report noted that clean energy additions grew by fifty percent over the past twelve months, though grid interconnection bottlenecks remain a challenge in several major economies.",
        "label": 1  # Real
    },
    {
        "title": "World Health Organization Issues Updated Guidance on Pediatric Vaccine Schedules",
        "text": "The Strategic Advisory Group of Experts on Immunization at the World Health Organization released revised global recommendations regarding seasonal influenza and routine childhood immunizations. The updated guidelines emphasize expanding coverage in developing nations and optimizing booster intervals based on multi-year longitudinal efficacy trials conducted across six continents.",
        "label": 1  # Real
    },
    {
        "title": "Semiconductor Manufacturers Announce Multi-Billion Dollar Fab Construction in Ohio",
        "text": "Leading chip manufacturers have commenced excavation on two state-of-the-art semiconductor fabrication facilities in central Ohio. The commercial investment, supported by federal CHIPS Act subsidies, aims to manufacture leading-edge silicon wafers domestically, strengthening domestic supply chains and establishing regional engineering research clusters.",
        "label": 1  # Real
    },
    {
        "title": "Department of Transportation Proposes Mandatory Compensation for Flight Cancellations",
        "text": "Federal transportation regulators published a notice of proposed rulemaking that would mandate commercial airlines provide cash compensation and hotel accommodations when flights are canceled or significantly delayed for reasons within the carrier's control. Aviation consumer advocacy groups praised the initiative while industry associations raised concerns about ticket cost pressures.",
        "label": 1  # Real
    },
    {
        "title": "Marine Biologists Document Coral Spawning Event Along Protected Barrier Reef",
        "text": "Scientists conducting nighttime underwater surveys reported a mass synchronized spawning event among thousands of coral colonies across the protected marine sanctuary. Researchers collected gamete samples to assess genetic diversity and thermal tolerance traits in response to fluctuating sea surface temperatures.",
        "label": 1  # Real
    },

    # FAKE / CLICKBAIT / FABRICATED SAMPLES
    {
        "title": "SHOCKING PROOF: Government Secretly Spraying Chemical Compounds to Control Thoughts!",
        "text": "BOMBSHELL REVELATION!! You won't believe what whistleblowers have finally revealed! Declassified documents PROVE that military planes have been spraying mind-altering toxins into the atmosphere every Tuesday! The mainstream media is completely silent while millions suffer! Share this urgent report before it gets banned everywhere forever!!",
        "label": 0  # Fake
    },
    {
        "title": "MIRACLE FRUIT CURES CANCER IN 48 HOURS: Doctors Banned from Revealing Truth!",
        "text": "Big Pharma doesn't want you to know about this exotic rainforest berry that destroys 100% of cancer cells in just two days! Corrupt health officials tried to suppress the clinical trial results because pills make them trillions. Thousands of patients are cured overnight while hospitals hide the secret formula. Click here now for the full leaked report before authorities take it down!",
        "label": 0  # Fake
    },
    {
        "title": "Celebrity Arrested at Airport with Millions in Illicit Gold Bars Following Standoff",
        "text": "BREAKING: A famous Hollywood star was handcuffed and dragged into custody after customs agents found hundreds of unrecorded gold ingots hidden inside their private luggage! Eyewitnesses claim federal marshals surrounded the tarmac in a chaotic standoff that lasted four hours. Neither airport officials nor local police will comment on the top-secret incident.",
        "label": 0  # Fake
    },
    {
        "title": "Secret Underground City Discovered Beneath Antarctic Ice Sheet with Advanced Technology",
        "text": "Satellite radar scans have exposed a colossal subterranean civilization buried two miles beneath the Antarctic ice shelf! Explorers report finding glowing crystal towers and extraterrestrial machinery operating in complete isolation for millennia. World leaders allegedly held an emergency secret summit in Switzerland to coordinate a global cover-up.",
        "label": 0  # Fake
    },
    {
        "title": "Local Politician Caught on Secret Video Admitting to Staging Entire Election",
        "text": "ANONYMOUS SOURCE LEAKS EXPLOSIVE TAPE! Watch this devastating video where the mayor openly brags about bribing election monitors and swapping ballot boxes in the basement! The corrupt political establishment is in total panic as the internet explodes with outrage. MUST SEE before it is deleted from the web!",
        "label": 0  # Fake
    },
    {
        "title": "5G Towers Emit Frequencies That Alter Human DNA According to Banned Scientist",
        "text": "WARNING TO ALL CITIZENS: A renowned whistleblower physicist was forced into hiding after exposing that telecommunication towers transmit classified micro-frequencies designed to disrupt cellular repair! Mainstream fact-checkers are working day and night to censor the scientific data! Read the leaked laboratory PDF immediately before censorship wipes it!",
        "label": 0  # Fake
    },
    {
        "title": "Ancient Alien Pyramid Found on Moon Surface in Uncensored Apollo Transmission",
        "text": "REVEALED AT LAST: Leaked high-resolution lunar photography reveals a three-sided pyramid made of black obsidian sitting quietly inside Tycho crater! NASA mission control allegedly scrubbed the audio feeds to prevent mass public panic during the broadcast. Military insiders confirm alien technology extraction is underway!",
        "label": 0  # Fake
    },
    {
        "title": "Drinking Lemon Juice with Baking Soda Dissolves All Chronic Diseases Overnight",
        "text": "URGENT HEALTH HACK: Hospitals are furious that this simple two-ingredient kitchen trick is wiping out chronic illnesses in 24 hours! No more expensive medications or doctor visits! Big healthcare conglomerates are lobbying governments to classify lemons as prescription-only! Share this life-saving truth before corrupt regulators delete it!",
        "label": 0  # Fake
    }
]


def get_benchmark_dataframe(clean_bias: bool = True) -> pd.DataFrame:
    """Returns a pandas DataFrame of the curated benchmark samples."""
    df = pd.DataFrame(BENCHMARK_SAMPLES)
    if clean_bias:
        df["text"] = df["text"].apply(scrub_wire_bias)
        df["title"] = df["title"].apply(scrub_wire_bias)
    return df


def load_custom_dataset(filepath: str, clean_bias: bool = True) -> pd.DataFrame:
    """
    Loads and standardizes an external CSV dataset.
    Expected columns: either ('title', 'text', 'label') or ('text', 'label').
    """
    df = pd.read_csv(filepath)
    # Normalize column names
    col_map = {col: col.lower().strip() for col in df.columns}
    df = df.rename(columns=col_map)
    
    if "label" not in df.columns:
        raise ValueError("Dataset CSV must contain a 'label' column (0 for Fake, 1 for Real).")
    
    # Text aggregation
    if "title" in df.columns and "text" in df.columns:
        df["full_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    elif "text" in df.columns:
        df["full_text"] = df["text"].fillna("")
    else:
        raise ValueError("Dataset CSV must contain a 'text' column.")

    if clean_bias:
        df["full_text"] = df["full_text"].apply(scrub_wire_bias)

    return df
