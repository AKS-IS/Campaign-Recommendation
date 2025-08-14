# Keyword Optimization System

An intelligent AI-powered system that analyzes and optimizes advertising keywords to improve campaign performance and reduce wasted ad spend.

## 🎯 What It Does

This system helps you:
- **Identify poor-performing keywords** using fixed thresholds and comprehensive scoring
- **Find better replacement keywords** with AI-generated suggestions and performance predictions
- **Understand keyword intent** (TRANSACTIONAL, NAVIGATIONAL, INFORMATIONAL) for better targeting
- **Predict performance metrics** (CTR, CPC, conversion rate) for new keywords in original scale
- **Optimize match types** with LLM-determined EXACT, PHRASE, or BROAD recommendations

## 🚀 Key Features

### LLM-Powered Analysis
- **Ad-Group Theme Derivation**: Analyzes all headlines and descriptions to extract core campaign theme
- **Relevance Scoring**: 0-10 scale scoring for brand, product, theme alignment, and competition
- **Intent Classification**: Automatically classifies keywords by user intent (buying, researching, browsing)
- **Match Type Prediction**: Uses LLM to determine optimal match type for replacement keywords

### Dynamic Performance Analysis
- **Multi-Metric Scoring**: Incorporates CTR, CPC, and conversion rate with configurable weights
- **Dynamic Normalization**: Scales performance metrics using actual data bounds from your payload
- **Fixed Thresholds**: Configurable negative keyword identification (default: 3.0 on 0-10 scale)
- **Performance Prediction**: Estimates CTR, CPC, and conversion for replacement keywords

### Intelligent Optimization
- **Relevance-Based Replacement Scoring**: Scores new keywords using only relevance metrics (no historical data)
- **Market Segment Context**: Competition analysis considers your specific brand and product categories
- **Comprehensive Response**: Returns retained, negative, and replacement keywords with full metrics
- **Configurable Weights**: Adjustable scoring weights via environment variables

## 🔄 Complete Flow: From Input to Output

### Step 1: Input Data
```json
{
  "keywords": [
    {
      "keyword": "running shoes",
      "cpc": 2.50,
      "ctr": 0.045,
      "conversion_rate": 0.08,
      "match_type": "PHRASE"
    },
    {
      "keyword": "athletic sneakers",
      "cpc": 3.20,
      "ctr": 0.038,
      "conversion_rate": 0.06,
      "match_type": "BROAD"
    },
    {
      "keyword": "casual sneakers",
      "cpc": 1.50,
      "ctr": 0.035,
      "conversion_rate": 0.04,
      "match_type": "BROAD"
    },
    {
      "keyword": "trail running shoes",
      "cpc": 2.20,
      "ctr": 0.039,
      "conversion_rate": 0.09,
      "match_type": "EXACT"
    }
  ],
  "brand_category": "athletic footwear",
  "product_category": "sports shoes",
  "headlines": [
    "Premium Running Shoes for Performance",
    "High-Performance Athletic Footwear",
    "Professional Running Gear"
  ],
  "descriptions": [
    "High-quality running shoes designed for maximum comfort and performance during your runs",
    "Advanced athletic footwear engineered for peak performance and durability",
    "Professional-grade running equipment for serious athletes"
  ]
}
```

### Step 2: Data Preprocessing
```python
# Normalize all text (lowercase, remove punctuation)
"running shoes" → "running shoes"
"Premium Running Shoes for Performance" → "premium running shoes for performance"
```

### Step 3: Ad-Group Theme Derivation
```python
# LLM analyzes all headlines and descriptions
Input: "premium running shoes for performance | high-performance athletic footwear | professional running gear"
       "high-quality running shoes designed for maximum comfort and performance during your runs | advanced athletic footwear engineered for peak performance and durability | professional-grade running equipment for serious athletes"

Output: "premium running performance footwear"
```

### Step 4: Performance Metric Normalization
```python
# Calculate dynamic bounds from input data
CTR bounds: 0.035 - 0.045 (3.5% to 4.5%)
CPC bounds: 1.50 - 3.20 (₹1.50 to ₹3.20)
Conversion bounds: 0.04 - 0.09 (4% to 9%)

# Normalize to 0-10 scale
"running shoes": CTR=10.0, CPC=6.2, Conversion=5.7
"athletic sneakers": CTR=3.0, CPC=0.0, Conversion=2.9
"casual sneakers": CTR=0.0, CPC=10.0, Conversion=0.0
"trail running shoes": CTR=4.0, CPC=7.4, Conversion=10.0
```

### Step 5: LLM Relevance Scoring
```python
# For each keyword, get scores (0-10 scale)
"running shoes":
  - Brand Alignment: 9/10 (excellent fit for athletic footwear)
  - Product Alignment: 9/10 (perfect for sports shoes)
  - Theme Alignment: 8/10 (fits premium running theme)
  - Competition Score: 7/10 (competitive but manageable)
  - Intent Classification: TRANSACTIONAL (users want to buy)

"casual sneakers":
  - Brand Alignment: 6/10 (moderate fit for athletic footwear)
  - Product Alignment: 7/10 (somewhat relevant to sports shoes)
  - Theme Alignment: 5/10 (doesn't fit premium running theme)
  - Competition Score: 8/10 (very competitive)
  - Intent Classification: INFORMATIONAL (users researching)
```

### Step 6: Composite Score Calculation
```python
# Weighted formula (using default weights)
Final Score = (0.20 × Brand) + (0.20 × Product) + (0.15 × Theme) 
              - (0.15 × Competition) + (0.10 × CTR) + (0.05 × CPC) + (0.15 × Conversion)

"running shoes": 8.2 = (0.20×9) + (0.20×9) + (0.15×8) - (0.15×7) + (0.10×10) + (0.05×6.2) + (0.15×5.7)
"casual sneakers": 2.5 = (0.20×6) + (0.20×7) + (0.15×5) - (0.15×8) + (0.10×0) + (0.05×10) + (0.15×0)
```

### Step 7: Negative Keyword Identification
```python
# Fixed threshold: 3.0 (from environment variable)
Threshold = 3.0

"running shoes": 8.2 > 3.0 → RETAINED ✅
"athletic sneakers": 6.8 > 3.0 → RETAINED ✅
"casual sneakers": 2.5 < 3.0 → NEGATIVE ❌
"trail running shoes": 7.9 > 3.0 → RETAINED ✅
```

### Step 8: Replacement Keyword Generation
```python
# For "casual sneakers" (negative keyword)
LLM generates: "performance running shoes"

# Calculate relevance scores for replacement
"performance running shoes":
  - Brand Alignment: 8/10
  - Product Alignment: 9/10
  - Theme Alignment: 8/10
  - Competition Score: 6/10
  - Intent Classification: TRANSACTIONAL

# Calculate replacement score (relevance only, no performance)
Replacement Score = (0.20×8) + (0.20×9) + (0.15×8) - (0.15×6) = 7.8

# Predict performance based on retained vs negative keyword differences
Retained avg: CTR=4.1%, CPC=₹2.63, Conv=7.7%
Negative avg: CTR=3.5%, CPC=₹1.50, Conv=4.0%
Predicted: CTR=4.05%, CPC=₹2.20, Conv=6.75% (70% improvement)

# Determine match type
LLM suggests: EXACT (specific, high-intent keyword)
```

### Step 9: Final Output
```json
{
  "ad_group_theme": "premium running performance footwear",
  "negative_keywords": [
    {
      "keyword": "casual sneakers",
      "final_score": 2.5,
      "relevance_scores": {
        "brand_alignment": 6,
        "product_alignment": 7,
        "theme_alignment": 5,
        "competition_score": 8
      },
      "intent_classification": "INFORMATIONAL",
      "match_type": "BROAD",
      "performance_metrics": {
        "normalized_ctr": 0.0,
        "normalized_cpc": 10.0,
        "normalized_conversion_rate": 0.0,
        "original_ctr": 0.035,
        "original_cpc": 1.50,
        "original_conversion_rate": 0.04
      }
    }
  ],
  "retained_keywords": [
    {
      "keyword": "running shoes",
      "final_score": 8.2,
      "relevance_scores": {
        "brand_alignment": 9,
        "product_alignment": 9,
        "theme_alignment": 8,
        "competition_score": 7
      },
      "intent_classification": "TRANSACTIONAL",
      "match_type": "PHRASE",
      "performance_metrics": {
        "normalized_ctr": 10.0,
        "normalized_cpc": 6.2,
        "normalized_conversion_rate": 5.7,
        "original_ctr": 0.045,
        "original_cpc": 2.50,
        "original_conversion_rate": 0.08
      }
    }
  ],
  "replacement_keywords": [
    {
      "keyword": "performance running shoes",
      "replaces": "casual sneakers",
      "relevance_scores": {
        "brand_alignment": 8,
        "product_alignment": 9,
        "theme_alignment": 8,
        "competition_score": 6,
        "intent_classification": "TRANSACTIONAL"
      },
      "composite_score": 7.8,
      "match_type": "EXACT",
      "predicted_performance": {
        "predicted_ctr": 0.0405,
        "predicted_cpc": 2.20,
        "predicted_conversion_rate": 0.0675
      }
    }
  ],
  "optimization_summary": {
    "total_keywords": 4,
    "negative_keywords_count": 1,
    "retained_keywords_count": 3,
    "replacement_keywords_count": 1,
    "average_final_score": 6.8,
    "score_std_dev": 2.1,
    "negative_keywords_list": ["casual sneakers"],
    "retained_keywords_list": ["running shoes", "athletic sneakers", "trail running shoes"],
    "replacement_keywords_list": ["performance running shoes"]
  }
}
```

## 📊 How It Works

### 1. **Data Analysis**
The system analyzes your keywords, headlines, and descriptions to understand your campaign context.

### 2. **Scoring & Classification**
Each keyword gets scored on:
- Brand alignment (how well it fits your brand)
- Product relevance (how well it matches your products)
- Theme fit (how well it matches your campaign theme)
- Competition level (how competitive the keyword is)
- Performance metrics (CTR, CPC, conversion rate)
- User intent (buying, researching, or browsing)

### 3. **Optimization**
- **Identifies negative keywords** that score below your threshold
- **Retains good keywords** that are performing well
- **Generates replacements** for poor-performing keywords
- **Predicts performance** for new keywords

### 4. **Results**
You get a clear report showing:
- Which keywords to keep
- Which keywords to remove
- What new keywords to try
- Expected performance for new keywords



## 🔧 Configuration

The system is highly configurable through environment variables:

- **Scoring Weights**: Adjust importance of different factors
- **Thresholds**: Set your own standards for negative keywords
- **LLM Settings**: Configure AI model parameters


