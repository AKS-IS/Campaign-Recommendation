from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import openai
import re
import logging
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI client from environment variables
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID"),
    project=os.getenv("OPENAI_PROJECT_ID")
)

# Load all configuration from environment variables
KEYWORD_BRAND_ALIGNMENT_WEIGHT = float(os.getenv("KEYWORD_BRAND_ALIGNMENT_WEIGHT", "0.20"))
KEYWORD_PRODUCT_ALIGNMENT_WEIGHT = float(os.getenv("KEYWORD_PRODUCT_ALIGNMENT_WEIGHT", "0.20"))
KEYWORD_THEME_ALIGNMENT_WEIGHT = float(os.getenv("KEYWORD_THEME_ALIGNMENT_WEIGHT", "0.15"))
KEYWORD_COMPETITION_WEIGHT = float(os.getenv("KEYWORD_COMPETITION_WEIGHT", "0.15"))
KEYWORD_CTR_WEIGHT = float(os.getenv("KEYWORD_CTR_WEIGHT", "0.10"))
KEYWORD_CPC_WEIGHT = float(os.getenv("KEYWORD_CPC_WEIGHT", "0.05"))
KEYWORD_CONVERSION_WEIGHT = float(os.getenv("KEYWORD_CONVERSION_WEIGHT", "0.15"))

KEYWORD_NEGATIVE_THRESHOLD = float(os.getenv("KEYWORD_NEGATIVE_THRESHOLD", "3.0"))
KEYWORD_REPLACEMENT_COUNT = int(os.getenv("KEYWORD_REPLACEMENT_COUNT", "3"))

KEYWORD_LLM_MODEL = os.getenv("KEYWORD_LLM_MODEL", "gpt-3.5-turbo")
KEYWORD_LLM_MAX_TOKENS = int(os.getenv("KEYWORD_LLM_MAX_TOKENS", "100"))
KEYWORD_LLM_TEMPERATURE = float(os.getenv("KEYWORD_LLM_TEMPERATURE", "0.3"))

logger = logging.getLogger("api.keyword_optimization")

router = APIRouter()

# Data Models
class KeywordData(BaseModel):
    keyword: str
    cpc: float
    ctr: float
    conversion_rate: float
    match_type: Optional[str] = None  # "EXACT", "PHRASE", "BROAD" or None

class AdGroupTheme(BaseModel):
    theme: str
    confidence: float

class RelevanceScore(BaseModel):
    brand_alignment: int
    product_alignment: int
    theme_alignment: int
    competition_score: int
    intent_classification: str  # "transactional", "navigational", "informational"

class KeywordScore(BaseModel):
    keyword: str
    relevance_scores: RelevanceScore
    normalized_ctr: float
    normalized_cpc: float
    normalized_conversion_rate: float
    final_score: float
    is_negative: bool

class KeywordOptimizationInput(BaseModel):
    keywords: List[KeywordData]
    brand_category: str
    product_category: str
    headlines: List[str]
    descriptions: List[str]
    replacement_count: Optional[int] = None

class KeywordOptimizationResponse(BaseModel):
    ad_group_theme: str
    negative_keywords: List[Dict[str, Any]]
    replacement_keywords: List[Dict[str, Any]]
    retained_keywords: List[Dict[str, Any]]
    optimization_summary: Dict[str, Any]

class KeywordOptimizationService:
    def __init__(self):
        # Read weights from environment variables with defaults
        self.default_weights = {
            'brand_alignment': KEYWORD_BRAND_ALIGNMENT_WEIGHT,
            'product_alignment': KEYWORD_PRODUCT_ALIGNMENT_WEIGHT,
            'theme_alignment': KEYWORD_THEME_ALIGNMENT_WEIGHT,
            'competition': KEYWORD_COMPETITION_WEIGHT,
            'ctr': KEYWORD_CTR_WEIGHT,
            'cpc': KEYWORD_CPC_WEIGHT,
            'conversion': KEYWORD_CONVERSION_WEIGHT
        }
        
        # Read threshold configuration
        self.negative_threshold = KEYWORD_NEGATIVE_THRESHOLD
        self.default_replacement_count = KEYWORD_REPLACEMENT_COUNT
        
        # Read LLM configuration
        self.llm_model = KEYWORD_LLM_MODEL
        self.llm_max_tokens = KEYWORD_LLM_MAX_TOKENS
        self.llm_temperature = KEYWORD_LLM_TEMPERATURE
        
        # Log configuration
        print(f"🔧 Configuration loaded:")
        print(f"  📊 Weights: {self.default_weights}")
        print(f"   Negative threshold: {self.negative_threshold}")
        print(f"   Replacement count: {self.default_replacement_count}")
        print(f"  🤖 LLM model: {self.llm_model}")
    
    def normalize_text(self, text: str) -> str:
        """Step 1: Normalize text (lowercase, strip punctuation)"""
        if not text:
            return ""
        # Convert to lowercase and remove punctuation
        normalized = re.sub(r'[^\w\s]', '', text.lower())
        return normalized.strip()
    
    def derive_ad_group_theme(self, headline: str, description: str) -> str:
        """Step 2: Derive ad-group theme using LLM"""
        try:
            prompt = f"""Based on this headline and description, summarize the core ad-group theme in one phrase.

            Headline: {headline}
            Description: {description}
                
            Core ad-group theme:"""
            
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.llm_max_tokens,
                temperature=self.llm_temperature
            )
            
            theme = response.choices[0].message.content.strip()
            return theme
        except Exception as e:
            logger.error(f"Error deriving ad-group theme: {str(e)}")
            # Don't wrap in HTTPException here, let it propagate
            raise e
    
    def get_llm_relevance_score(self, keyword: str, category: str, prompt_type: str) -> int:
        """Get LLM relevance score for a specific prompt type"""
        try:
            prompts = {
                'brand': f"Rate on a scale of 0-10 how well '{keyword}' aligns with the brand category '{category}'. Only respond with an integer.",
                'product': f"Rate on a scale of 0-10 how well '{keyword}' aligns with the product category '{category}'. Only respond with an integer.",
                'theme': f"Rate on a scale of 0-10 how well '{keyword}' fits the ad-group theme '{category}'. Only respond with an integer.",
                'competition': f"Rate on a scale of 0-10 how competitive '{keyword}' is within the market segment '{category}'. Consider factors like search volume, bid competition, and market saturation. Only respond with an integer."
            }
            
            print(f"    🤖 Getting {prompt_type} score for '{keyword}' with category '{category}'")
            
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompts[prompt_type]}],
                max_tokens=10,
                temperature=0.1
            )
            
            score_text = response.choices[0].message.content.strip()
            # Extract integer from response
            score = int(re.findall(r'\d+', score_text)[0])
            final_score = max(0, min(10, score))  # Ensure score is between 0-10
            
            print(f"    📊 {prompt_type.capitalize()} score: {final_score}/10")
            return final_score
            
        except Exception as e:
            logger.error(f"Error getting {prompt_type} relevance score for '{keyword}': {str(e)}")
            # Don't wrap in HTTPException here, let it propagate
            raise e
    
    def classify_keyword_intent(self, keyword: str, brand_category: str, product_category: str) -> str:
        """Classify keyword intent as transactional, navigational, or informational"""
        try:
            prompt = f"""Classify the search intent for the keyword '{keyword}' in the context of brand '{brand_category}' and product '{product_category}'.

Choose one of these three intent types:
- TRANSACTIONAL: User wants to buy/purchase (e.g., "buy running shoes", "running shoes price", "order athletic shoes")
- NAVIGATIONAL: User wants to find a specific brand/website (e.g., "Nike running shoes", "Adidas website", "Puma store")
- INFORMATIONAL: User wants to learn/research (e.g., "best running shoes", "running shoes reviews", "how to choose running shoes")

Respond with only the intent type: TRANSACTIONAL, NAVIGATIONAL, or INFORMATIONAL"""
            
            print(f"    🎯 Classifying intent for '{keyword}':")
            
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=15,
                temperature=0.1
            )
            
            intent = response.choices[0].message.content.strip().upper()
            
            # Validate intent type
            valid_intents = ['TRANSACTIONAL', 'NAVIGATIONAL', 'INFORMATIONAL']
            if intent not in valid_intents:
                intent = 'INFORMATIONAL'  # Default fallback
            
            print(f"      📊 Intent: {intent}")
            return intent
            
        except Exception as e:
            logger.error(f"Error classifying keyword intent for '{keyword}': {str(e)}")
            return "INFORMATIONAL"  # Default fallback
    
    def calculate_relevance_scores(self, keyword: str, brand_category: str, 
                                 product_category: str, ad_group_theme: str) -> RelevanceScore:
        """Step 3 & 4: Calculate all LLM-driven relevance scores including intent classification"""
        print(f"    🎯 Calculating relevance scores for '{keyword}':")
        
        brand_alignment = self.get_llm_relevance_score(keyword, brand_category, 'brand')
        product_alignment = self.get_llm_relevance_score(keyword, product_category, 'product')
        theme_alignment = self.get_llm_relevance_score(keyword, ad_group_theme, 'theme')
        
        # Use brand_category and product_category to provide market segment context
        market_segment = f"{brand_category} - {product_category}"
        competition_score = self.get_llm_relevance_score(keyword, market_segment, 'competition')
        
        # Classify keyword intent
        intent_classification = self.classify_keyword_intent(keyword, brand_category, product_category)
        
        print(f"    📋 Relevance Summary for '{keyword}':")
        print(f"      - Brand Alignment: {brand_alignment}/10")
        print(f"      - Product Alignment: {product_alignment}/10")
        print(f"      - Theme Alignment: {theme_alignment}/10")
        print(f"      - Competition Score: {competition_score}/10")
        print(f"      - Intent: {intent_classification}")
        
        return RelevanceScore(
            brand_alignment=brand_alignment,
            product_alignment=product_alignment,
            theme_alignment=theme_alignment,
            competition_score=competition_score,
            intent_classification=intent_classification,
        )
    
    def normalize_performance_metrics(self, keywords: List[KeywordData]) -> Dict[str, List[float]]:
        """Step 5: Scale CPC, CTR, and Conversion Rate to 0-10 range using dynamic bounds from payload data"""
        ctr_values = [k.ctr for k in keywords]
        cpc_values = [k.cpc for k in keywords]
        conversion_values = [k.conversion_rate for k in keywords]
        
        print(f"📊 Original CTR values: {ctr_values}")
        print(f"📊 Original CPC values: {cpc_values}")
        print(f"📊 Original Conversion Rate values: {conversion_values}")
        
        # Calculate dynamic bounds from payload data
        ctr_min = min(ctr_values) if ctr_values else 0.0
        ctr_max = max(ctr_values) if ctr_values else 1.0
        cpc_min = min(cpc_values) if cpc_values else 0.0
        cpc_max = max(cpc_values) if cpc_values else 1.0
        conversion_min = min(conversion_values) if conversion_values else 0.0
        conversion_max = max(conversion_values) if conversion_values else 1.0
        
        print(f"📊 Dynamic CTR bounds: {ctr_min} - {ctr_max}")
        print(f"📊 Dynamic CPC bounds: {cpc_min} - {cpc_max}")
        print(f"📊 Dynamic Conversion Rate bounds: {conversion_min} - {conversion_max}")
        
        # Scale CTR: higher CTR = higher score (good)
        normalized_ctr = []
        for ctr in ctr_values:
            if ctr_max == ctr_min:  # All values are the same
                score = 5.0  # Middle score
            else:
                score = ((ctr - ctr_min) / (ctr_max - ctr_min)) * 10
            normalized_ctr.append(score)
        
        # Scale CPC: lower CPC = higher score (good)
        normalized_cpc = []
        for cpc in cpc_values:
            if cpc_max == cpc_min:  # All values are the same
                score = 5.0  # Middle score
            else:
                score = ((cpc_max - cpc) / (cpc_max - cpc_min)) * 10
            normalized_cpc.append(score)
        
        # Scale Conversion Rate: higher conversion rate = higher score (good)
        normalized_conversion = []
        for conversion in conversion_values:
            if conversion_max == conversion_min:  # All values are the same
                score = 5.0  # Middle score
            else:
                score = ((conversion - conversion_min) / (conversion_max - conversion_min)) * 10
            normalized_conversion.append(score)
        
        print(f"📊 Scaled CTR (0-10): {[round(x, 3) for x in normalized_ctr]}")
        print(f"📊 Scaled CPC (0-10): {[round(x, 3) for x in normalized_cpc]}")
        print(f"📊 Scaled Conversion Rate (0-10): {[round(x, 3) for x in normalized_conversion]}")
        
        return {
            'normalized_ctr': normalized_ctr,
            'normalized_cpc': normalized_cpc,
            'normalized_conversion': normalized_conversion
        }
    
    def calculate_composite_score(self, relevance_scores: RelevanceScore, 
                                normalized_ctr: float, normalized_cpc: float, normalized_conversion: float,
                                weights: Dict[str, float]) -> float:
        """Step 6: Calculate composite score using weighted formula including conversion rate"""
        print(f"    🧮 Calculating composite score:")
        print(f"      - Normalized CTR: {normalized_ctr:.3f}/10")
        print(f"      - Normalized CPC: {normalized_cpc:.3f}/10")
        print(f"      - Normalized Conversion Rate: {normalized_conversion:.3f}/10")
        
        brand_weighted = weights['brand_alignment'] * relevance_scores.brand_alignment
        product_weighted = weights['product_alignment'] * relevance_scores.product_alignment
        theme_weighted = weights['theme_alignment'] * relevance_scores.theme_alignment
        competition_weighted = weights['competition'] * relevance_scores.competition_score
        ctr_weighted = weights['ctr'] * normalized_ctr
        cpc_weighted = weights['cpc'] * normalized_cpc
        conversion_weighted = weights['conversion'] * normalized_conversion
        
        print(f"      📊 Weighted Components:")
        print(f"        - Brand Alignment: {relevance_scores.brand_alignment} × {weights['brand_alignment']} = {brand_weighted:.3f}")
        print(f"        - Product Alignment: {relevance_scores.product_alignment} × {weights['product_alignment']} = {product_weighted:.3f}")
        print(f"        - Theme Alignment: {relevance_scores.theme_alignment} × {weights['theme_alignment']} = {theme_weighted:.3f}")
        print(f"        - Competition: -{relevance_scores.competition_score} × {weights['competition']} = -{competition_weighted:.3f}")
        print(f"        - CTR: {normalized_ctr:.3f} × {weights['ctr']} = {ctr_weighted:.3f}")
        print(f"        - CPC: {normalized_cpc:.3f} × {weights['cpc']} = {cpc_weighted:.3f}")
        print(f"        - Conversion Rate: {normalized_conversion:.3f} × {weights['conversion']} = {conversion_weighted:.3f}")
        
        final_score = (
            brand_weighted +
            product_weighted +
            theme_weighted -
            competition_weighted +
            ctr_weighted +
            cpc_weighted +
            conversion_weighted
        )
        
        print(f"      🎯 Final Score: {final_score:.3f}")
        return round(final_score, 3)
    
    def identify_negative_keywords(self, keyword_scores: List[KeywordScore]) -> List[str]:
        """Step 7: Identify negative keywords based on fixed threshold"""
        if not keyword_scores:
            return []
        
        scores = [ks.final_score for ks in keyword_scores]
        keywords = [ks.keyword for ks in keyword_scores]
        
        print(f"    📊 All keyword scores: {list(zip(keywords, scores))}")
        
        # Fixed threshold from environment variable
        threshold = self.negative_threshold
        print(f"    🎯 Fixed threshold: {threshold:.3f}")
        
        negative_keywords = []
        for ks in keyword_scores:
            if ks.final_score < threshold:
                ks.is_negative = True
                negative_keywords.append(ks.keyword)
                print(f"    ❌ '{ks.keyword}' marked as negative (score: {ks.final_score:.3f} < threshold: {threshold:.3f})")
            else:
                print(f"    ✅ '{ks.keyword}' retained (score: {ks.final_score:.3f} >= threshold: {threshold:.3f})")
        
        return negative_keywords
    
    def calculate_replacement_score(self, relevance_scores: RelevanceScore, weights: Dict[str, float]) -> float:
        """Calculate score for replacement keywords using only relevance scores (no performance metrics)"""
        print(f"    🧮 Calculating replacement score (relevance only):")
        
        brand_weighted = weights['brand_alignment'] * relevance_scores.brand_alignment
        product_weighted = weights['product_alignment'] * relevance_scores.product_alignment
        theme_weighted = weights['theme_alignment'] * relevance_scores.theme_alignment
        competition_weighted = weights['competition'] * relevance_scores.competition_score
        
        print(f"      📊 Weighted Components:")
        print(f"        - Brand Alignment: {relevance_scores.brand_alignment} × {weights['brand_alignment']} = {brand_weighted:.3f}")
        print(f"        - Product Alignment: {relevance_scores.product_alignment} × {weights['product_alignment']} = {product_weighted:.3f}")
        print(f"        - Theme Alignment: {relevance_scores.theme_alignment} × {weights['theme_alignment']} = {theme_weighted:.3f}")
        print(f"        - Competition: -{relevance_scores.competition_score} × {weights['competition']} = -{competition_weighted:.3f}")
        
        final_score = (
            brand_weighted +
            product_weighted +
            theme_weighted -
            competition_weighted
        )
        
        print(f"      🎯 Replacement Score: {final_score:.3f}")
        return round(final_score, 3)
    
    def generate_replacement_keywords(self, brand_category: str, product_category: str, 
                                    ad_group_theme: str, negative_keywords: List[str], 
                                    keyword_scores: List[KeywordScore], original_keywords: List[KeywordData], 
                                    replacement_count: int = None) -> List[Dict[str, Any]]:
        """Step 8: Generate multiple replacement alternatives for each negative keyword"""
        try:
            # Use provided count or fall back to default
            count = replacement_count if replacement_count is not None else self.default_replacement_count
            
            replacement_keywords = []
            
            print(f"    📊 Generating {count} replacement alternatives per negative keyword (relevance-based scoring only)")
            
            # Separate retained and negative keyword scores
            retained_scores = [ks for ks in keyword_scores if not ks.is_negative]
            negative_scores = [ks for ks in keyword_scores if ks.is_negative]
            
            # Predict performance metrics for replacement keywords
            predicted_metrics = self.predict_performance_metrics(retained_scores, negative_scores, original_keywords)
            
            for negative_keyword in negative_keywords:
                print(f"    🔄 Generating {count} alternatives for '{negative_keyword}':")
                
                prompt = f"""Given that '{negative_keyword}' was identified as a negative keyword for brand '{brand_category}', product '{product_category}', and theme '{ad_group_theme}', suggest exactly {count} better alternative keywords that would be highly relevant and effective for advertising.

The replacements should be more specific, relevant, and have better potential performance than '{negative_keyword}'.

Respond with exactly {count} keywords, separated by newlines, with no additional text:"""
                
                response = client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.llm_max_tokens * count,  # Increase token limit for multiple keywords
                    temperature=0.7
                )
                
                # Parse multiple keywords from response
                response_text = response.choices[0].message.content.strip()
                suggested_keywords = [kw.strip() for kw in response_text.split('\n') if kw.strip()]
                
                # Ensure we got the expected number of keywords
                if len(suggested_keywords) < count:
                    print(f"      ⚠️ LLM returned {len(suggested_keywords)} keywords, expected {count}")
                    # Pad with the last keyword if needed
                    while len(suggested_keywords) < count:
                        suggested_keywords.append(suggested_keywords[-1] if suggested_keywords else negative_keyword)
                
                print(f"      🤖 LLM suggested: {suggested_keywords}")
                
                # Process each alternative
                alternatives = []
                for i, replacement_keyword in enumerate(suggested_keywords[:count]):  # Limit to requested count
                    # Step 3 & 4: Calculate relevance scores
                    relevance_scores = self.calculate_relevance_scores(
                        replacement_keyword, brand_category, product_category, ad_group_theme
                    )
                    
                    # Use default weights for replacement keywords
                    composite_score = self.calculate_replacement_score(
                        relevance_scores, self.default_weights
                    )
                    
                    # Use LLM to determine match type for replacement keyword
                    match_type_prompt = f"""Given the keyword '{replacement_keyword}', classify its match type as one of the following: EXACT, PHRASE, or BROAD. Respond with only the match type."""
                    match_type_response = client.chat.completions.create(
                        model=self.llm_model,
                        messages=[{"role": "user", "content": match_type_prompt}],
                        max_tokens=5,
                        temperature=0.1
                    )
                    match_type = match_type_response.choices[0].message.content.strip().upper()
                    if match_type not in ["EXACT", "PHRASE", "BROAD"]:
                        match_type = "EXACT"  # Default fallback
                    
                    alternatives.append({
                        'keyword': replacement_keyword,
                        'composite_score': composite_score,
                        'rank': i + 1,
                        'match_type': match_type,
                        'relevance_scores': relevance_scores.dict(),
                        'predicted_performance': predicted_metrics
                    })
                    
                    print(f"      ✅ Alternative {i+1}: '{replacement_keyword}' (score: {composite_score:.3f})")
                
                # Sort alternatives by score (highest first)
                alternatives.sort(key=lambda x: x['composite_score'], reverse=True)
                
                # Update ranks after sorting
                for i, alt in enumerate(alternatives):
                    alt['rank'] = i + 1
                
                # Add grouped alternatives to response
                replacement_keywords.append({
                    'original_negative': negative_keyword,
                    'alternatives': alternatives
                })
            
            return replacement_keywords
            
        except Exception as e:
            logger.error(f"Error generating replacement keywords: {str(e)}")
            # Don't wrap in HTTPException here, let it propagate
            raise e

    def predict_performance_metrics(self, retained_keywords: List[KeywordScore], negative_keywords: List[KeywordScore], 
                                  original_keywords: List[KeywordData]) -> Dict[str, float]:
        """Predict performance metrics for replacement keywords based on retained vs negative keyword differences"""
        if not retained_keywords and not negative_keywords:
            return {'predicted_ctr': 0.04, 'predicted_cpc': 2.0, 'predicted_conversion_rate': 0.06}
        
        # Get original values for retained and negative keywords
        retained_original = []
        negative_original = []
        
        for ks in retained_keywords + negative_keywords:
            for original in original_keywords:
                if original.keyword == ks.keyword:
                    if ks.is_negative:
                        negative_original.append(original)
                    else:
                        retained_original.append(original)
                    break
        
        # Calculate averages in original scale
        retained_ctr = np.mean([kw.ctr for kw in retained_original]) if retained_original else 0.04
        retained_cpc = np.mean([kw.cpc for kw in retained_original]) if retained_original else 2.5
        retained_conversion = np.mean([kw.conversion_rate for kw in retained_original]) if retained_original else 0.06
        
        negative_ctr = np.mean([kw.ctr for kw in negative_original]) if negative_original else 0.03
        negative_cpc = np.mean([kw.cpc for kw in negative_original]) if negative_original else 3.0
        negative_conversion = np.mean([kw.conversion_rate for kw in negative_original]) if negative_original else 0.04
        
        # Predict based on improvement over negative keywords in original scale
        # Target: Better than negative keywords, approaching retained keyword performance
        predicted_ctr = negative_ctr + (retained_ctr - negative_ctr) * 0.7  # 70% improvement
        predicted_cpc = negative_cpc + (retained_cpc - negative_cpc) * 0.7   # 70% improvement
        predicted_conversion = negative_conversion + (retained_conversion - negative_conversion) * 0.7  # 70% improvement
        
        print(f"    📊 Performance Prediction (Original Scale):")
        print(f"      - Retained avg: CTR={retained_ctr:.4f}, CPC={retained_cpc:.2f}, Conv={retained_conversion:.4f}")
        print(f"      - Negative avg: CTR={negative_ctr:.4f}, CPC={negative_cpc:.2f}, Conv={negative_conversion:.4f}")
        print(f"      - Predicted: CTR={predicted_ctr:.4f}, CPC={predicted_cpc:.2f}, Conv={predicted_conversion:.4f}")
        
        return {
            'predicted_ctr': round(predicted_ctr, 4),
            'predicted_cpc': round(predicted_cpc, 2),
            'predicted_conversion_rate': round(predicted_conversion, 4)
        }

@router.post(
    "/v1/keyword-optimization",
    response_model=KeywordOptimizationResponse,
    summary="Optimize Keywords Using LLM-Driven Analysis",
    description="Analyze and optimize keywords using LLM-driven relevance scoring, performance metrics, and intelligent replacement suggestions",
    tags=["Keyword Optimization"]
)
async def optimize_keywords(input_data: KeywordOptimizationInput):
    """
    Complete keyword optimization flow including:
    1. Data preprocessing and normalization
    2. Ad-group theme derivation
    3. LLM-driven relevance scoring
    4. Performance metric normalization
    5. Composite score calculation
    6. Negative keyword identification
    7. Replacement keyword generation
    """
    try:
        print(f"🚀 Starting keyword optimization for {len(input_data.keywords)} keywords")
        logger.info(f"Starting keyword optimization for {len(input_data.keywords)} keywords")
        
        service = KeywordOptimizationService()
        
        # Step 1: Data preprocessing
        print("📝 Step 1: Data preprocessing...")
        for keyword_data in input_data.keywords:
            keyword_data.keyword = service.normalize_text(keyword_data.keyword)
        
        # Normalize headlines and descriptions
        normalized_headlines = [service.normalize_text(headline) for headline in input_data.headlines]
        normalized_descriptions = [service.normalize_text(description) for description in input_data.descriptions]
        
        # Step 2: Derive ad-group theme (use all headlines and descriptions)
        print("🎯 Step 2: Deriving ad-group theme...")
        if input_data.headlines and input_data.descriptions:
            # Combine all headlines and descriptions
            all_headlines = " | ".join(normalized_headlines)
            all_descriptions = " | ".join(normalized_descriptions)
            ad_group_theme = service.derive_ad_group_theme(all_headlines, all_descriptions)
            print(f"✅ Ad-group theme: {ad_group_theme}")
        else:
            ad_group_theme = ""
            print("⚠️ No headlines/descriptions provided, using empty theme")
        
        # Step 3: Normalize performance metrics
        print("📊 Step 3: Normalizing performance metrics...")
        normalized_metrics = service.normalize_performance_metrics(input_data.keywords)
        
        # Use default weights
        weights = service.default_weights
        
        # Process each keyword
        print("🔍 Step 4: Processing keywords with LLM scoring...")
        keyword_scores = []
        for i, keyword_data in enumerate(input_data.keywords):
            print(f"  Processing keyword {i+1}/{len(input_data.keywords)}: '{keyword_data.keyword}'")
            
            # Step 3 & 4: Calculate relevance scores
            relevance_scores = service.calculate_relevance_scores(
                keyword_data.keyword,
                input_data.brand_category,
                input_data.product_category,
                ad_group_theme
            )
            
            # Step 6: Calculate composite score
            final_score = service.calculate_composite_score(
                relevance_scores,
                normalized_metrics['normalized_ctr'][i],
                normalized_metrics['normalized_cpc'][i],
                normalized_metrics['normalized_conversion'][i],
                weights
            )
            
            keyword_score = KeywordScore(
                keyword=keyword_data.keyword,
                relevance_scores=relevance_scores,
                normalized_ctr=normalized_metrics['normalized_ctr'][i],
                normalized_cpc=normalized_metrics['normalized_cpc'][i],
                normalized_conversion_rate=normalized_metrics['normalized_conversion'][i],
                final_score=final_score,
                is_negative=False
            )
            keyword_scores.append(keyword_score)
            print(f"    Score: {final_score:.3f}")
        
        # Step 5: Identify negative keywords
        print("❌ Step 5: Identifying negative keywords...")
        negative_keywords = service.identify_negative_keywords(keyword_scores)
        print(f"✅ Found {len(negative_keywords)} negative keywords: {negative_keywords}")
        
        # Format negative keywords with scores
        negative_keywords_with_scores = []
        for ks in keyword_scores:
            if ks.is_negative:
                # Find original keyword data
                original_data = None
                for kw in input_data.keywords:
                    if kw.keyword == ks.keyword:
                        original_data = kw
                        break
                
                negative_keywords_with_scores.append({
                    'keyword': ks.keyword,
                    'final_score': ks.final_score,
                    'relevance_scores': {
                        'brand_alignment': ks.relevance_scores.brand_alignment,
                        'product_alignment': ks.relevance_scores.product_alignment,
                        'theme_alignment': ks.relevance_scores.theme_alignment,
                        'competition_score': ks.relevance_scores.competition_score
                    },
                    'intent_classification': ks.relevance_scores.intent_classification,
                    'match_type': original_data.match_type if original_data else "BROAD",
                    'performance_metrics': {
                        'normalized_ctr': round(ks.normalized_ctr, 3),
                        'normalized_cpc': round(ks.normalized_cpc, 3),
                        'normalized_conversion_rate': round(ks.normalized_conversion_rate, 3),
                        'original_ctr': original_data.ctr if original_data else 0.0,
                        'original_cpc': original_data.cpc if original_data else 0.0,
                        'original_conversion_rate': original_data.conversion_rate if original_data else 0.0
                    }
                })
        
        # Separate retained keywords (non-negative)
        retained_keywords = []
        for ks in keyword_scores:
            if not ks.is_negative:
                # Find original keyword data
                original_data = None
                for kw in input_data.keywords:
                    if kw.keyword == ks.keyword:
                        original_data = kw
                        break
                
                retained_keywords.append({
                    'keyword': ks.keyword,
                    'final_score': ks.final_score,
                    'relevance_scores': {
                        'brand_alignment': ks.relevance_scores.brand_alignment,
                        'product_alignment': ks.relevance_scores.product_alignment,
                        'theme_alignment': ks.relevance_scores.theme_alignment,
                        'competition_score': ks.relevance_scores.competition_score
                    },
                    'intent_classification': ks.relevance_scores.intent_classification,
                    'match_type': original_data.match_type if original_data else "BROAD",
                    'performance_metrics': {
                        'normalized_ctr': round(ks.normalized_ctr, 3),
                        'normalized_cpc': round(ks.normalized_cpc, 3),
                        'normalized_conversion_rate': round(ks.normalized_conversion_rate, 3),
                        'original_ctr': original_data.ctr if original_data else 0.0,
                        'original_cpc': original_data.cpc if original_data else 0.0,
                        'original_conversion_rate': original_data.conversion_rate if original_data else 0.0
                    }
                })
        
        print(f"✅ Retained {len(retained_keywords)} keywords")
        
        # Step 8: Generate replacement keywords
        print("🔄 Step 6: Generating replacement keywords...")
        if negative_keywords:
  
            replacement_keywords = service.generate_replacement_keywords(
                input_data.brand_category,
                input_data.product_category,
                ad_group_theme,
                negative_keywords,
                keyword_scores,
                input_data.keywords,
                input_data.replacement_count
            )
            total_alternatives = sum(len(group['alternatives']) for group in replacement_keywords)
            print(f"✅ Generated {len(replacement_keywords)} replacement groups with {total_alternatives} total alternatives")
        else:
            replacement_keywords = []
            if not negative_keywords:
                print("✅ No negative keywords found, no replacements needed")
            else:
                print("⚠️ No headlines/descriptions provided, skipping replacement keywords")
        
        # Prepare optimization summary
        optimization_summary = {
            "total_keywords": len(input_data.keywords),
            "negative_keywords_count": len(negative_keywords_with_scores),
            "retained_keywords_count": len(retained_keywords),
            "replacement_groups_count": len(replacement_keywords),
            "replacement_keywords_count": sum(len(group['alternatives']) for group in replacement_keywords) if replacement_keywords else 0,
            "average_final_score": round(np.mean([ks.final_score for ks in keyword_scores]) if keyword_scores else 0, 3),
            "score_std_dev": round(np.std([ks.final_score for ks in keyword_scores]) if keyword_scores else 0, 3),
            "negative_keywords_list": [kw['keyword'] for kw in negative_keywords_with_scores],
            "retained_keywords_list": [kw['keyword'] for kw in retained_keywords],
            "replacement_keywords_list": [alt['keyword'] for group in replacement_keywords for alt in group['alternatives']] if replacement_keywords else []
        }
        
        print("🎉 Keyword optimization completed successfully!")
        print(f"📈 Summary: {optimization_summary}")
        print(f"❌ Negative keywords: {[kw['keyword'] for kw in negative_keywords_with_scores]}")
        print(f"✅ Retained keywords: {[kw['keyword'] for kw in retained_keywords]}")
        replacement_keywords_list = [alt['keyword'] for group in replacement_keywords for alt in group['alternatives']] if replacement_keywords else []
        print(f"🔄 Replacement keywords: {replacement_keywords_list}")
        
        return KeywordOptimizationResponse(
            ad_group_theme=ad_group_theme,
            negative_keywords=negative_keywords_with_scores,
            replacement_keywords=replacement_keywords,
            retained_keywords=retained_keywords,
            optimization_summary=optimization_summary
        )
        
    except Exception as e:
        print(f"💥 Error in keyword optimization: {str(e)}")
        logger.error(f"Keyword optimization failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Keyword optimization failed: {str(e)}")

