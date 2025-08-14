import struct
from itertools import chain, repeat
import urllib
import pandas as pd
from azure.identity import ClientSecretCredential
import pyodbc
from sqlalchemy import create_engine
import os
import logging
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Data Models
class KeywordViewData(BaseModel):
    keyword: str
    cpc: float
    ctr: float
    conversion_rate: float
    match_type: Optional[str] = None

class AdGroupFetchRequest(BaseModel):
    ad_group_id: int



router = APIRouter()

class FabricDataFetcher:
    def __init__(self):
        # Service principal credentials from environment variables
        self.tenant_id = os.getenv("AZURE_TENANT_ID")
        self.client_id = os.getenv("AZURE_CLIENT_ID")
        self.client_secret = os.getenv("AZURE_CLIENT_SECRET")
        self.resource_url = os.getenv("AZURE_RESOURCE_URL", "https://database.windows.net/.default")
        
        # Connection details from environment variables
        self.sql_endpoint = os.getenv("AZURE_SQL_ENDPOINT")
        self.database = os.getenv("AZURE_DATABASE_NAME", "Gold_WH")
        
        # Validate required environment variables
        if not all([self.tenant_id, self.client_id, self.client_secret, self.sql_endpoint]):
            raise ValueError("Missing required Azure/Fabric environment variables")
        
        logger.info(f"FabricDataFetcher initialized with endpoint: {self.sql_endpoint}, database: {self.database}")
    
    def create_connection(self):
        """Create connection to Fabric using service principal authentication"""
        try:
            logger.info("Creating Azure credential...")
            # Create credential using service principal
            credential = ClientSecretCredential(
                tenant_id=self.tenant_id,
                client_id=self.client_id,
                client_secret=self.client_secret
            )
            token_object = credential.get_token(self.resource_url)
            logger.info("Successfully obtained Azure token")
            
            # Create connection string
            connection_string = (
                f"Driver={{ODBC Driver 18 for SQL Server}};"
                f"Server={self.sql_endpoint},1433;"
                f"Database={self.database};"
                f"Encrypt=Yes;"
                f"TrustServerCertificate=No"
            )
            params = urllib.parse.quote(connection_string)
            logger.info(f"Connecting to: {self.sql_endpoint}, Database: {self.database}")
            
            # Prepare access token for ODBC
            token_as_bytes = bytes(token_object.token, "UTF-8")
            encoded_bytes = bytes(chain.from_iterable(zip(token_as_bytes, repeat(0))))
            token_bytes = struct.pack("<i", len(encoded_bytes)) + encoded_bytes
            attrs_before = {1256: token_bytes}  # SQL_COPT_SS_ACCESS_TOKEN
            
            # Create connection with token
            conn = pyodbc.connect(
                connection_string,
                attrs_before={1256: token_bytes}  # SQL_COPT_SS_ACCESS_TOKEN
            )
            logger.info("Successfully connected to Fabric database")
            
            # Create SQLAlchemy engine
            engine = create_engine(
                "mssql+pyodbc:///?odbc_connect={0}".format(params),
                connect_args={'attrs_before': attrs_before}
            )
            
            return conn, engine
            
        except Exception as e:
            logger.error(f"Failed to create Fabric connection: {str(e)}")
            raise Exception(f"Failed to create Fabric connection: {str(e)}")
    

    
    def fetch_keywords_by_ad_group(self, ad_group_id: int, schema: str = "Silver_LH", sub_schema: str = "google_ads", table: str = "keyword_view") -> List[KeywordViewData]:
        """Fetch keyword data for a specific ad group ID"""
        conn = None
        try:
            logger.info(f"Fetching keywords for ad_group_id: {ad_group_id}")
            conn, engine = self.create_connection()
            
            # Build SQL query to filter by ad_group_id - only fetch required columns with averages
            sql = f"""
            SELECT 
                ad_group_criterion_keyword_text,
                ad_group_criterion_keyword_match_type,
                AVG(CAST(metrics_cost_micros AS FLOAT)) as avg_cost_micros,
                SUM(metrics_clicks) as total_clicks,
                AVG(CAST(metrics_ctr AS FLOAT)) as avg_ctr,
                SUM(metrics_conversions) as total_conversions
            FROM [{schema}].[{sub_schema}].[{table}] 
            WHERE ad_group_id = {ad_group_id}
            AND ad_group_criterion_negative = 0
            GROUP BY ad_group_criterion_keyword_text, ad_group_criterion_keyword_match_type
            ORDER BY total_clicks DESC
            """
            
            logger.info(f"Executing SQL query on table: [{schema}].[{sub_schema}].[{table}]")
            logger.info(f"SQL Query: {sql}")
            
            # Execute query and fetch data
            df = pd.read_sql(sql, engine)
            logger.info(f"Query executed successfully. Retrieved {len(df)} rows")
            
            # Convert DataFrame to list of KeywordViewData objects
            keywords = []
            logger.info("Processing retrieved data...")
            for i, (_, row) in enumerate(df.iterrows()):
                # Calculate CPC from average cost_micros (convert from micros to dollars)
                avg_cost_micros = row.get('avg_cost_micros', 0)
                total_clicks = row.get('total_clicks', 0)
                cpc = (avg_cost_micros / 1000000) if total_clicks > 0 else 0.0
                
                # Use average CTR
                avg_ctr = row.get('avg_ctr', 0.0)
                
                # Calculate conversion rate from totals
                total_conversions = row.get('total_conversions', 0.0)
                conversion_rate = total_conversions / total_clicks if total_clicks > 0 else 0.0
                
                keyword = str(row.get('ad_group_criterion_keyword_text', ''))
                match_type = str(row.get('ad_group_criterion_keyword_match_type', 'BROAD'))
                
                keyword_data = KeywordViewData(
                    keyword=keyword,
                    cpc=cpc,
                    ctr=avg_ctr,
                    conversion_rate=conversion_rate,
                    match_type=match_type
                )
                keywords.append(keyword_data)
                
                if i < 3:  # Log first 3 keywords for debugging
                    logger.info(f"Processed keyword {i+1}: '{keyword}' (CPC: ${cpc:.2f}, CTR: {avg_ctr:.3f}, Conv: {conversion_rate:.3f}, Type: {match_type})")
            
            logger.info(f"Successfully processed {len(keywords)} keywords")
            return keywords
            
        except Exception as e:
            logger.error(f"Failed to fetch keyword data for ad group {ad_group_id}: {str(e)}")
            raise Exception(f"Failed to fetch keyword data for ad group {ad_group_id}: {str(e)}")
        finally:
            if conn:
                conn.close()
                logger.info("Database connection closed")

# Create global instance
try:
    data_fetcher = FabricDataFetcher()
    logger.info("FabricDataFetcher initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize FabricDataFetcher: {e}")
    data_fetcher = None

@router.post(
    "/v1/fetch-keywords-by-ad-group",
    response_model=List[KeywordViewData],
    summary="Fetch Keywords by Ad Group ID",
    description="Fetch keyword data for a specific ad group ID from Fabric table",
    tags=["Data Fetching"]
)
async def fetch_keywords_by_ad_group(request: AdGroupFetchRequest):
    """
    Fetch keyword data for a specific ad group ID from Fabric table
    """
    try:
        logger.info(f"Received request to fetch keywords for ad_group_id: {request.ad_group_id}")
        
        if not data_fetcher:
            logger.error("Fabric data fetcher not initialized")
            raise HTTPException(
                status_code=500, 
                detail="Fabric data fetcher not initialized. Check environment variables."
            )
        
        keywords = data_fetcher.fetch_keywords_by_ad_group(ad_group_id=request.ad_group_id)
        
        logger.info(f"Successfully returning {len(keywords)} keywords for ad_group_id: {request.ad_group_id}")
        return keywords
        
    except Exception as e:
        logger.error(f"Error in fetch_keywords_by_ad_group endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch keywords for ad group {request.ad_group_id}: {str(e)}")