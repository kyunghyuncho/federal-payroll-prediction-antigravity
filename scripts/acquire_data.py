import requests
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# USAJOBS API INSTRUCTIONS FOR STUDENTS
# ==============================================================================
# 1. Go to https://developer.usajobs.gov/ and request an API key.
# 2. Copy `.env.example` to `.env` and fill in your email and the API key.
# ==============================================================================

AUTHORIZATION_KEY = os.environ.get("USAJOBS_API_KEY", "")
USER_AGENT_EMAIL = os.environ.get("USAJOBS_EMAIL", "")

if not AUTHORIZATION_KEY or not USER_AGENT_EMAIL:
    print("Error: USAJOBS_API_KEY or USAJOBS_EMAIL not found in environment.")
    print("Please set up your .env file according to .env.example.")
    exit(1)

HOST = "data.usajobs.gov"
BASE_URL = "https://data.usajobs.gov/api/Search"

def fetch_usajobs_data(num_pages=500, results_per_page=100):
    headers = {
        "Host": HOST,
        "User-Agent": USER_AGENT_EMAIL,
        "Authorization-Key": AUTHORIZATION_KEY
    }

    all_jobs = []

    for page in range(1, num_pages + 1):
        print(f"Fetching page {page}...")
        params = {
            "Page": page,
            "ResultsPerPage": results_per_page,
            # You can add other filters here, e.g., "Keyword": "Data Scientist"
        }
        
        response = requests.get(BASE_URL, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching page {page}: {response.status_code}")
            print(response.text)
            break
            
        data = response.json()
        search_results = data.get("SearchResult", {}).get("SearchResultItems", [])
        
        if not search_results:
            print("No more results found.")
            break
            
        for item in search_results:
            job = item.get("MatchedObjectId")
            position = item.get("MatchedObjectDescriptor", {})
            
            # Extract Text Description
            major_duties = position.get("UserArea", {}).get("Details", {}).get("MajorDuties", [])
            job_summary = position.get("JobSummary", "")
            
            # MajorDuties is often a list of strings
            if isinstance(major_duties, list):
                major_duties_text = "\n".join(major_duties)
            else:
                major_duties_text = str(major_duties)
                
            description = major_duties_text if major_duties_text.strip() else job_summary
            
            # Extract Date
            publication_date = position.get("PublicationStartDate", "")
            
            # Extract Salary Bounds
            remuneration = position.get("PositionRemuneration", [])
            minimum_range = None
            maximum_range = None
            rate_interval = None
            
            if remuneration and len(remuneration) > 0:
                minimum_range = remuneration[0].get("MinimumRange")
                maximum_range = remuneration[0].get("MaximumRange")
                rate_interval = remuneration[0].get("RateIntervalCode") # e.g., "Per Year"
                
            extracted_data = {
                "JobId": job,
                "Description": description,
                "PublicationStartDate": publication_date,
                "MinimumRange": minimum_range,
                "MaximumRange": maximum_range,
                "RateIntervalCode": rate_interval
            }
            all_jobs.append(extracted_data)
            
        # Optional: Add a small delay to respect API rate limits
        time.sleep(0.5)
        
    print(f"Total jobs extracted: {len(all_jobs)}")
    
    # Save the raw data
    output_fpath = "raw_usajobs_data.json"
    with open(output_fpath, "w") as f:
        json.dump(all_jobs, f, indent=4)
        
    print(f"Saved to {output_fpath}")

if __name__ == "__main__":
    fetch_usajobs_data()
