# FILE: create_dataset.py
#
# PURPOSE:
# A complete, standalone script to scrape UFC fighter data,
# clean and engineer features, and save the final datasets.
# This version uses the corrected 'page=all' scraping logic.

import pandas as pd
import requests
from bs4 import BeautifulSoup
import string
import time

# --- Configuration ---
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
BASE_URL = "http://ufcstats.com/statistics/fighters"
SNAPSHOT_DATE = pd.to_datetime('today').strftime('%Y-%m-%d')


# --- Scraping Functions ---

def get_all_fighter_urls():
    """
    Finds all fighter URLs by scraping the 'page=all' for each letter in the A-Z index.
    This is the corrected, more reliable method.
    """
    fighter_urls = set()
    
    for letter in string.ascii_lowercase:
        # Construct the URL to get all fighters for a given letter on a single page
        url = f"{BASE_URL}?char={letter}&page=all"
        print(f"Scraping all fighters for letter: '{letter.upper()}'")
        
        try:
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"  -> Could not fetch page for letter '{letter.upper()}': {e}")
            continue # Move to the next letter

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all fighter links on the page
        fighter_links = soup.select('tr.b-statistics__table-row td:first-child a')
        if not fighter_links:
            print(f"  -> No fighters found for letter '{letter.upper()}'.")
            continue
            
        for link in fighter_links:
            if link.has_attr('href'):
                fighter_urls.add(link['href'])
        
        # Politeness delay between each letter's page request
        time.sleep(1)

    print(f"\nDiscovered a total of {len(fighter_urls)} unique fighter URLs.")
    return list(fighter_urls)


# --- All other functions (scrape_fighter_data, engineer_fighter_features, etc.) remain unchanged ---

def scrape_fighter_data(fighter_url):
    """Scrapes a single fighter's page for their details and fight history."""
    print(f"  -> Scraping details for: {fighter_url.split('/')[-1]}")
    try:
        response = requests.get(fighter_url, headers=HEADERS)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"    -> Could not fetch {fighter_url}: {e}")
        return None, None
    soup = BeautifulSoup(response.text, 'html.parser')
    fighter_details = {}
    name_element = soup.select_one('span.b-content__title-highlight')
    fighter_details['Name'] = name_element.text.strip() if name_element else None
    record_element = soup.select_one('span.b-content__title-record')
    if record_element:
        fighter_details['Record'] = record_element.get_text(strip=True).replace('Record:', '').strip()
    else:
        fighter_details['Record'] = None
    detail_elements = soup.select('ul.b-list__box-list li.b-list__box-list-item')
    for item in detail_elements:
        text_content = item.get_text(separator=":", strip=True)
        if ":" in text_content:
            key, value = text_content.split(":", 1)
            fighter_details[key.strip()] = value.strip()
    fight_history = []
    history_table = soup.select_one('table.b-fight-details__table')
    if history_table:
        rows = history_table.select('tr.b-fight-details__table-row[onclick]')
        for row in rows:
            cols = row.select('td')
            if len(cols) > 1:
                fighters = cols[1].select('a')
                fighter_name = fighters[0].get_text(strip=True) if len(fighters) > 0 else None
                opponent_name = fighters[1].get_text(strip=True) if len(fighters) > 1 else None
                if not opponent_name: continue
                fight = {'Outcome': cols[0].get_text(strip=True),'Fighter': fighter_name,'Opponent': opponent_name,'Event': cols[6].get_text(strip=True),'Method': cols[7].get_text(strip=True),'Round': cols[8].get_text(strip=True),'Time': cols[9].get_text(strip=True),}
                fight_history.append(fight)
    return fighter_details, fight_history

def _height_to_inches(h):
    if pd.isna(h): return None
    try:
        feet, inches = h.replace('"', '').split("' ")
        return int(feet) * 12 + int(inches)
    except: return None

def _split_record(record):
    if pd.isna(record): return None, None, None
    try:
        parts = record.split('-')
        wins = int(parts[0])
        losses = int(parts[1])
        draws = int(parts[2].split(' ')[0]) if len(parts) > 2 else 0
        return wins, losses, draws
    except (ValueError, IndexError):
        return None, None, None

def engineer_fighter_features(raw_df):
    df = raw_df.copy()
    df.replace('--', pd.NA, inplace=True)
    df['Height (inches)'] = df['Height'].apply(_height_to_inches)
    df['Weight (lbs)'] = pd.to_numeric(df['Weight'].str.replace(' lbs.', '', regex=False), errors='coerce')
    df['Reach (in)'] = pd.to_numeric(df['Reach'].str.replace('"', '', regex=False), errors='coerce')
    percent_cols = ['Str. Acc.', 'Str. Def.', 'TD Acc.', 'TD Def.']
    for col in percent_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].str.replace('%', '', regex=False), errors='coerce') / 100.0
    per_min_cols = ['SLpM', 'SApM', 'TD Avg.', 'Sub. Avg.']
    for col in per_min_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    record_split = df['Record'].apply(_split_record)
    df[['Wins', 'Losses', 'Draws']] = pd.DataFrame(record_split.tolist(), index=df.index)
    df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
    df['Age'] = ((pd.to_datetime(SNAPSHOT_DATE) - df['DOB']).dt.days / 365.25).round(1)
    if 'STANCE' in df.columns:
        df['STANCE'].fillna('Unknown', inplace=True)
        stance_dummies = pd.get_dummies(df['STANCE'], prefix='STANCE', dtype=int)
        df = pd.concat([df, stance_dummies], axis=1)
    cols_to_drop = ['Height', 'Weight', 'Reach', 'Record', 'DOB', 'STANCE']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    return df

# --- Main Execution Block ---

def main():
    print(f"--- PROCESS STARTED ON: {SNAPSHOT_DATE} ---")
    print("\n--- STEP 1: Scraping All Fighter URLs from A-Z Index ---")
    fighter_urls = get_all_fighter_urls()
    print("\n--- STEP 2: Scraping Individual Fighter Pages ---")
    all_fighter_details, all_fight_histories = [], []
    for url in fighter_urls:
        details, history = scrape_fighter_data(url)
        if details:
            details['fighter_url'] = url
            all_fighter_details.append(details)
        if history:
            for fight in history:
                fight['fighter_url'] = url
            all_fight_histories.extend(history)
        time.sleep(1.5)
    print("\n--- STEP 3: Saving Raw Scraped Data ---")
    fighters_raw_df = pd.DataFrame(all_fighter_details)
    fights_raw_df = pd.DataFrame(all_fight_histories)
    fighters_raw_df.to_csv('ufc_fighters_raw.csv', index=False)
    fights_raw_df.to_csv('ufc_fights.csv', index=False)
    print("Raw data saved to 'ufc_fighters_raw.csv' and 'ufc_fights.csv'")
    print("\n--- STEP 4: Engineering Features ---")
    fighters_engineered_df = engineer_fighter_features(fighters_raw_df)
    print("\n--- STEP 5: Saving Final Engineered Data ---")
    fighters_engineered_df.to_csv('ufc_fighters_engineered.csv', index=False)
    print("Engineered fighter data saved to 'ufc_fighters_engineered.csv'")
    print("\n--- PROCESS COMPLETE ---")
    print(f"Final engineered dataset has {fighters_engineered_df.shape[0]} rows and {fighters_engineered_df.shape[1]} columns.")
    print("You are now ready to start your modeling notebook.")

if __name__ == "__main__":
    main()