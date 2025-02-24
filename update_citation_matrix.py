import os
import re
import subprocess
import requests
from bs4 import BeautifulSoup

# ===========================
# CONFIGURATION
# ===========================
SCHOLAR_URL = "https://scholar.google.com/citations?user=p6fjrJIAAAAJ&hl=en"   
SEMANTIC_SCHOLAR_ID = "YOUR_SEMANTIC_SCHOLAR_ID"  # Optional
USE_SEMANTIC_SCHOLAR = False  # Set to True for API-based data (recommended)
HTML_FILE_PATH = "index.html"

# ===========================
# FETCH CITATION DATA
# ===========================
def fetch_citation_data_google(scholar_url):
    """Fetches citation count from Google Scholar (prone to blocking)."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(scholar_url, headers=headers)

    if response.status_code != 200:
        print(f"⚠️ Failed to fetch Google Scholar data (HTTP {response.status_code})")
        return {"citation_count": "Error"}

    # Print raw HTML to debug structure changes
    print("🔍 Raw HTML Content (First 1000 characters):")
    print(response.text[:1000])

    soup = BeautifulSoup(response.content, 'html.parser')
    citation_data = {}

    try:
        # Find citation count
        citation_count_element = soup.find_all('td', class_='gsc_rsb_std')
        if citation_count_element:
            citation_count = citation_count_element[0].text.strip()
            citation_data['citation_count'] = citation_count
            print(f"✅ Extracted Citation Count: {citation_count}")
        else:
            print("⚠️ Could not find the citation count element!")
            citation_data['citation_count'] = "N/A"
    except Exception as e:
        print(f"❌ Error while parsing Google Scholar: {e}")
        citation_data['citation_count'] = "Error"

    return citation_data

def fetch_citation_data_semantic(semantic_scholar_id):
    """Fetches citation count using Semantic Scholar API (recommended)."""
    url = f"https://api.semanticscholar.org/v1/author/{semantic_scholar_id}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"⚠️ Failed to fetch Semantic Scholar data (HTTP {response.status_code})")
        return {"citation_count": "Error"}

    data = response.json()
    return {"citation_count": str(data.get("citedByCount", 0))}

# ===========================
# UPDATE HTML FILE
# ===========================
def update_citation_matrix(citation_data, file_path):
    """Updates citation count in index.html if changed."""
    if not os.path.exists(file_path):
        print(f"❌ HTML file not found: {file_path}")
        return False

    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Extract existing citation count
    match = re.search(r'<span id="citation_count">(\d+)</span>', html_content)
    if match:
        old_citation_count = match.group(1)
        if old_citation_count == citation_data["citation_count"]:
            print(f"🔹 No update needed: Citation count is still {old_citation_count}")
            return False  # No change detected

    # Replace citation count in the HTML
    new_html_content = re.sub(
        r'(<span id="citation_count">)(\d+)(</span>)',
        rf'\1{citation_data["citation_count"]}\3',
        html_content
    )

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_html_content)

    print(f"✅ Updated citation count to {citation_data['citation_count']} in {file_path}")
    return True

# ===========================
# COMMIT & PUSH CHANGES
# ===========================
def commit_and_push_changes(repo_path, commit_message):
    """Commits and pushes changes to GitHub."""
    os.chdir(repo_path)
    subprocess.run(['git', 'add', 'index.html'])
    subprocess.run(['git', 'commit', '-m', commit_message])
    subprocess.run(['git', 'push'])

# ===========================
# MAIN SCRIPT
# ===========================
if __name__ == "__main__":
    repo_path = os.getenv('GITHUB_WORKSPACE', os.getcwd())  # GitHub Actions workspace
    html_file_path = os.path.join(repo_path, HTML_FILE_PATH)

    print(f"📂 Repository path: {repo_path}")
    print(f"📄 HTML file path: {html_file_path}")

    # Fetch citation count
    if USE_SEMANTIC_SCHOLAR:
        citation_data = fetch_citation_data_semantic(SEMANTIC_SCHOLAR_ID)
    else:
        citation_data = fetch_citation_data_google(SCHOLAR_URL)

    # Update the HTML file
    update_needed = update_citation_matrix(citation_data, html_file_path)

    # Commit & push only if changes were made
    if update_needed:
        commit_and_push_changes(repo_path, '🔄 Auto-update citation matrix')
    else:
        print("✅ No changes detected, skipping commit.")
