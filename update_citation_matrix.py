import os
import re
import subprocess
import requests
from bs4 import BeautifulSoup

# ===========================
# CONFIGURATION
# ===========================
SCHOLAR_URL = "https://scholar.google.com/citations?user=p6fjrJIAAAAJ&hl=en"
SEMANTIC_SCHOLAR_ID = "YOUR_SEMANTIC_SCHOLAR_ID"
USE_SEMANTIC_SCHOLAR = False
HTML_FILE_PATH = "index.html"

# ===========================
# FETCH CITATION DATA
# ===========================
def fetch_citation_data_google(scholar_url):
    headers = {
        'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/91.0.4472.124 Safari/537.36')
    }
    response = requests.get(scholar_url, headers=headers)
    if response.status_code != 200:
        print(f"⚠️ Failed to fetch Google Scholar (HTTP {response.status_code})")
        return {"citation_count": "Error"}
    
    soup = BeautifulSoup(response.content, 'html.parser')
    citation_count_element = soup.find('td', class_='gsc_rsb_std')
    
    if citation_count_element:
        citation_text = citation_count_element.text.strip()
        return {"citation_count": citation_text if citation_text.isdigit() else "Error"}
    return {"citation_count": "Error"}

# ===========================
# UPDATE HTML FILE
# ===========================
def fix_and_update_citation_matrix(citation_data, file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False

    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Fix malformed HTML structure
    fixed_html = re.sub(
        r'<div\s+id="citation-matrix">.*?</div>',  # Match any content inside the div
        r'<div id="citation-matrix">Citations: <span id="citation_count">0</span></div>',
        html_content,
        flags=re.DOTALL
    )

    # Ensure the span exists even if the div wasn't found
    if '<span id="citation_count">' not in fixed_html:
        fixed_html = fixed_html.replace(
            '</body>',
            '<div id="citation-matrix">Citations: <span id="citation_count">0</span></div></body>',
            1
        )

    # Update citation count
    updated_html = re.sub(
        r'(<span id="citation_count">)(.*?)(</span>)',
        rf'\g<1>{citation_data["citation_count"]}\g<3>',
        fixed_html
    )

    if updated_html == html_content:
        print("🔹 No changes needed")
        return False

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(updated_html)
    
    print(f"✅ Updated citation to {citation_data['citation_count']}")
    return True

# ===========================
# MAIN SCRIPT
# ===========================
if __name__ == "__main__":
    repo_path = os.getenv('GITHUB_WORKSPACE', os.getcwd())
    html_path = os.path.join(repo_path, HTML_FILE_PATH)
    
    if USE_SEMANTIC_SCHOLAR:
        data = fetch_citation_data_semantic(SEMANTIC_SCHOLAR_ID)
    else:
        data = fetch_citation_data_google(SCHOLAR_URL)
    
    if fix_and_update_citation_matrix(data, html_path):
        subprocess.run(['git', 'config', '--global', 'user.name', 'github-actions[bot]'])
        subprocess.run(['git', 'config', '--global', 'user.email', 'github-actions[bot]@users.noreply.github.com'])
        subprocess.run(['git', 'add', html_path])
        subprocess.run(['git', 'commit', '-m', 'Update citation count'], check=False)
        subprocess.run(['git', 'push'], check=False)
