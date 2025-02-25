import os
import re
import subprocess
import requests
from bs4 import BeautifulSoup

# ===========================
# CONFIGURATION
# ===========================
SCHOLAR_URL = "https://scholar.google.com/citations?user=p6fjrJIAAAAJ&hl=en"
# If you want to use Semantic Scholar API instead, set USE_SEMANTIC_SCHOLAR to True and update SEMANTIC_SCHOLAR_ID.
SEMANTIC_SCHOLAR_ID = "YOUR_SEMANTIC_SCHOLAR_ID"  # Optional
USE_SEMANTIC_SCHOLAR = False  # Set to True for API-based data (recommended)
HTML_FILE_PATH = "index.html"

# ===========================
# FETCH CITATION DATA
# ===========================
def fetch_citation_data_google(scholar_url):
    """Fetches citation count from Google Scholar (prone to blocking)."""
    headers = {
        'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/91.0.4472.124 Safari/537.36')
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
        # Find citation count element
        citation_count_element = soup.find_all('td', class_='gsc_rsb_std')
        # Replace this block in the fetch_citation_data_google function
        if citation_count_element:
            citation_count = citation_count_element[0].text.strip()
            # Handle cases where citation count is not a number
            if citation_count.isdigit():
                citation_data['citation_count'] = citation_count
            else:
                citation_data['citation_count'] = "Error"
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
def fix_and_update_citation_matrix(citation_data, file_path):
    """
    Fixes malformed HTML and updates the citation count in index.html.
    """
    if not os.path.exists(file_path):
        print(f"❌ HTML file not found: {file_path}")
        return False

    # Read the HTML content
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Debugging: Print HTML content
    print("🔍 Original HTML Content:")
    print(html_content[:1000])

    # Step 1: Fix malformed HTML (replace "Citations: K5</span>" with proper structure)
    fixed_html_content = re.sub(
        r'<div id="citation-matrix">\s*Citations:\s*K5</span>\s*</div>',
        r'<div id="citation-matrix">\n    Citations: <span id="citation_count">0</span>\n</div>',
        html_content,
        flags=re.DOTALL
    )

    # Check if the fix was applied
    if fixed_html_content != html_content:
        print("✅ Fixed malformed HTML structure.")
        html_content = fixed_html_content
    else:
        print("🔹 No malformed HTML detected.")

    # Step 2: Update the citation count in the corrected structure
    match = re.search(r'<span id="citation_count">(.*?)</span>', html_content)
    if match:
        old_citation_count = match.group(1).strip()
        if old_citation_count == citation_data["citation_count"]:
            print(f"🔹 No update needed: Citation count is still {old_citation_count}")
            return False  # No change detected
    else:
        print("⚠️ Could not find the citation count placeholder in HTML.")
        print("Please ensure your index.html contains exactly:")
        print('<span id="citation_count">0</span>')
        return False

    # Replace the citation count in the HTML
    new_html_content = re.sub(
        r'(<span id="citation_count">)(.*?)(</span>)',
        rf'\1{citation_data["citation_count"]}\3',
        html_content
    )

    # Write the updated HTML back to the file
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
    # Set Git user identity to avoid "Author identity unknown" errors.
    subprocess.run(['git', 'config', '--global', 'user.name', 'github-actions[bot]'])
    subprocess.run(['git', 'config', '--global', 'user.email', 'github-actions[bot]@users.noreply.github.com'])
    
    subprocess.run(['git', 'add', 'index.html'])
    commit_proc = subprocess.run(['git', 'commit', '-m', commit_message])
    if commit_proc.returncode == 0:
        subprocess.run(['git', 'push'])
    else:
        print("⚠️ No changes to commit.")

# ===========================
# MAIN SCRIPT
# ===========================
if __name__ == "__main__":
    repo_path = os.getenv('GITHUB_WORKSPACE', os.getcwd())  # GitHub Actions workspace or current directory
    html_file_path = os.path.join(repo_path, HTML_FILE_PATH)

    print(f"📂 Repository path: {repo_path}")
    print(f"📄 HTML file path: {html_file_path}")

    # Fetch citation count (choose method based on configuration)
    if USE_SEMANTIC_SCHOLAR:
        citation_data = fetch_citation_data_semantic(SEMANTIC_SCHOLAR_ID)
    else:
        citation_data = fetch_citation_data_google(SCHOLAR_URL)

    # Update the HTML file if needed
    update_needed = fix_and_update_citation_matrix(citation_data, html_file_path)
 

    # Commit & push only if changes were made
    if update_needed:
        commit_and_push_changes(repo_path, '🔄 Auto-update citation matrix')
    else:
        print("✅ No changes detected, skipping commit.")
