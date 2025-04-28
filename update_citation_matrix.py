import os
import subprocess
from scholarly import search_author_id
import requests

# ===========================
# CONFIGURATION
# ===========================
SCHOLAR_USER_ID = "p6fjrJIAAAAJ"  # Your Google Scholar user ID
SEMANTIC_SCHOLAR_ID = "YOUR_SEMANTIC_SCHOLAR_ID"
USE_SEMANTIC_SCHOLAR = False
HTML_FILE_PATH = "index.html"

# ===========================
# FETCH CITATION DATA
# ===========================
def fetch_citation_data_scholarly(user_id):
    try:
        author_iter = search_author_id(user_id)
        author = next(author_iter, None)
        if author and 'citedby' in author:
            return {"citation_count": str(author['citedby'])}
        else:
            print("⚠️ Could not find citation count with scholarly.")
            return {"citation_count": "Error"}
    except Exception as e:
        print(f"⚠️ Scholarly failed: {e}")
        return {"citation_count": "Error"}

def fetch_citation_data_semantic(semantic_scholar_id):
    url = f"https://api.semanticscholar.org/graph/v1/author/{semantic_scholar_id}?fields=citationCount"
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            data = resp.json()
            return {"citation_count": str(data.get("citationCount", "Error"))}
        else:
            print(f"⚠️ Failed to fetch Semantic Scholar (HTTP {resp.status_code})")
            return {"citation_count": "Error"}
    except Exception as e:
        print(f"⚠️ Semantic Scholar failed: {e}")
        return {"citation_count": "Error"}

# ===========================
# UPDATE HTML FILE
# ===========================
import re

def fix_and_update_citation_matrix(citation_data, file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False

    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Fix or add the citation matrix div
    fixed_html = re.sub(
        r'<div\s+id="citation-matrix">.*?</div>',
        r'<div id="citation-matrix">Citations: <span id="citation_count">0</span></div>',
        html_content,
        flags=re.DOTALL
    )

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
        data = fetch_citation_data_scholarly(SCHOLAR_USER_ID)
    
    if fix_and_update_citation_matrix(data, html_path):
        subprocess.run(['git', 'config', '--global', 'user.name', 'github-actions[bot]'])
        subprocess.run(['git', 'config', '--global', 'user.email', 'github-actions[bot]@users.noreply.github.com'])
        subprocess.run(['git', 'add', html_path])
        subprocess.run(['git', 'commit', '-m', 'Update citation count'], check=False)
        subprocess.run(['git', 'push'], check=False)
