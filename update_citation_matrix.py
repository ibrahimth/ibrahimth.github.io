from scholarly import scholarly
import os
import subprocess
import re

# Function to fetch citation count from Google Scholar
def fetch_citation_data(scholar_profile_id):
    try:
        author = scholarly.search_author_id(scholar_profile_id)
        author = scholarly.fill(author, sections=["basics", "indices"])
        citation_count = author.get("citedby", 0)  # Get citation count, default to 0
        return {"citation_count": citation_count}
    except Exception as e:
        print(f"Error fetching citation data: {e}")
        return {"citation_count": 0}  # Return 0 if fetching fails

# Function to update citation count in HTML file
def update_citation_matrix(citation_data, file_path):
    if not os.path.exists(file_path):
        print(f"❌ HTML file not found: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Use regex to replace only the citation count inside the <span> tag
    new_html_content = re.sub(
        r'(<span id="citation_count">)(\d+)(</span>)',
        rf'\1{citation_data["citation_count"]}\3',
        html_content
    )

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_html_content)

    print(f"✅ Updated citation count to {citation_data['citation_count']} in {file_path}")

# Function to commit & push changes to GitHub
def commit_and_push_changes(repo_path, commit_message):
    os.chdir(repo_path)
    subprocess.run(['git', 'add', 'index.html'])
    subprocess.run(['git', 'commit', '-m', commit_message])
    subprocess.run(['git', 'push'])

if __name__ == "__main__":
    scholar_id = "p6fjrJIAAAAJ"  # Replace with your actual Google Scholar ID
    repo_path = os.getenv('GITHUB_WORKSPACE', os.getcwd())  # GitHub Actions repo path
    html_file_path = os.path.join(repo_path, 'index.html')

    print(f"📂 Repository path: {repo_path}")
    print(f"📄 HTML file path: {html_file_path}")

    # Fetch latest citation count
    citation_data = fetch_citation_data(scholar_id)

    # Update HTML file with new citation count
    update_citation_matrix(citation_data, html_file_path)

    # Commit & push changes to GitHub
    commit_and_push_changes(repo_path, '🔄 Auto-update citation matrix')
