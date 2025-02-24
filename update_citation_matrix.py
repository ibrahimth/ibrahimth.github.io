from scholarly import scholarly
import os
import subprocess

# Function to fetch citation count from Google Scholar
def fetch_citation_data(scholar_profile_id):
    try:
        author = scholarly.search_author_id(scholar_profile_id)
        author = scholarly.fill(author, sections=["basics", "indices"])
        citation_count = author.get("citedby", 0)  # Get citation count or default to 0
        return {"citation_count": citation_count}
    except Exception as e:
        print(f"Error fetching citation data: {e}")
        return {"citation_count": 0}  # Return 0 if fetching fails

# Function to update citation count in the HTML file
def update_citation_matrix(citation_data, file_path):
    if not os.path.exists(file_path):
        print(f"HTML file not found at path: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    new_html_content = html_content.replace(
        'Citations: <span id="citation_count">0</span>',
        f'Citations: <span id="citation_count">{citation_data["citation_count"]}</span>'
    )

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_html_content)

    print(f"Updated citation count to {citation_data['citation_count']} in {file_path}")

# Function to commit & push changes to GitHub
def commit_and_push_changes(repo_path, commit_message):
    os.chdir(repo_path)
    subprocess.run(['git', 'add', '.'])
    subprocess.run(['git', 'commit', '-m', commit_message])
    subprocess.run(['git', 'push'])

if __name__ == "__main__":
    scholar_id = "p6fjrJIAAAAJ"  # Replace with the correct Google Scholar ID
    repo_path = os.getenv('GITHUB_WORKSPACE', os.getcwd())  # GitHub Actions repo path
    html_file_path = os.path.join(repo_path, 'index.html')

    print(f"Repository path: {repo_path}")
    print(f"HTML file path: {html_file_path}")

    citation_data = fetch_citation_data(scholar_id)
    update_citation_matrix(citation_data, html_file_path)
    commit_and_push_changes(repo_path, 'Update citation matrix')
