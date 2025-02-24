from scholarly import scholarly
import os
import subprocess
import re

# Function to fetch citation count from Google Scholar
def fetch_citation_data(scholar_url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(scholar_url, headers=headers)

    if response.status_code != 200:
        raise Exception(f'Failed to fetch data from Google Scholar, Status Code: {response.status_code}')
    
    # Print the raw HTML to check if Google is blocking the request
    print("🔍 Raw HTML Content:")
    print(response.text[:1000])  # Print the first 1000 characters to check structure

    soup = BeautifulSoup(response.content, 'html.parser')
    citation_data = {}

    try:
        # Locate citation count
        citation_count_element = soup.find_all('td', class_='gsc_rsb_std')
        if citation_count_element:
            citation_count = citation_count_element[0].text.strip()
            citation_data['citation_count'] = citation_count
            print(f"✅ Extracted Citation Count: {citation_count}")
        else:
            print("⚠️ Could not find the citation count element!")
            citation_data['citation_count'] = "N/A"

    except Exception as e:
        print(f"❌ Error while parsing: {e}")
        citation_data['citation_count'] = "Error"

    return citation_data


# Function to update citation count in HTML file
def update_citation_matrix(citation_data, file_path):
    if not os.path.exists(file_path):
        print(f"❌ HTML file not found: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Use regex to replace only the number inside <span id="citation_count">
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
    scholar_id = "p6fjrJIAAAAJ&hl=en"   
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
