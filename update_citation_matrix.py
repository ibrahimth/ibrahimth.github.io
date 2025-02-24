import requests
from bs4 import BeautifulSoup
import os
import subprocess

# Function to fetch citation data from Google Scholar
def fetch_citation_data(scholar_url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(scholar_url, headers=headers)
    if response.status_code != 200:
        raise Exception('Failed to fetch data from Google Scholar')
    
    soup = BeautifulSoup(response.content, 'html.parser')
    citation_data = {}
    
    # Example: Parse the citation count
    citation_count = soup.find('td', class_='gsc_rsb_std').text
    citation_data['citation_count'] = citation_count
    
    return citation_data

# Function to update the citation matrix in the HTML file
def update_citation_matrix(citation_data, file_path):
    with open(file_path, 'r') as f:
        html_content = f.read()
    
    new_html_content = html_content.replace(
        'Citations: <span id="citation_count">0</span>',
        f'Citations: <span id="citation_count">{citation_data["citation_count"]}</span>'
    )
    
    with open(file_path, 'w') as f:
        f.write(new_html_content)

# Function to commit and push changes to GitHub
def commit_and_push_changes(repo_path, commit_message):
    os.chdir(repo_path)
    subprocess.run(['git', 'add', '.'])
    subprocess.run(['git', 'commit', '-m', commit_message])
    subprocess.run(['git', 'push'])

if __name__ == "__main__":
    scholar_url = 'https://scholar.google.com/citations?user=p6fjrJIAAAAJ&hl=en'
    repo_path = ''  # This is the default path for GitHub Actions
    html_file_path = os.path.join(repo_path, 'index.html')
    
    citation_data = fetch_citation_data(scholar_url)
    update_citation_matrix(citation_data, html_file_path)
    commit_and_push_changes(repo_path, 'Update citation matrix')
