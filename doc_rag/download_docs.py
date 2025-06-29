import requests
from bs4 import BeautifulSoup
import os
import urllib
from dotenv import load_dotenv

load_dotenv("../env")
# The URL to scrape
url = "https://python.langchain.com/api_reference/"
 
# The directory to store files in
output_dir = "./langchain-docs/"
 
# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
 
# Fetch the page
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
 
# Find all links to .html files
links = soup.find_all('a', href=True)

for link in links:
    href = link['href']
    
    # If it's a .html file
    if href.endswith('.html'):
        # Make a full URL if necessary
        
        if not href.startswith('http'):
            href = urllib.parse.urljoin(url, href)
        print(href)    
        # Fetch the .html file
        file_response = requests.get(href)
        
        # Write it to a file
        # Extract path components and create custom filename
        parsed_url = urllib.parse.urlparse(href)
        path_parts = parsed_url.path.strip('/').split('/')
        if len(path_parts) >= 2:
            # Create filename like: anthropic_chat_models.html
            custom_filename = f"{path_parts[-2]}_{path_parts[-1]}"
        else:
            custom_filename = os.path.basename(href)
        
        file_name = os.path.join(output_dir, custom_filename)
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(file_response.text)

