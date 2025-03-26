from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np 

# Load the sentiment analysis model
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Function to get sentiment score
def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits)) + 1

# Get comments off Yelp - Scraper code
def scrape_yelp_reviews(url, max_pages=5):
    reviews = []
    current_page = 1
    
    while current_page <= max_pages:
        print(f'Scraping page {current_page}...')
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')

        # Adjust the regex as needed based on the actual class names
        regex = re.compile('.*comment.*')
        results = soup.find_all('p', {'class': regex})
        
        # Collect reviews
        for result in results:
            reviews.append(result.text)
        
        # Check for the next page
        next_button = soup.find('a', class_='next-link navigation-button__09f24__m9qRz', attrs={'aria-label': 'Next'})
        if next_button and 'href' in next_button.attrs:
            next_url = next_button['href']
            url = next_url  # Update to the next page URL
        else:
            break  # Exit loop if no next page is found
        
        current_page += 1

    return reviews

# Scrape reviews from the Yelp page
url = 'https://www.yelp.com/biz/international-smoke-san-francisco-15'
reviews = scrape_yelp_reviews(url, max_pages=5)

# Create DataFrame
df = pd.DataFrame(np.array(reviews), columns=['review'])

# Calculate sentiment for each review
df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x[:512]))

# Print the DataFrame and save to CSV
print(df)
df.to_csv('intsmoke.csv', index=False)
