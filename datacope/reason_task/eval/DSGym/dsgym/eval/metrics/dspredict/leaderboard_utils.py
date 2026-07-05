import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
import time
import litellm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class KaggleScraper:
    def __init__(self, timeout=30):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.timeout = timeout
        
    def setup_driver(self):
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(self.timeout)
        driver.implicitly_wait(10)
        return driver

    def extract_competition_name_from_url(self, url):
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')
        if len(path_parts) >= 2 and path_parts[0] == 'competitions':
            return path_parts[1]
        return None

    def clean_description_with_llm(self, raw_description):
        if not raw_description:
            return None
            
        prompt = f"""

You will be given in input a Kaggle competition description that has been scraped from the web. Clean this Kaggle competition description by removing scraper noise, HTML artifacts, navigation elements, and irrelevant text. Keep ALL important information including competition objective, dataset details, evaluation criteria, submission requirements, timeline, prizes, and any technical specifications. Only remove the noise:

<raw_description>
{raw_description}
</raw_description>

DO NOT ADD ANYTHING TO THE DESCRIPTION. ONLY REMOVE THE NOISE.

Cleaned description:"""

        response = litellm.completion(
            model="together_ai/Qwen/Qwen3-235B-A22B-Thinking-2507",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30000,
            temperature=0.1,
        )
        
        return response.choices[0].message.content.split("</think>")[1].strip()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=10),
        retry=retry_if_exception_type((TimeoutException, WebDriverException)),
        reraise=True
    )
    def scrape_page(self, url):
        driver = None
        try:
            print(f"Loading page with Selenium: {url}")
            driver = self.setup_driver()
            
            driver.get(url)
            time.sleep(20)
            
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            text_elements = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'span', 'section', 'article', 'td', 'th']
            all_texts = []
            for tag in text_elements:
                elements = soup.find_all(tag)
                for elem in elements:
                    text = elem.get_text(strip=True)
                    if text and len(text) > 10:
                        all_texts.append({
                            'tag': tag,
                            'text': text,
                            'length': len(text)
                        })
            
            print(f"  ✓ Successfully scraped {len(all_texts)} text elements")
            return all_texts
            
        finally:
            if driver:
                driver.quit()

    def scrape_overview_and_description(self, url):
        overview_texts = self.scrape_page(url + "/overview")
        overview_raw = "\n".join([item['text'] for item in overview_texts])
        overview_cleaned = self.clean_description_with_llm(overview_raw)
        data_texts = self.scrape_page(url + "/data")
        data_raw = "\n".join([item['text'] for item in data_texts])
        data_cleaned = self.clean_description_with_llm(data_raw)

        final_description = f"Challenge description:\n{overview_cleaned}\n\nData description:\n{data_cleaned}"
        return overview_cleaned, data_cleaned


def main():
    scraper = KaggleScraper()
    
    link = "https://www.kaggle.com/competitions/stanford-covid-vaccine"
    # link = "https://www.kaggle.com/competitions/playground-series-s5e5"
    
    print("=== Testing Kaggle Scraper ===")
    
    print("\n📝 Scraping Overview Page...")
    overview_texts = scraper.scrape_page(link + "/overview")
    
    print("\n📊 Scraping Data Page...")
    data_texts = scraper.scrape_page(link + "/data")
    
    print(f"\nOverview texts found: {len(overview_texts)}")
    overview_raw = "\n".join([item['text'] for item in overview_texts])
    print(f"Overview raw:\n{overview_raw}")
    overview_cleaned = scraper.clean_description_with_llm(overview_raw)
    print("\n📝 Overview Description:")
    print(overview_cleaned)
    
    # print(f"\nData texts found: {len(data_texts)}")
    # data_raw = "\n".join([item['text'] for item in data_texts])
    # data_cleaned = scraper.clean_description_with_llm(data_raw)
    # print("\n📊 Data Description:")
    # print(data_cleaned)
    
    print("\n=== Complete! ===")

if __name__ == "__main__":
    main()
