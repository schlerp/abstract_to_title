import os
from typing import Dict, List, Tuple
import lxml.html
import lxml.html.soupparser
import json

import requests


class Crawler(object):
    def __init__(
        self,
        output_path: str,
        base_url: str = "https://arxiv.org",
        include_url_pattern: str = "https://arxiv.org",
        process_pattern: str = "https://arxiv.org/abs",
        # exclude_url_patterns: List[str] = [],
        scrape_definitions: Dict = {
            "id": "/html/head/meta[@name='citation_arxiv_id']/@content",
            "title": "/html/head/meta[@name='citation_title']/@content",
            "abstract": "/html/head/meta[@name='citation_abstract']/@content",
            "date": "/html/head/meta[@name='citation_date']/@content",
        },
    ):
        self.yet_to_crawl = set()
        self.already_crawled = set()
        self.output_path = output_path
        self.base_url = base_url
        self.process_pattern = process_pattern
        self.include_url_pattern = include_url_pattern
        # self.exclude_url_patterns = exclude_url_patterns
        self.scrape_definitions = scrape_definitions

    def crawl(self, start_url: str):
        self.yet_to_crawl.add(start_url)
        while self.yet_to_crawl:
            this_url = self.yet_to_crawl.pop()
            print("processing {}...".format(this_url))
            content, urls = self.process_url(this_url)
            if self.process_pattern in this_url:
                self.process_content(this_url, content)

            for url in urls:
                if (
                    url not in self.already_crawled
                    and self.include_url_pattern in url
                    # and url not in self.exclude_url_patterns
                ):
                    self.yet_to_crawl.add(url)
                    print("  added url to crawl {}".format(url))

    def process_url(self, url: str) -> Tuple[str, List[str]]:
        response = requests.get(url, timeout=5, verify=False)
        urls = []
        html = lxml.html.HtmlElement()
        if response.status_code == 200:
            self.already_crawled.add(url)
            content = response.text
            html = lxml.html.fromstring(content.encode("utf8"))
            for a in html.xpath("//a/@href"):
                if a[0:4] != "http":
                    urls.append(f"{self.base_url}{a}")
                else:
                    urls.append(a)
        return html, urls

    def process_content(self, url, html):
        content_dict = {
            key: html.xpath(xpath_sel)[0]
            for key, xpath_sel in self.scrape_definitions.items()
        }
        self.output_content_dict(content_dict)

    def output_content_dict(self, content_dict: Dict):
        output_filename = os.path.join(
            self.output_path, "{}.json".format(content_dict["id"])
        )
        print("outputting content_dict to {}".format(output_filename))
        with open(output_filename, "w+") as f:
            json.dump(content_dict, f)


if __name__ == "__main__":
    crawler = Crawler(
        output_path="/tmp",
    )
    crawler.crawl("https://arxiv.org/list/math/recent")
