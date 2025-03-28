import re
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup, NavigableString, Tag

from bs4.element import Comment


def extract_visible_html(html):
    soup = BeautifulSoup(html, "html.parser")

    # Remove script, style, meta, head, link, hidden, aria-hidden, and other non-visible elements
    for hidden in soup(
        ["script", "style", "meta", "head", "link", "[hidden]", '[aria-hidden="true"]']
    ):
        hidden.decompose()

    # Remove elements with inline styles that hide them
    for element in soup.find_all(True):
        style = element.get("style", "")

        # Check for display: none, visibility: hidden, or opacity: 0 in the style attribute
        if (
            "display: none" in style
            or "visibility: hidden" in style
            or "opacity: 0" in style
        ):
            element.decompose()

    # Remove comment nodes
    for comment in soup.findAll(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Loop through all elements and remove tags that are empty or hidden
    for element in soup.find_all(True):
        if not element.get_text(strip=True):  # Remove elements without visible text
            element.decompose()

    # Return the modified HTML as a string
    return soup.prettify()


def get_text_and_urls(url_content):
    # Parse the HTML content
    soup = BeautifulSoup(url_content, "html.parser")

    # Find all <a> tags and replace them with formatted URLs
    for link in soup.find_all("a"):
        href = link.get("href")
        text = link.get_text()
        new_tag = soup.new_tag("a", href=href)
        new_tag.string = text
        link.replace_with(new_tag)

    # Extract the modified HTML content
    output = str(soup)
    return output


def tag_visible(element):
    if element.parent.name in ["a", "p"]:
        return True

    if element.parent.name in [
        "style",
        "script",
        "head",
        "title",
        "meta",
        # "[document]",
    ]:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = MyBeautifulSoup(body, "html.parser")
    texts = soup.findAll(string=True)
    visible_texts = filter(tag_visible, texts)

    url_list = []
    for link in soup.find_all("a"):
        url_list.append(f"[ {link.get('href')} ] {link.text}")
        # print(link.get('href'))

    return " ".join(t.strip() for t in visible_texts), "\n".join(url_list)


# https://stackoverflow.com/questions/52026274/beautiful-soup-get-all-text-but-preserve-link-html
class MyBeautifulSoup(BeautifulSoup):
    def get_text(self, separator="", strip=False, types=(NavigableString,)):
        text_parts = []

        for element in self.descendants:
            if isinstance(element, NavigableString):
                text_parts.append(str(element))
            elif isinstance(element, Tag):
                if element.name == "a" and "href" in element.attrs:
                    text_parts.append(
                        element.get_text(separator=separator, strip=strip)
                    )
                    text_parts.append("( " + element["href"] + " )")
                elif isinstance(element, types):
                    text_parts.append(
                        element.get_text(separator=separator, strip=strip)
                    )

        return separator.join(text_parts)


class ToolGetUrlContent:
    def __init__(self, query_llm, max_subpages_to_read=60):
        self.name = "get_url_content"
        self.query_llm = query_llm
        self.max_subpages_to_read = max_subpages_to_read

        self.tool_description = {
            "name": self.name,
            "description": f"""Retrieves the contents of one or more internet URLs specified by internet_urls, separated by commas. For example, use this tool when the user requests you to read a webpage, or find something on the internet. A prompt should be provided to extract only the desired relevant content from the URLs. If the prompt is left empty, the raw contents of the webpage are returned. Note that the prompt will be provided individually to query each URL.

You can use {self.name} in these <use_cases></use_cases> and others as needed when:
<use_cases>
<use_case>It is easy to retrieve up-to-date information from a webpage</use_case>
<use_case>It is necessary to navigate internet web pages to retrieve information</use_case>
<use_case>The user wants to browse websites containing news. In this case, follow the steps:

- Read the initial URL using recursion_level=0 to find the URLs to the webpages linked.
- Find a pattern that will retrieve only the relevant URLs. In general, strings like "/news/" or "/economy/" will be contained in the URL.
- Read the initial URL using recursion_level=1 and the appropriate recursion_regex_condition to retrieve only relevant news.
</use_case>
</use_cases>

Raises ValueError: if the request is invalid.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "internet_urls": {
                        "type": "string",
                        "description": """Website URLs separated by commas, as in the <url_examples></url_examples>:
<url_examples>
<url_example>http://www.folha.com</url_example>
<url_example>http://www.g1.com, https://www.walljournal.com, http://websitename.com.uk</url_example>
</url_examples>""",
                    },
                    "prompt": {
                        "type": "string",
                        "description": """Query prompt to retrieve specific information from a single web page. For example:
<website_prompt_examples>
<website_prompt_example>Summarize the contents of this web page. Make sure to include all links in the page.</website_prompt_example>
<website_prompt_example>Read this web page and extract only the URLs related to economic news.</website_prompt_example>
</website_prompt_examples>
Leave argument prompt empty ("") to retrieve the raw text.
""",
                    },
                    "return_all_visible_html": {
                        "type": "string",
                        "enum": ["True", "False"],
                        "description": """Allowed choices are True and False.
If True, returns all raw HTML that contains visible elements.
If False, only returns the visible text of the website and a list of URLs found.""",
                    },
                    "recursion_level": {
                        "type": "integer",
                        "default": 0,
                        "description": """Read all links from the original webpage recursion_level times. Default is 0.
If recursion_level is greater than 0, also read and return the contents of links found in the webpages.
This is useful to automatically follow links and retrieve detailed contents from pages.
For example: if the user requests reading news from a webpage, setting recursion_level to 1 will usually also retrieve the contents of the articles.
""",
                    },
                    "recursion_regex_condition": {
                        "type": "string",
                        "description": """If recursion_level is greater than 0, child pages must match this regex to be loaded.
Use this parameter to avoid reading an excessive amount of useless data in subpaged.
Note that the webpage URL will be checked using python's regex re package using: re.match(recursion_regex_condition, URL)
""",
                    },
                },
                "required": ["internet_urls"],
            },
        }

    def __call__(
        self,
        internet_urls,
        prompt="",
        return_all_visible_html=False,
        recursion_level=0,
        recursion_regex_condition="",
        **kwargs,
    ):
        if len(kwargs) > 0:
            return f"Error: Unexpected parameter(s): {','.join([x for x in kwargs])}"

        return_all_visible_html = str(return_all_visible_html).lower().strip() == "true"
        recursion_level = int(recursion_level)
        internet_urls = internet_urls.split(",")
        internet_urls = [x.strip() for x in internet_urls]
        ans = []
        for u in internet_urls:
            ans.append("<url_content>")
            ans.append(f"<url>{u}</url>")
            content = self._get_url_content(
                u,
                return_all_visible_html,
                max_recursion_level=recursion_level,
                recursion_regex_condition=recursion_regex_condition,
            )
            if prompt.strip() != "":
                if self.query_llm is None:
                    content = "Could not answer prompt because a Language Model was not provided."
                else:
                    sys_prompt = "Read the contents of the following webpage to answer questions:"
                    sys_prompt = (
                        sys_prompt + f"\n<webpage_contents>{content}<webpage_contents>"
                    )
                    sys_prompt = (
                        sys_prompt + f"\nAlways include relevant links in your answers."
                    )
                    llm_ans = self.query_llm(
                        prompt,
                        system_prompt=sys_prompt,
                    )
                    for x in llm_ans:
                        pass
                    content = x

            ans.append(f"<content>{content}</content>")
            ans.append("</url_content>")
        return "\n".join(ans)

    def _get_url_content(
        self,
        internet_url,
        return_all_visible_html,
        cur_recursion_level=0,
        max_recursion_level=0,
        recursion_regex_condition="",
    ):
        """Retrieves URL contents

        Arguments:
            internet_url: URL to read
            return_all_visible_html: whether to return all HTML that renders visible elements or just the text
            cur_recursion_level: current recursion level, for cases when subpages need to be retrieved
            max_recursion_level: maximum recursion level to go to
        """
        # keep track of pages already visited to avoid looping
        if cur_recursion_level == 0:
            self.retrieved_pages = {internet_url: True}

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"
            }
            c = requests.get(internet_url, headers=headers)

            if str(return_all_visible_html).lower().strip() == "true":
                # return all visible HTML (remove only scripts and hidden elements)
                visible_html = extract_visible_html(c.content)
                ans = f"<source_url>{c.url}</source_url><status_code>{c.status_code}</status_code>\n<contents>{visible_html}</contents>"
            else:
                # only extract texts and URLs
                texts, urls = text_from_html(c.text)
                # don't return URLs if navigating sub-URLs
                if max_recursion_level == 0:
                    ans = f"<source_url>{c.url}</source_url><status_code>{c.status_code}</status_code>\n<contents>{texts}</contents><urls>{urls}</urls>"
                else:
                    ans = f"<source_url>{c.url}</source_url><status_code>{c.status_code}</status_code>\n<contents>{texts}</contents>"

            # if the user requested sublinks:
            if cur_recursion_level < max_recursion_level:
                sub_links = self._extract_links(c.content)
                sub_links = [
                    x
                    for x in sub_links
                    if (
                        re.match(recursion_regex_condition, x)
                        or recursion_regex_condition in x
                    )
                ]
                assert (
                    len(sub_links) < self.max_subpages_to_read
                ), f"""Error: tried to read too many sublinks: {len(sub_links)}.
The maximum allowed is {self.max_subpages_to_read}.
Please narrow down the number of webpages using a recursion_regex_condition.
Tried to read the following pages:
{sub_links}
"""
                sub_contents = []
                for sub_link in sub_links:
                    if not self.retrieved_pages.get(sub_link, False):
                        sub_content = self._get_url_content(
                            sub_link,
                            return_all_visible_html,
                            cur_recursion_level + 1,
                            max_recursion_level,
                        )
                        self.retrieved_pages[sub_link] = True
                        sub_contents.append(sub_content)
                ans = ans + "\n" + "\n".join(sub_contents)

            return ans
        except Exception as e:
            return f"Could not retrieve page from URL {internet_url}.\nError description: {str(e)}"

        return str(ans)

    def _extract_links(self, html_text, base_url=None):
        """
        Extracts all URLs from the given HTML text.

        Parameters:
            html_text (str): The HTML content to parse.
            base_url (str): The base URL to resolve relative links (optional).

        Returns:
            list: A list of all extracted URLs.
        """
        soup = BeautifulSoup(html_text, "html.parser")
        links = []

        # Find all 'a' tags with an 'href' attribute
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if base_url:
                # Resolve relative links against the base URL
                href = urljoin(base_url, href)
            links.append(href)

        return [x.strip() for x in links]
