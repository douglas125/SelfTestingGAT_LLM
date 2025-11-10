import os
import re
import pypdf
import base64
from pathlib import Path
from io import BytesIO

import pandas as pd
from markitdown import MarkItDown
from pdf2image import convert_from_path


def extract_text(file) -> str:
    file = Path(file)
    extension = file.suffix.lower()

    # Check the file type and read
    if extension in [
        ".txt",
        ".py",
        ".md",
        ".srt",
        ".js",
        ".jsx",
        ".html",
        ".css",
        ".xml",
    ]:
        with open(file, "r", encoding="utf-8") as f:
            ans = f.read()
        ans = f"<contents>\n{ans}\n</contents>"
    elif extension in [".docx", ".pptx", ".xlsx", ".xls"]:
        md = MarkItDown()
        ans = md.convert(file).text_content
    elif extension == ".pdf":
        ans = pdf_to_xml(file)
    elif extension == ".csv":
        ans = csv_to_xml(file)
    else:
        raise ValueError(f"Unsupported file extension: {extension}")

    return ans


def pdf_to_xml(pdf_path):
    # Open the PDF
    pdf = pypdf.PdfReader(pdf_path)
    xml_content = ["<contents>"]

    # Loop through pages and extract text
    for page_num, page in enumerate(pdf.pages, start=1):
        text = page.extract_text()
        xml_content.append(f"<page_{page_num}>{text}</page_{page_num}>")

    xml_content.append("</contents>")
    return "\n".join(xml_content)


def pdf_pages_to_base64_images(pdf_path, dpi=200, fmt="JPEG", jpeg_quality=95):
    """
    Convert each page of a PDF into a Base64-encoded image.

    Args:
        pdf_path (str): Path to the PDF file.
        dpi (int): Resolution of output images (higher dpi = better quality).
        fmt (str): Image format, e.g., 'PNG', 'JPEG'.

    Returns:
        List[str]: Base64-encoded strings of each page image.
    """
    images = convert_from_path(pdf_path, dpi=dpi)
    base64_images = []

    for img in images:
        buffered = BytesIO()
        img.save(buffered, format=fmt, quality=jpeg_quality)
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_images.append(img_base64)

    return base64_images


def sanitize_column_name(name):
    """
    Convert column names to valid XML tag names by:
    - Removing or replacing invalid characters
    - Replacing whitespace with underscores
    - Removing carriage return and newline characters
    """
    name = name.strip().replace("\r", "").replace("\n", "")
    name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
    if name[0].isdigit():
        name = f"col_{name}"
    return name


def csv_to_xml(csv_path):
    # Try different encodings and delimiter if necessary
    for encoding in ["utf-8", "latin-1", "utf-16"]:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            # Try with semicolon as delimiter if parsing fails
            try:
                df = pd.read_csv(csv_path, encoding=encoding, delimiter=";")
                break
            except Exception:
                continue
    else:
        raise ValueError(
            "Unable to read the CSV file with common encodings and delimiters."
        )

    # Remove unnamed columns and sanitize column names
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
    df.columns = [sanitize_column_name(col) for col in df.columns]

    # Convert DataFrame to XML
    xml_data = df.to_xml(index=False, root_name="data", row_name="record")
    return xml_data


def normalize_xml_content(xml_content):
    # Split into lines, strip trailing whitespace from each, then rejoin
    xml_content = "\n".join(line.rstrip() for line in xml_content.splitlines())

    # Add EOF
    xml_content += "\n"
    return xml_content


class ToolReadLocalFile:
    def __init__(self, query_llm=None):
        self.name = "read_local_files"
        self.query_llm = query_llm

        self.tool_description = {
            "name": self.name,
            "description": """Reads one or more files and return their contents. Provide one file per line.
Only read files of type <allowed_extensions></allowed_extensions>:
<allowed_extensions>
pdf
docx
txt
md
pptx
xlsx
xls
py
srt
csv
</allowed_extensions>

Do not attempt to read files outside the types described in the <allowed_extensions></allowed_extensions>.
Do not attempt to read files that are usually in binary format.

Raises ValueError: if the file does not exist.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path_to_files": {
                        "type": "string",
                        "description": """Local path to the files whose contents should be retrieved. Provide one file per line as in the <example></example>:
<example>
file1.pdf
file2.docx
subfolder/file3.txt
</example>""",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Prompt that can be used to directly query and retrieve information from the files, without fetching their full content.",
                    },
                    "pdfs_to_read_as_images": {
                        "type": "string",
                        "description": """Local path of PDF files that need to be read as images.
The following <use_case_examples></use_case_examples> provide some examples of use cases that require analyzing the PDF as images:
<use_case_examples>
<use_case_example>
Automated proofreading: Identify graphs, diagrams, or handwritten texts drawn.
</use_case_example>
<use_case_example>
Recognition of mathematical formulas: Extract handwritten equations or scientific diagrams.
</use_case_example>
<use_case_example>
Analysis of academic works: Check figures, tables, and graphs.
</use_case_example>
</use_case_examples>
Provide this parameter only after analyzing whether there is a need to read images from the PDF, based on the user's query. This is important because parsing images is expensive and we do not want to do it unnecessarily.

Provide one file per line as in the <paths_example></paths_example>:
<paths_example>
file1.pdf
subfolder/file2.pdf
</paths_example>""",
                    },
                },
                "required": ["path_to_files", "prompt"],
            },
        }

    def __call__(self, path_to_files, prompt="", pdfs_to_read_as_images="", **kwargs):
        if len(kwargs) > 0:
            yield f"Error: Unexpected parameter(s): {','.join([x for x in kwargs])}"
            return

        # make sure all pdfs to read as images are in the file list
        for x in pdfs_to_read_as_images:
            if x not in path_to_files:
                path_to_files.append(x)

        final_ans = ["<files>"]
        all_files = [x.strip() for x in path_to_files.splitlines() if x.strip() != ""]
        pdfs_to_read_as_images = [
            x.strip() for x in pdfs_to_read_as_images.splitlines() if x.strip() != ""
        ]
        b64_images = None

        for path_to_file in all_files:
            ans = ""
            if not os.path.isfile(path_to_file):
                ans = f"Error: Did not find file `{path_to_file}`"
                yield f"<scratchpad>{ans}</scratchpad>"
                ans = f"<error>\n{ans}\n</error>"
            else:
                try:
                    yield f"<scratchpad>Reading {path_to_file}</scratchpad>"
                    ans = extract_text(path_to_file)
                    if path_to_file in pdfs_to_read_as_images:
                        if b64_images is None:
                            b64_images = []
                        b64_images += pdf_pages_to_base64_images(path_to_file)
                except Exception as e:
                    ans = (
                        f"Error: Failed to process the file `{path_to_file}`: {str(e)}"
                    )
                    yield f"<scratchpad>{ans}</scratchpad>"
                    ans = f"<error>\n{ans}\n</error>"

            final_ans.append("<file>")
            final_ans.append(f"<file_name>{path_to_file}</file_name>")
            final_ans.append(ans)
            final_ans.append("</file>")

        final_ans.append("</files>")
        final_ans = "\n".join(final_ans)

        # if a subquery has been asked
        if prompt is not None and prompt != "":
            if self.query_llm is None:
                yield "Error: Cannot retrieve data from the document because a LLM has not been provided. Please set prompt to '' to return the full document."
                return
            sys_prompt = "Read the contents of the following <files></files> to answer questions:\n"
            sys_prompt += final_ans
            llm_ans = self.query_llm(
                prompt,
                b64images=b64_images,
                system_prompt=sys_prompt,
            )
            for x in llm_ans:
                yield f"<scratchpad>{x}</scratchpad>"
            final_ans = x

        final_ans = normalize_xml_content(final_ans)

        yield final_ans
