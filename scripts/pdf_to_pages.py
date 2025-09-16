# pdf_to_pages.py
import sys, json
import pdfplumber  # pip install pdfplumber

def pdf_to_pages(pdf_path: str) -> list[str]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, p in enumerate(pdf.pages, start=1):
            text = p.extract_text() or ""
            # Normalise: remove double spaces, keep line breaks
            text = " ".join(text.split())
            pages.append(text)
    return pages

def write_txt(pages: list[str], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for i, page in enumerate(pages, start=1):
            f.write(f"<<<PAGE {i}>>>\n{pages[i-1]}\n\n")

def write_json(pages: list[str], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python pdf_to_pages.py input.pdf output.(txt|json)")
        sys.exit(1)

    pdf_path, out_path = sys.argv[1], sys.argv[2]
    pages = pdf_to_pages(pdf_path)

    if out_path.endswith(".txt"):
        write_txt(pages, out_path)
    elif out_path.endswith(".json"):
        write_json(pages, out_path)
    else:
        print("Output must end with .txt or .json")
        sys.exit(1)

    print(f"Extracted {len(pages)} pages → {out_path}")
