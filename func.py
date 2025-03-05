from lib import *


class DocumentWithMetadata:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata
def load_pdf(file_path):
  try:


    # Chuyển đổi PDF thành ảnh
    images = convert_from_path(file_path)
    metadata = {
              'source': file_path,    # Đường dẫn nguồn
          }

    pages = convert_from_path(file_path)
    
    full_text = ""
    
    for page in pages:
        # Sử dụng Tesseract để trích xuất văn bản từ hình ảnh
        text = pytesseract.image_to_string(page, lang = "vie")
        full_text += text
        
    document = DocumentWithMetadata(page_content=full_text, metadata=metadata)
    return [document]
  except Exception as e:
      print(f"Cannot load DOCX file: {file_path}, Error: {e}")
      return []

def load_docx(file_path):
    try:
        doc = Document(file_path)
        metadata = {
            'source': file_path,    # Đường dẫn nguồn
        }
        doc_text = ""

        # Read paragraphs (standard text)
        for para in doc.paragraphs:
            doc_text += para.text + "\n"

        # Read tables (if any)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    doc_text += cell.text + "\n"

        document = DocumentWithMetadata(page_content=doc_text, metadata=metadata)
        return [document]
    except Exception as e:
        print(f"Cannot load DOCX file: {file_path}, Error: {e}")
        return []

# Hàm để xử lý TXT
def load_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
          doc_text = file.read()
          metadata = {
            'source': file_path,    # Đường dẫn nguồn
        }
          document = DocumentWithMetadata(page_content=doc_text, metadata=metadata)
        return [document]
    except Exception as e:
        print(f"Cannot load TXT file: {file_path}, Error: {e}")
        return []
    
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embedding = HuggingFaceEmbeddings()
