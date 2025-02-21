from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader
import os


def load_file(file_path):
    # Get the file extension
    _, file_extension = os.path.splitext(file_path)

    # Call the appropriate loader function based on the file extension
    if file_extension.lower() == '.pdf':
        return pdf_loader(file_path)
    elif file_extension.lower() == '.docx':
        return doc_loader(file_path)


def doc_loader(file):
    loader = UnstructuredFileLoader(file_path=file)
    docs = loader.load_and_split()

    return docs


def pdf_loader(file):
    loader = PyPDFLoader(file_path=file)
    docs = loader.load_and_split()

    return docs
