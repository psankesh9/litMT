from PIL import Image
import glob
import pytesseract
import os
from tqdm import tqdm


"""
An example of OCR to convert scanned kindle book pages to single text file.
Scans should be in single dir, with each book's scans in separate subdirs.
Result is dir with text files for each scanned book.

We used tesseract for OCR. Non fine-tuned tesseract produces text with some errors that need to be post-processed.
Other options for OCR include TrOCR and docTR.
"""

# directory where to save text files of scanned books
save_dir = 'russian_lit/ru_books_text_unprocessed'

# directory where all kindle book scans are stored
scans_dir = "/Users/kk/Desktop/book_scans"
book_subdirs = [name for name in os.listdir(scans_dir) if os.path.isdir(os.path.join(scans_dir, name))]


for book in book_subdirs:
    book_path = scans_dir + '/' + book
    book_text_file = save_dir + '/' + book + '.txt'
    
    if not os.path.isfile(book_text_file):
        print('OCRing ' + book, ':')
        text_list = []

        files = glob.glob(book_path + '/*.jpg')
        files_sorted = sorted(files, key=lambda x: int(x.partition('Page ')[2].partition('.jpg')[0]))

        for filename in tqdm(files_sorted):
            im = Image.open(filename)

            extracted_text = pytesseract.image_to_string(im)
            text_list.append(extracted_text)
            print('\n Scanned: ', filename)

        text = '\n'.join(text_list)
        with open(book_text_file, "w") as text_file:
            text_file.write(text)