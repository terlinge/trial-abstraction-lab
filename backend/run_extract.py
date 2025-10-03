import json
import os
import sys

# Run the extraction pipeline on the DASH PDF and write output to out_draft.json
repo_root = os.path.dirname(os.path.dirname(__file__))
pdf_path = os.path.join(repo_root, 'backend', 'data', 'uploads', '326b83b5-7598-4a75-8d1e-22ebbd26eb6c.pdf')
if not os.path.exists(pdf_path):
    # try alternate relative path
    pdf_path = os.path.join(repo_root, 'data', 'uploads', '326b83b5-7598-4a75-8d1e-22ebbd26eb6c.pdf')

print('PDF path:', pdf_path)
if not os.path.exists(pdf_path):
    print('ERROR: PDF not found at expected locations')
    sys.exit(2)

try:
    # Import the extractor (it uses optional libraries; in this env they should be present)
    from extractors import extract_first_pass
except Exception as e:
    print('Failed to import extractors:', repr(e))
    raise

print('Running extract_first_pass with force_ocr=True...')
draft = extract_first_pass(pdf_path, force_ocr=True)

out_path = os.path.join(os.path.dirname(__file__), 'out_draft.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(draft, f, indent=2, ensure_ascii=False)

print('Wrote draft to', out_path)
# Print brief summary
study = draft.get('study', {})
print('Title:', study.get('title'))
print('Authors:', study.get('authors'))
print('Flags:', draft.get('_flags'))
print('Tables metadata keys:', list(draft.get('_tables', {}).keys()) if draft.get('_tables') else None)
