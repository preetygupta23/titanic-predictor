import pytest
from datetime import datetime

def pytest_html_results_table_header(cells):
    cells.insert(2, "<th>Time</th>")

def pytest_html_results_table_row(report, cells):
    cells.insert(2, f"<td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>")

def pytest_configure(config):
    # This adds metadata to the top of the HTML report
    config._metadata['Project Name'] = 'Titanic Survival Predictor'
    config._metadata['Model Version'] = '1.0.0'
    config._metadata['QA Engineer'] = 'Gemini-Flash'