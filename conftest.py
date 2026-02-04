import pytest
from datetime import datetime
import pytest
import pytest_html
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options


@pytest.fixture
def driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Essential: No GUI
    chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
    chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
    chrome_options.add_argument("--disable-gpu")  # Applicable to windows os only

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    yield driver
    driver.quit()

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hooks into the test result to capture screenshots on failure."""
    outcome = yield
    report = outcome.get_result()
    extras = getattr(report, "extras", [])

    if report.when == "call" and report.failed:
        # Check if the test has a 'driver' fixture
        if "driver" in item.fixturenames:
            driver = item.funcargs["driver"]
            screenshot = driver.get_screenshot_as_base64()
            # Embed the screenshot into the HTML report
            extras.append(pytest_html.extras.image(screenshot, "Failure Screenshot"))
            report.extras = extras


@pytest.fixture
def driver():
    """Setup for headless Chrome browser."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run without a visible window
    chrome_options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    yield driver
    driver.quit()

def pytest_html_results_table_header(cells):
    cells.insert(2, "<th>Time</th>")

def pytest_html_results_table_row(report, cells):
    cells.insert(2, f"<td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>")

def pytest_configure(config):
    # This adds metadata to the top of the HTML report
    config._metadata['Project Name'] = 'Titanic Survival Predictor'
    config._metadata['Model Version'] = '1.0.0'
    config._metadata['QA Engineer'] = 'Gemini-Flash'