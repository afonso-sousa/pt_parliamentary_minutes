import argparse
import urllib.request
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
import time

def save_pdf(href, save_path):
    response = urllib.request.urlopen(href)
    file = open(save_path, "wb")
    file.write(response.read())
    file.close()
    print(f"Saved file at: {save_path.name}")


def download_session_pdfs(driver, leg, session, save_path):
    # Get list of divs with content
    results = driver.find_element(
        By.ID, "ctl00_ctl50_g_aa94ec59_77a0_4cf6_bfdd_99098fef46d5_ctl00_pnlResults"
    )
    children = results.find_elements(
        By.XPATH, "//div[@class='row margin_h0 margin-Top-15']"
    )
    children = children[2:]

    for child in children:
        element = child.find_element(By.TAG_NAME, "a")
        num = element.text.split()[-1]
        href = element.get_attribute("href")
        pdf_save_path = save_path / f"dar_serie_I_{leg}_{session}_{num}.pdf"
        save_pdf(href, pdf_save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Download parliamentary minutes")
    parser.add_argument(
        "--leg", type=str, default="XIII", help="session number",
    )
    parser.add_argument(
        "--session", type=int, nargs="?", choices=[1, 2, 3, 4], help="session number",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    print(args)

    # Setup
    dir_root_path = Path(__file__).parent.resolve()
    # driver_path = Path(dir_root_path.parent.parent, "chromedriver")
    # print(f"Driver path is set to:\n{driver_path}")

    save_path = dir_root_path / f"data/pdf_minutes/{args.leg}"
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Save path is set to:\n{save_path}")

    options = Options()
    options.add_argument("--headless")
    prefs = {
        "plugins.plugins_list": [
            {"enabled": False, "name": "Chrome PDF Viewer"}
        ],  # Disable Chrome's PDF Viewer
        "download.default_directory": dir_root_path.as_posix(),
        "download.extensions_to_open": "applications/pdf",
    }
    options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )

    url = f"https://www.parlamento.pt/DAR/Paginas/DAR1Serie.aspx"
    driver.get(url)

    time.sleep(2)

    # Select legislature option
    leg_selector_name = (
        "ctl00$ctl50$g_aa94ec59_77a0_4cf6_bfdd_99098fef46d5$ctl00$ddlLegislatura"
    )
    try:
        leg_selector_elem = WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.NAME, leg_selector_name,))
        )
    finally:
        leg_select = Select(
            leg_selector_elem
        )
        for option in leg_select.options:
            leg = option.text.split()[0]
            if leg == args.leg:
                leg_select.select_by_visible_text(option.text)
                break

    time.sleep(2)

    # Select session option
    if args.session:
        session_selector_name = (
            "ctl00$ctl50$g_aa94ec59_77a0_4cf6_bfdd_99098fef46d5$ctl00$ddlSessaoLegislativa"
        )
        try:
            session_selector_elem = WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((By.NAME, session_selector_name,))
            )
        finally:
            session_select = Select(
                session_selector_elem
            )
            for option in session_select.options:
                selected_session = option.text.split()[0]
                selected_session = selected_session.split(".")[0]
                if selected_session == args.session:
                    session_select.select_by_visible_text(option.text)
                    break

        download_session_pdfs(driver, args.leg, selected_session, save_path)

    else:
        session_selector_name = (
            "ctl00$ctl50$g_aa94ec59_77a0_4cf6_bfdd_99098fef46d5$ctl00$ddlSessaoLegislativa"
        )
        try:
            session_selector_elem = WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((By.NAME, session_selector_name,))
            )
        finally:
            session_select = Select(
                session_selector_elem
            )
            num_sessions = len(session_select.options)
            print(f"Downloading all {num_sessions} sessions")
            for session in list(range(1, len(session_select.options) + 1)):
                for option in session_select.options:
                    selected_session = option.text.split()[0]
                    selected_session = selected_session.split(".")[0]
                    if selected_session == args.session:
                        session_select.select_by_visible_text(option.text)
                        break
                download_session_pdfs(driver, args.leg, session, save_path)
