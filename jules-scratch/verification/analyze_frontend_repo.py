from playwright.sync_api import sync_playwright, expect, TimeoutError

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()

    try:
        print("Navigating to the frontend repository...")
        page.goto("https://github.com/Comfy-Org/ComfyUI_frontend", timeout=90000)

        print("Waiting for repository page to load...")
        expect(page.get_by_role("button", name="Go to file")).to_be_visible(timeout=60000)
        print("Repository page loaded. Scraping file and directory names...")

        file_and_dir_items = page.locator("div[role='gridcell'] a[class*='Link--primary']").all()
        file_names = [item.inner_text() for item in file_and_dir_items]

        print("\n--- Repository File List ---")
        for name in file_names:
            print(name)
        print("--------------------------\n")

        # --- Get package.json content ---
        print("Navigating to package.json...")
        page.get_by_role("link", name="package.json", exact=True).click()

        print("Waiting for package.json content to load...")
        json_container = page.locator("div.blob-wrapper")
        expect(json_container).to_be_visible(timeout=30000)
        package_json_content = json_container.inner_text()

        print("\n--- package.json Content ---")
        print(package_json_content)
        print("----------------------------\n")

        # Go back to the main page
        page.go_back()

        # --- Get vite.config.mts content ---
        print("Navigating to vite.config.mts...")
        expect(page.get_by_role("button", name="Go to file")).to_be_visible(timeout=60000)
        page.get_by_role("link", name="vite.config.mts", exact=True).click()

        print("Waiting for vite.config.mts content to load...")
        code_container = page.locator("div.blob-wrapper")
        expect(code_container).to_be_visible(timeout=30000)
        vite_config_content = code_container.inner_text()

        print("\n--- vite.config.mts Content ---")
        print(vite_config_content)
        print("-------------------------------\n")

        print("Analysis complete.")

    except Exception as e:
        print(f"An error occurred: {e}")
        page.screenshot(path="jules-scratch/verification/playwright_error.png")
        print("Screenshot taken of the error page.")

    finally:
        browser.close()

with sync_playwright() as playwright:
    run(playwright)
